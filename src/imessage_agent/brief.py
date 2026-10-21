"""Daily culture brief: what people are joking about online, for the bot's prompt.

Once a day (BRIEF_TIME, local) the brief is rebuilt from one of three sources:
  * web   - the task agent searches the web, with the same fenced toolset as /task and
            a generic prompt (no chat content is sent);
  * x     - the brief topics are pulled from the X API (xsearch.py) and the most-engaged
            posts are written up by the cloud model, or the local model if there is no
            Anthropic key;
  * both  - the X posts are handed to the task agent as grounding and it may web-search
            to fill in what they refer to.
The result is 10-15 one-line references, stored as a 0600 markdown file the owner can
edit in the dashboard, and added to the bot's system prompt while it's fresh.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from .logsafe import exc_name, log_event
from .task import TaskResult

STALE_AFTER_HOURS = 72
MAX_LINE = 180
DEFAULT_TOPICS = (
    "New York sports (Jets, Giants, Knicks, Nets, Yankees, Mets, Rangers, Islanders), "
    "NFL Twitter and NBA Twitter, and general internet culture and memes"
)

_FORMAT = (
    "Final answer: ONLY the brief, 10 to 15 lines, each in the form\n"
    "- <the reference, phrase or moment>: <one short line on what it is and why people are joking "
    "about it>\nEach line under 160 characters. No intro, no outro, no links, no headings, nothing "
    "about yourself."
)

BRIEF_PROMPT = (
    "You compile a daily culture brief for a group of friends' group-chat buddy, so he's up on what "
    "people are talking and joking about. Focus: {topics}.\n"
    "Search for what's trending and being joked about in the last two or three days: what's blowing "
    "up on X/Twitter, TikTok, Reddit and sports media, viral moments, memes and catchphrases, "
    "sports drama, trades, meltdowns and hot takes. Prefer specific names, phrases and moments over "
    "generic topics. Skip deaths, disasters and violent crime.\n" + _FORMAT
)

# Added to BRIEF_PROMPT (source=both) after the X digest is pulled.
X_GROUNDING = (
    "Start from the posts below: the most-engaged posts on X in the last {hours} hours for each "
    "topic, with like and repost counts. Use web search to work out what the ones you don't "
    "recognise refer to, and to catch anything big they miss. Ignore ads, giveaways, spam and "
    "engagement bait.\n\n{digest}"
)

# source=x: no web search, the writer only sees the digest.
X_WRITER_SYSTEM = (
    "You compile a daily culture brief for a group of friends' group-chat buddy, so he's up on what "
    "people are talking and joking about. Focus: {topics}.\n"
    "You are given the most-engaged posts on X in the last {hours} hours for each topic, with like "
    "and repost counts. From them, and only them, pick the moments, names, phrases and running jokes "
    "people are repeating. Prefer specific over generic; higher counts matter more. Ignore ads, "
    "giveaways, spam, engagement bait and anything you can't make sense of. Skip deaths, disasters "
    "and violent crime.\n" + _FORMAT
)


class CultureBrief:
    def __init__(self, path: str, runner=None, topics: str = "", at: str = "08:00", max_chars: int = 2000,
                 clock=time.time, x=None, source: str = "web", writer=None):
        self.path = Path(path).expanduser()
        self.runner = runner
        self.topics = topics.strip() or DEFAULT_TOPICS
        self.at = at
        self.max_chars = max_chars
        self._clock = clock
        self._lock = asyncio.Lock()
        self.last_error: str | None = None
        # X pulls (xsearch.XSearch) and how they're used; see the module docstring.
        self.x = x
        self.source = source if x is not None else "web"
        # source=x only: anything with .reply(system, messages) -> str and .name.
        self.writer = writer
        self.last_cost_usd: float = 0.0

    # -- storage --
    def age_hours(self) -> float | None:
        try:
            return (self._clock() - self.path.stat().st_mtime) / 3600
        except FileNotFoundError:
            return None

    def raw(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            log_event("brief.read_failed", error=exc_name(exc))
            return ""

    def load(self) -> str:
        """The brief for the prompt, or '' if missing or stale."""
        age = self.age_hours()
        if age is None or age > STALE_AFTER_HOURS:
            return ""
        return self.raw().strip()

    def save(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".brief-", suffix=".md", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- generation --
    @staticmethod
    def clean(text: str) -> str:
        lines = [l.strip() for l in (text or "").splitlines()]
        items = [l if l.startswith("- ") else "- " + l.lstrip("-•* ").strip() for l in lines if re.match(r"^[-•*]\s*\S", l)]
        items = [re.sub(r"https?://\S+", "", l).rstrip() for l in items]
        items = [l if len(l) <= MAX_LINE else l[:MAX_LINE].rsplit(" ", 1)[0].rstrip(",;:") + "…" for l in items]
        return "\n".join(items[:15])

    async def _generate(self) -> TaskResult:
        """Run the configured source. Returns a TaskResult whose cost includes the X pulls."""
        digest, x_cost = "", 0.0
        if self.x is not None and self.source in ("x", "both"):
            pull = await self.x.pull()
            digest, x_cost = pull.digest, pull.cost_usd
            if not digest:
                if self.source == "x" or self.runner is None:
                    return TaskResult(False, reason=(pull.errors[0] if pull.errors else "x_no_posts"), cost_usd=x_cost)
                log_event("brief.x_empty", errors=len(pull.errors))  # both: fall back to web only
        hours = str(self.x.lookback_hours) if self.x is not None else "48"
        if self.source == "x":
            system = X_WRITER_SYSTEM.replace("{topics}", self.topics).replace("{hours}", hours)
            user = digest + "\n\nNow write the brief, 10 to 15 lines in the exact form above."
            text = await self.writer.reply(system, [{"role": "user", "content": user}])
            return TaskResult(bool(text), text=text or "", cost_usd=x_cost, reason="" if text else "empty")
        request = BRIEF_PROMPT.replace("{topics}", self.topics)
        if digest:
            request += "\n\n" + X_GROUNDING.replace("{hours}", hours).replace("{digest}", digest)
        res = await self.runner.run(request, "the group", [])
        res.cost_usd = (res.cost_usd or 0.0) + x_cost
        return res

    async def refresh(self) -> bool:
        if self._lock.locked():
            return False
        async with self._lock:
            started = time.monotonic()
            try:
                res = await self._generate()
            except Exception as exc:  # noqa: BLE001
                self.last_error = exc_name(exc)
                log_event("brief.failed", error=self.last_error, source=self.source)
                return False
            self.last_cost_usd = res.cost_usd
            items = self.clean(res.text) if res.ok else ""
            if not items or items.count("\n") < 4:
                self.last_error = res.reason or "too_short"
                log_event("brief.failed", reason=self.last_error[:40], source=self.source, cost_cents=round(res.cost_usd * 100))
                return False
            stamp = datetime.fromtimestamp(self._clock()).strftime("%a %b %-d")
            body = f"## What people are talking about (updated {stamp})"
            for line in items.splitlines():  # cut between lines, never mid-line
                if len(body) + 1 + len(line) > self.max_chars:
                    break
                body += "\n" + line
            self.save(body)
            self.last_error = None
            log_event("brief.updated", chars=len(body), source=self.source, cost_cents=round(res.cost_usd * 100),
                      duration_ms=int((time.monotonic() - started) * 1000))
            return True

    def status(self) -> dict:
        return {
            "source": self.source,
            "writer": getattr(self.writer, "name", None) if self.source == "x" else ("task agent" if self.runner else None),
            "last_cost_usd": round(self.last_cost_usd, 4),
            "x": self.x.status() if self.x is not None else None,
        }

    async def aclose(self) -> None:
        for obj in (self.x, self.writer):
            close = getattr(obj, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass

    # -- schedule --
    def next_run(self, now: datetime) -> datetime:
        try:
            hh, mm = (int(x) for x in self.at.split(":", 1))
        except ValueError:
            hh, mm = 8, 0
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return t if t > now else t + timedelta(days=1)

    async def run_forever(self, startup_delay: float = 60.0) -> None:
        age = self.age_hours()
        if age is None or age > 20:
            await asyncio.sleep(startup_delay)  # let the server settle first
            await self.refresh()
        while True:
            now = datetime.fromtimestamp(self._clock())
            await asyncio.sleep(max(1.0, (self.next_run(now) - now).total_seconds()))
            await self.refresh()
