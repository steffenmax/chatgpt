"""Daily culture brief: what people are joking about online, for the bot's prompt.

Once a day (BRIEF_TIME, local) the task agent searches the web, with the same fenced
toolset as /task and a generic prompt (no chat content is sent), and writes 10-15
one-line references. The brief is stored as a 0600 markdown file the owner can edit
in the dashboard, and is added to the bot's system prompt while it's fresh.
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

STALE_AFTER_HOURS = 72
MAX_LINE = 180
DEFAULT_TOPICS = (
    "New York sports (Jets, Giants, Knicks, Nets, Yankees, Mets, Rangers, Islanders), "
    "NFL Twitter and NBA Twitter, and general internet culture and memes"
)

BRIEF_PROMPT = (
    "You compile a daily culture brief for a group of friends' group-chat buddy, so he's up on what "
    "people are talking and joking about. Focus: {topics}.\n"
    "Search for what's trending and being joked about in the last two or three days: what's blowing "
    "up on X/Twitter, TikTok, Reddit and sports media, viral moments, memes and catchphrases, "
    "sports drama, trades, meltdowns and hot takes. Prefer specific names, phrases and moments over "
    "generic topics. Skip deaths, disasters and violent crime.\n"
    "Final answer: ONLY the brief, 10 to 15 lines, each in the form\n"
    "- <the reference, phrase or moment>: <one short line on what it is and why people are joking "
    "about it>\nEach line under 160 characters. No intro, no outro, no links, no headings, nothing "
    "about yourself."
)


class CultureBrief:
    def __init__(self, path: str, runner, topics: str = "", at: str = "08:00", max_chars: int = 2000,
                 clock=time.time):
        self.path = Path(path).expanduser()
        self.runner = runner
        self.topics = topics.strip() or DEFAULT_TOPICS
        self.at = at
        self.max_chars = max_chars
        self._clock = clock
        self._lock = asyncio.Lock()
        self.last_error: str | None = None

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

    async def refresh(self) -> bool:
        if self._lock.locked():
            return False
        async with self._lock:
            started = time.monotonic()
            try:
                res = await self.runner.run(BRIEF_PROMPT.replace("{topics}", self.topics), "the group", [])
            except Exception as exc:  # noqa: BLE001
                self.last_error = exc_name(exc)
                log_event("brief.failed", error=self.last_error)
                return False
            items = self.clean(res.text) if res.ok else ""
            if not items or items.count("\n") < 4:
                self.last_error = res.reason or "too_short"
                log_event("brief.failed", reason=self.last_error[:40], cost_cents=round(res.cost_usd * 100))
                return False
            stamp = datetime.fromtimestamp(self._clock()).strftime("%a %b %-d")
            body = f"## What people are talking about (updated {stamp})"
            for line in items.splitlines():  # cut between lines, never mid-line
                if len(body) + 1 + len(line) > self.max_chars:
                    break
                body += "\n" + line
            self.save(body)
            self.last_error = None
            log_event("brief.updated", chars=len(body), cost_cents=round(res.cost_usd * 100),
                      duration_ms=int((time.monotonic() - started) * 1000))
            return True

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
