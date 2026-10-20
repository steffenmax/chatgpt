"""X (Twitter) API pulls for the daily culture brief.

Each brief topic becomes one recent-search query (GET /2/tweets/search/recent,
sorted by relevancy, last X_LOOKBACK_HOURS). The most-engaged posts per topic are
ranked, trimmed and joined into a plain-text digest the brief writer works from.

Money: X bills pay-per-use, POST_READ_USD per post returned, so every query asks for
exactly the number of posts it will use and the whole refresh is capped at
X_MAX_USD_PER_REFRESH (queries are thinned, then dropped, to fit). No author
expansions are requested (user reads cost extra). A small ledger next to the brief
file keeps a running estimate of what has been spent; the Developer Console is the
truth. Nothing from the group chat is ever sent to X: queries come from config only.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from .logsafe import exc_name, log_event

API_URL = "https://api.x.com/2/tweets/search/recent"
POST_READ_USD = 0.005  # pay-per-use "Posts: Read", per post returned
MIN_RESULTS, MAX_RESULTS = 10, 100  # endpoint limits for max_results
MAX_QUERY_CHARS = 512  # self-serve limit for recent search
MAX_QUERIES = 12
POST_MAX_CHARS = 240
FILTERS = "lang:en -is:retweet -is:reply"

# Hand-tuned queries for brief.DEFAULT_TOPICS. Each is one search; the filters above are appended.
DEFAULT_QUERIES = [
    "(Jets OR Giants) NFL",
    "(Knicks OR Nets) NBA",
    "(Yankees OR Mets) MLB",
    "(Rangers OR Islanders) NHL",
    "NFL",
    "NBA",
    "(meme OR viral OR lmaooo)",
]

_FILLER = re.compile(r"^(general|the|and|also|plus)\s+", re.IGNORECASE)
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def topics_to_queries(topics: str) -> list[str]:
    """Turn BRIEF_TOPICS into search queries.

    Items are separated by ';' (or ',' and ' and ' when there is no ';'). A parenthesised
    list becomes an OR group: "NY sports (Jets, Giants)" -> "(Jets OR Giants)". A plain item
    is searched as its words. Empty input -> DEFAULT_QUERIES.
    """
    text = (topics or "").strip()
    if not text:
        return list(DEFAULT_QUERIES)
    out: list[str] = []

    def add(q: str) -> None:
        q = _WS.sub(" ", q).strip(" ,;.")
        q = _FILLER.sub("", q).strip()
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q[:MAX_QUERY_CHARS - len(FILTERS) - 1])

    # OR groups first, so their inner commas don't split the list
    for m in _PAREN.finditer(text):
        alts = [a.strip() for a in re.split(r",|\bor\b", m.group(1)) if a.strip()]
        if alts:
            add("(" + " OR ".join(_quote(a) for a in alts) + ")")
    rest = _PAREN.sub("", text)
    parts = rest.split(";") if ";" in rest else re.split(r",|\band\b", rest)
    for p in parts:
        add(p)
    return out[:MAX_QUERIES] or list(DEFAULT_QUERIES)


def _quote(term: str) -> str:
    term = term.strip().strip('"')
    return f'"{term}"' if " " in term else term


def full_query(q: str, filters: str = FILTERS) -> str:
    """Append the filters (X_QUERY_FILTERS) unless the query already sets them."""
    q = q.strip()
    extra = []
    for f in filters.split():
        op = f.lstrip("-")
        marker = "lang:" if op.startswith("lang:") else op  # any lang: counts; is:x only matches itself
        if marker not in q:
            extra.append(f)
    return (q + " " + " ".join(extra)).strip()[:MAX_QUERY_CHARS]


def parse_queries(raw: str) -> list[str]:
    """X_QUERIES: one search per ';'-separated item."""
    return [q.strip() for q in (raw or "").split(";") if q.strip()][:MAX_QUERIES]


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k".replace(".0k", "k") if n >= 1000 else str(n)


def _clean(text: str) -> str:
    text = _URL.sub("", text or "")
    text = _WS.sub(" ", text).strip()
    return text if len(text) <= POST_MAX_CHARS else text[:POST_MAX_CHARS].rsplit(" ", 1)[0] + "…"


def rank(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Most engaged first; drop near-duplicates (copy-pasta, spam)."""
    def score(p: dict[str, Any]) -> int:
        m = p.get("public_metrics") or {}
        return int(m.get("like_count") or 0) + 2 * (int(m.get("retweet_count") or 0) + int(m.get("quote_count") or 0)) + int(m.get("reply_count") or 0)

    seen: set[str] = set()
    out = []
    for p in sorted(posts, key=score, reverse=True):
        text = _clean(str(p.get("text") or ""))
        key = re.sub(r"\W+", "", text.lower())[:60]
        if len(key) < 4 or key in seen:  # empty, emoji-only, or a copy of one we kept
            continue
        seen.add(key)
        out.append({**p, "text": text, "_score": score(p)})
    return out


@dataclass
class PullResult:
    digest: str = ""
    posts: int = 0
    queries_ok: int = 0
    queries_failed: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0


class XSearch:
    def __init__(
        self,
        bearer_token: str,
        queries: list[str],
        posts_per_query: int = 15,
        lookback_hours: int = 48,
        max_usd_per_refresh: float = 0.75,
        ledger_path: str | None = None,
        digest_max_chars: int = 30000,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock=time.time,
        filters: str = FILTERS,
    ):
        self.queries = [q for q in queries if q.strip()][:MAX_QUERIES]
        self.filters = filters.strip() or FILTERS
        self.posts_per_query = max(MIN_RESULTS, min(MAX_RESULTS, posts_per_query))
        self.lookback_hours = lookback_hours
        self.max_usd = max_usd_per_refresh
        self.ledger_path = Path(ledger_path).expanduser() if ledger_path else None
        self.digest_max_chars = digest_max_chars
        self._clock = clock
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {bearer_token}", "User-Agent": "imessage-agent-brief"},
            timeout=timeout,
            transport=transport,
        )
        self.last: PullResult | None = None

    # -- budget --
    def plan(self) -> list[tuple[str, int]]:
        """(query, max_results) pairs that fit the per-refresh cap: thin first, then drop."""
        if not self.queries:
            return []
        allowed = int(self.max_usd / POST_READ_USD + 1e-9)
        n = self.posts_per_query
        while n * len(self.queries) > allowed and n > MIN_RESULTS:
            n -= 1
        keep = max(0, min(len(self.queries), allowed // n)) if n else 0
        return [(q, n) for q in self.queries[:keep]]

    def estimate_usd(self) -> float:
        return round(sum(n for _, n in self.plan()) * POST_READ_USD, 4)

    # -- ledger (counts and dollars only, never post text) --
    def ledger(self) -> dict[str, Any]:
        if self.ledger_path is None:
            return {}
        try:
            d = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _record(self, res: PullResult) -> None:
        if self.ledger_path is None:
            return
        d = self.ledger()
        now = datetime.fromtimestamp(self._clock()).isoformat(timespec="seconds")
        d.setdefault("since", now)
        d["refreshes"] = int(d.get("refreshes") or 0) + 1
        d["posts"] = int(d.get("posts") or 0) + res.posts
        d["usd"] = round(float(d.get("usd") or 0) + res.cost_usd, 4)
        d["last"] = {"at": now, "posts": res.posts, "usd": round(res.cost_usd, 4),
                     "queries_ok": res.queries_ok, "queries_failed": res.queries_failed}
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".x-spend-", suffix=".json", dir=self.ledger_path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(d, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.ledger_path)
        except OSError as exc:
            log_event("x.ledger_failed", error=exc_name(exc))

    def status(self) -> dict[str, Any]:
        led = self.ledger()
        return {
            "queries": len(self.plan()),
            "queries_configured": len(self.queries),
            "posts_per_query": self.plan()[0][1] if self.plan() else 0,
            "est_usd_per_refresh": self.estimate_usd(),
            "spent_usd": led.get("usd", 0.0),
            "spent_since": led.get("since"),
            "refreshes": led.get("refreshes", 0),
            "last": led.get("last"),
        }

    # -- pulling --
    async def pull(self) -> PullResult:
        started = time.monotonic()
        res = PullResult()
        start_time = (datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sections: list[str] = []
        for query, n in self.plan():
            params = {
                "query": full_query(query, self.filters),
                "max_results": n,
                "sort_order": "relevancy",
                "start_time": start_time,
                "tweet.fields": "created_at,public_metrics,lang",
            }
            try:
                r = await self._client.get(API_URL, params=params)
            except httpx.HTTPError as exc:
                res.queries_failed += 1
                res.errors.append(exc_name(exc))
                log_event("x.query_failed", error=exc_name(exc))
                continue
            if r.status_code != 200:
                res.queries_failed += 1
                reason = f"http_{r.status_code}"
                res.errors.append(reason)
                log_event("x.query_failed", status=r.status_code)
                if r.status_code in (401, 402, 403, 429):
                    break  # bad token, no credits, no access, rate limited: don't keep trying
                continue
            try:
                posts = [p for p in (r.json().get("data") or []) if isinstance(p, dict)]
            except ValueError:
                res.queries_failed += 1
                res.errors.append("bad_json")
                continue
            res.queries_ok += 1
            res.posts += len(posts)
            res.cost_usd += len(posts) * POST_READ_USD
            ranked = rank(posts)
            if ranked:
                lines = [f"## {query}"]
                for p in ranked:
                    m = p.get("public_metrics") or {}
                    lines.append(f"- [{_k(int(m.get('like_count') or 0))} likes, {_k(int(m.get('retweet_count') or 0))} reposts] {p['text']}")
                sections.append("\n".join(lines))
        digest = "\n\n".join(sections)
        if len(digest) > self.digest_max_chars:  # cut between lines, never mid-line
            digest = digest[: self.digest_max_chars].rsplit("\n", 1)[0]
        res.digest = digest
        res.cost_usd = round(res.cost_usd, 4)
        res.duration_ms = int((time.monotonic() - started) * 1000)
        self.last = res
        self._record(res)
        log_event("x.pulled", posts=res.posts, queries_ok=res.queries_ok, queries_failed=res.queries_failed,
                  cost_cents=round(res.cost_usd * 100, 1), duration_ms=res.duration_ms)
        return res

    async def aclose(self) -> None:
        await self._client.aclose()
