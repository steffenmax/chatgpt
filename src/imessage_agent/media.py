"""GIFs and funny tweets the bot can drop into the chat.

  * GIFs: GIPHY search (needs a free GIPHY_API_KEY), rated GIF_RATING (default pg-13).
    The GIF is downloaded (size-capped) and sent as an attachment.
  * Tweets: one X recent search (the same pay-per-use API as the brief) for what the chat
    is talking about. The most-liked post that X doesn't flag as possibly sensitive is sent
    as a link. The search words come from the chat, so they go to X; each search is billed
    per post returned, and TWEETS_PER_HOUR caps how often it can happen.
Only counts and costs are logged, never the search words or post text.
"""

from __future__ import annotations

import random
import re
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .logsafe import exc_name, log_event
from .xsearch import API_URL, POST_READ_USD, full_query, rank

GIPHY_SEARCH = "https://api.giphy.com/v1/gifs/search"
MAX_GIF_BYTES = 8_000_000


def clean_query(q: str, limit: int = 80) -> str:
    """Search words only: no URLs, @handles, operators or quotes."""
    q = re.sub(r"https?://\S+|@\w+|\b\w+:\S+", " ", q or "")
    q = re.sub(r"[^\w\s'&-]", " ", q)
    return " ".join(q.split())[:limit].strip()


@dataclass
class Gif:
    path: Path
    query: str
    title: str = ""


class GifSearch:
    def __init__(self, api_key: str, rating: str = "pg-13", timeout: float = 15.0, client: httpx.AsyncClient | None = None):
        self.api_key = api_key
        self.rating = rating
        self._client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def find(self, query: str) -> Gif | None:
        """Search, pick one of the top results, download it. None if nothing usable."""
        q = clean_query(query, 50)
        if not q:
            return None
        r = await self._client.get(GIPHY_SEARCH, params={"api_key": self.api_key, "q": q, "limit": 8,
                                                          "rating": self.rating, "lang": "en"})
        r.raise_for_status()
        results = [g for g in (r.json().get("data") or []) if isinstance(g, dict)]
        top = results[:4]
        random.shuffle(top)  # some variety among the best matches
        for g in top + results[4:]:
            imgs = g.get("images") or {}
            for key in ("downsized", "fixed_height", "original"):
                url = (imgs.get(key) or {}).get("url")
                size = int((imgs.get(key) or {}).get("size") or 0)
                if url and url.startswith("https://") and (not size or size <= MAX_GIF_BYTES):
                    data = await self._download(url)
                    if data:
                        f = tempfile.NamedTemporaryFile(prefix="gif-", suffix=".gif", delete=False)
                        f.write(data)
                        f.close()
                        return Gif(Path(f.name), q, str(g.get("title") or "")[:120])
        return None

    async def _download(self, url: str) -> bytes | None:
        async with self._client.stream("GET", url) as r:
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                return None
            body = b""
            async for chunk in r.aiter_bytes():
                body += chunk
                if len(body) > MAX_GIF_BYTES:
                    return None
            return body


@dataclass
class Tweet:
    id: str
    text: str
    likes: int

    @property
    def url(self) -> str:
        return f"https://x.com/i/status/{self.id}"


class TweetFinder:
    def __init__(self, bearer_token: str, min_likes: int = 200, posts_per_search: int = 10, per_hour: int = 6,
                 filters: str = "lang:en -is:retweet -is:reply", timeout: float = 20.0,
                 transport: httpx.AsyncBaseTransport | None = None, clock=time.monotonic):
        self.min_likes = min_likes
        self.posts_per_search = max(10, min(100, posts_per_search))
        self.per_hour = per_hour
        self.filters = filters
        self._clock = clock
        self._recent: deque[float] = deque()
        self._sent: deque[str] = deque(maxlen=200)  # don't send the same post twice
        self.spent_usd = 0.0
        self._client = httpx.AsyncClient(headers={"Authorization": f"Bearer {bearer_token}", "User-Agent": "imessage-agent"},
                                         timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    def allowed(self) -> bool:
        now = self._clock()
        while self._recent and now - self._recent[0] > 3600:
            self._recent.popleft()
        return len(self._recent) < self.per_hour

    async def find(self, query: str) -> Tweet | None:
        q = clean_query(query)
        if not q or not self.allowed():
            return None
        self._recent.append(self._clock())
        start = (datetime.now(timezone.utc) - timedelta(days=6, hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = await self._client.get(API_URL, params={
            "query": full_query(q, self.filters), "max_results": self.posts_per_search, "sort_order": "relevancy",
            "start_time": start, "tweet.fields": "public_metrics,possibly_sensitive,created_at",
        })
        if r.status_code != 200:
            log_event("tweet.search_failed", status=r.status_code)
            return None
        posts = [p for p in (r.json().get("data") or []) if isinstance(p, dict)]
        cost = round(len(posts) * POST_READ_USD, 4)
        self.spent_usd = round(self.spent_usd + cost, 4)
        ok = [p for p in posts if not p.get("possibly_sensitive") and str(p.get("id")) not in self._sent
              and int((p.get("public_metrics") or {}).get("like_count") or 0) >= self.min_likes]
        ranked = rank(ok)
        log_event("tweet.searched", posts=len(posts), count=len(ranked), cost_cents=round(cost * 100, 1))
        if not ranked:
            return None
        best = ranked[0]
        self._sent.append(str(best["id"]))
        return Tweet(str(best["id"]), best["text"], int((best.get("public_metrics") or {}).get("like_count") or 0))


def gif_from_reaction(text: str) -> str | None:
    """A chime-in written as 'GIF: <search>' means send a GIF instead of text."""
    m = re.match(r"^\s*gif\s*:\s*(.+)$", text or "", re.IGNORECASE)
    if not m:
        return None
    return clean_query(m.group(1), 50) or None


def describe_tweet(t: Tweet) -> str:
    """What goes into the bot's context after it sends a tweet."""
    text = " ".join(t.text.replace("[", "(").replace("]", ")").split())[:240]
    return f'[sent a tweet: "{text}" ({t.likes} likes) {t.url}]'

