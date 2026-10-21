"""Read-only Kalshi prediction-market odds (public market data API, no key, GET only).

There's no account and no API key, so nothing here can trade. A lightweight index of open
events (title, subtitle, outcome labels; no prices) is rebuilt every 30 minutes; a search
picks the best event and its prices are fetched fresh. Search words go to nobody: matching
happens on this Mac, and Kalshi only sees "list events" and "get event X".
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .logsafe import log_event

API = "https://api.elections.kalshi.com/trade-api/v2"
INDEX_TTL = 1800.0

# How Kalshi labels teams people say by nickname, plus the sport (so "knicks" finds basketball).
TEAMS = {
    "knicks": ("new york", "basketball"), "nets": ("brooklyn", "basketball"), "liberty": ("new york", "women's basketball"),
    "giants": ("new york g", "football"), "jets": ("new york j", "football"), "yankees": ("new york y", "baseball"),
    "yanks": ("new york y", "baseball"), "mets": ("new york m", "baseball"), "rangers": ("new york r", "hockey"),
    "islanders": ("new york i", "hockey"), "isles": ("new york i", "hockey"), "patriots": ("new england", "football"),
    "pats": ("new england", "football"), "eagles": ("philadelphia", "football"), "birds": ("philadelphia", "football"),
    "sixers": ("philadelphia", "basketball"), "76ers": ("philadelphia", "basketball"), "flyers": ("philadelphia", "hockey"),
    "phillies": ("philadelphia", "baseball"), "celtics": ("boston", "basketball"), "red sox": ("boston", "baseball"),
    "cowboys": ("dallas", "football"), "lakers": ("los angeles l", "basketball"), "devils": ("new jersey", "hockey"),
}
TEAM_LABELS = {nick: label for nick, (label, _) in TEAMS.items()}

# Everyday words -> Kalshi's wording.
SYNONYMS = {
    "title": "champion", "championship": "champion", "ring": "champion", "chip": "champion", "win it all": "champion",
    "super bowl": "pro football champion", "nba finals": "pro basketball champion", "world series": "baseball champion",
    "nba": "pro basketball", "nfl": "pro football", "nhl": "hockey", "mlb": "baseball",
    "playoffs": "playoff", "make the playoffs": "playoff qualifiers", "rate cut": "cut", "rates": "rate",
}
_ESPORTS = re.compile(r"\b(ebasketball|esoccer|e-?football|efootball|esports?|2k\d*|fifa|madden|nba 2k)\b", re.IGNORECASE)
_STOP = {"ricky", "kalshi", "odds", "on", "of", "the", "a", "an", "what", "whats", "what's", "are", "is", "will", "to",
         "for", "in", "at", "and", "chance", "chances", "prediction", "market", "markets", "say", "says", "think",
         "does", "do", "win", "wins", "price", "prices", "per", "according", "pull", "up", "look", "me", "us"}

KALSHI_ASK = re.compile(r"\bkalshi\b|\bprediction markets?\b", re.IGNORECASE)


def wants_kalshi(text: str) -> bool:
    return bool(KALSHI_ASK.search(text or ""))


def _words(text: str) -> list[str]:
    t = (text or "").lower()
    for phrase, kalshi in SYNONYMS.items():
        t = re.sub(rf"\b{re.escape(phrase)}\b", f"{phrase} {kalshi}", t)
    for nick, (label, sport) in TEAMS.items():
        t = re.sub(rf"\b{re.escape(nick)}\b", f"{nick} {label} {sport}", t)
    return [w for w in re.findall(r"[a-z0-9$%.]+", t) if w not in _STOP and len(w) > 1 or w.isdigit()]


@dataclass
class EventDoc:
    ticker: str
    title: str
    sub_title: str
    category: str
    outcomes: str  # every market's outcome label, lowercased, " | "-joined
    volume: float = 0.0

    def score(self, words: list[str], phrase: str, esports_ok: bool = False) -> float:
        title = f"{self.title} {self.sub_title}".lower()
        hits = 0.0
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", title):
                hits += 2.0
            elif re.search(rf"\b{re.escape(w)}\b", self.outcomes):
                hits += 1.0
        if phrase and (phrase in self.outcomes or phrase in title):
            hits += 3.0
        if hits < 2.0:
            return hits
        # focused titles beat ones full of things nobody asked about; busy markets beat obscure ones
        extra = len([t for t in re.findall(r"[a-z0-9]+", title) if t not in words and t not in _STOP and len(t) > 2])
        s = hits / (1 + 0.08 * extra) + 0.35 * math.log10(1 + self.volume)
        if _ESPORTS.search(title) and not esports_ok:
            s *= 0.2
        return s


@dataclass
class Outcome:
    label: str
    prob: float | None  # 0..1
    volume: float


def _price(m: dict[str, Any]) -> float | None:
    def f(k):
        try:
            v = m.get(k)
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None
    last, bid, ask = f("last_price_dollars"), f("yes_bid_dollars"), f("yes_ask_dollars")
    if bid is not None and ask is not None and ask > 0 and ask - bid <= 0.15:
        return (bid + ask) / 2
    return last if last else (bid or ask)


class Kalshi:
    def __init__(self, timeout: float = 30.0, client: httpx.AsyncClient | None = None, clock=time.monotonic):
        self._client = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "imessage-agent"})
        self._clock = clock
        self.index: list[EventDoc] = []
        self._built = -1e9
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with backoff on 429 (Kalshi rate-limits the public API)."""
        for attempt in range(4):
            r = await self._client.get(url, params=params)
            if r.status_code != 429:
                r.raise_for_status()
                return r.json()
            try:
                wait = float(r.headers.get("retry-after") or 0)
            except ValueError:
                wait = 0.0
            await asyncio.sleep(max(wait, 2.0 * (attempt + 1)))
        r.raise_for_status()
        return {}

    async def _ensure_index(self) -> None:
        async with self._lock:
            if self.index and self._clock() - self._built < INDEX_TTL:
                return
            try:
                await self._build()
            except httpx.HTTPError as exc:
                if not self.index:
                    raise
                log_event("kalshi.index_failed", status=getattr(getattr(exc, "response", None), "status_code", None))

    async def _build(self) -> None:
        if True:
            docs, cursor = [], None
            for _ in range(200):
                params: dict[str, Any] = {"status": "open", "limit": 200, "with_nested_markets": "true"}
                if cursor:
                    params["cursor"] = cursor
                    await asyncio.sleep(0.2)  # pace the pages
                d = await self._get(f"{API}/events", params)
                for e in d.get("events") or []:
                    mk = e.get("markets") or []
                    outs = " | ".join(str(m.get("yes_sub_title") or "") for m in mk)
                    vol = 0.0
                    for m in mk:
                        try:
                            vol += float(m.get("volume_fp") or 0)
                        except (TypeError, ValueError):
                            pass
                    docs.append(EventDoc(e["event_ticker"], str(e.get("title") or ""), str(e.get("sub_title") or ""),
                                         str(e.get("category") or ""), outs.lower(), vol))
                cursor = d.get("cursor")
                if not cursor:
                    break
            self.index, self._built = docs, self._clock()
            log_event("kalshi.indexed", count=len(docs))

    async def search(self, query: str) -> tuple[EventDoc, list[Outcome], list[str]] | None:
        """(best event, its outcomes most likely first, outcome labels the query named), or None."""
        await self._ensure_index()
        words = _words(query)
        if not words:
            return None
        phrase = next((label for nick, label in TEAM_LABELS.items() if re.search(rf"\b{nick}\b", query, re.I)), "")
        esports_ok = bool(_ESPORTS.search(query))
        best = max(self.index, key=lambda d: d.score(words, phrase, esports_ok), default=None)
        if best is None or best.score(words, phrase, esports_ok) < 2.0:
            return None
        d = await self._get(f"{API}/events/{best.ticker}", {"with_nested_markets": "true"})
        markets = d.get("markets") or (d.get("event") or {}).get("markets") or []
        outs = []
        for m in markets:
            try:
                vol = float(m.get("volume_fp") or 0)
            except (TypeError, ValueError):
                vol = 0.0
            outs.append(Outcome(str(m.get("yes_sub_title") or m.get("title") or "?"), _price(m), vol))
        outs.sort(key=lambda o: (o.prob is None, -(o.prob or 0)))
        if phrase:  # a team: highlight exactly that one ("new york j", not every New York)
            named = [o.label for o in outs if o.label.lower() == phrase or o.label.lower().startswith(phrase + " ")
                     and not re.match(rf"{re.escape(phrase)} [a-z]\b", o.label.lower())]
        else:
            named = [o.label for o in outs if any(re.search(rf"\b{re.escape(w)}\b", o.label.lower()) for w in words if len(w) > 3)]
        return best, outs, named


def format_odds(event: EventDoc, outcomes: list[Outcome], named: list[str], limit: int = 6) -> str:
    def pct(p: float | None) -> str:
        if p is None:
            return "no trades"
        return "<1%" if 0 < p < 0.005 else ">99%" if 0.995 < p < 1 else f"{round(p * 100)}%"

    def row(o: Outcome) -> str:
        return f"- {o.label}: {pct(o.prob)}"
    title = event.title + (f" ({event.sub_title})" if event.sub_title and event.sub_title not in event.title else "")
    first = [o for o in outcomes if o.label in named][:3]
    rest = [o for o in outcomes if o not in first][: max(0, limit - len(first))]
    if len(outcomes) == 1:
        return f"Kalshi: {title}: {pct(outcomes[0].prob)} yes"
    return f"Kalshi: {title}\n" + "\n".join(row(o) for o in first + rest)


