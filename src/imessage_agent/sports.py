"""NFL scores, scoring plays and odds from ESPN's public site API (no key, no account).

Used by the sports alerts (alerts.py) and the bookie (bookie.py). Read-only: nothing
from the chat is ever sent to ESPN; requests are just "this week's scoreboard" and
"scoring plays for game N".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

# Nicknames people actually type, on top of ESPN's names (location, nickname, abbreviation).
EXTRA_ALIASES = {
    "NYG": ["big blue", "g men", "gmen", "g-men", "jints"],
    "NYJ": ["gang green", "jest", "j-e-t-s"],
    "NE": ["pats", "new england", "patsies"],
    "PHI": ["birds", "iggles", "philly"],
    "DAL": ["boys", "cowgirls"],
    "KC": ["chiefs kingdom"],
    "SF": ["niners", "9ers", "49ers", "sf"],
    "TB": ["bucs", "tampa"],
    "GB": ["pack", "green bay", "cheeseheads"],
    "LV": ["raiders", "vegas"],
    "LAR": ["rams"],
    "LAC": ["bolts", "chargers"],
    "JAX": ["jags"],
    "WSH": ["commies", "commanders", "washington"],
    "ARI": ["cards", "cardinals"],
    "IND": ["colts", "indy"],
    "NO": ["saints", "new orleans", "nola"],
    "MIN": ["vikes", "vikings"],
    "BAL": ["ravens", "bmore"],
    "CIN": ["bengals", "cincy"],
    "PIT": ["steelers", "pitt"],
    "HOU": ["texans", "houston"],
}


def american_profit(stake: int, odds: int | None) -> int:
    """Winnings (not counting the stake) on American odds; even money if unknown."""
    if not odds:
        return stake
    return int(stake * odds / 100) if odds > 0 else int(stake * 100 / abs(odds))


def fmt_odds(odds: int | None) -> str:
    if not odds:
        return "even"
    return f"+{odds}" if odds > 0 else str(odds)


def _int(v: Any) -> int | None:
    try:
        return int(str(v).replace("+", "").strip())
    except (TypeError, ValueError):
        return None


@dataclass
class Team:
    abbr: str
    name: str  # "Giants"
    full: str  # "New York Giants"
    score: int = 0
    moneyline: int | None = None


@dataclass
class Game:
    id: str
    start: datetime
    state: str  # pre | in | post
    detail: str  # "Final", "Halftime", "2nd 4:12"
    period: int
    clock: str
    home: Team
    away: Team
    week: int = 0
    season_type: int = 2
    spread: str = ""  # "HOU -10"
    total: float | None = None
    completed: bool = False

    def team(self, abbr: str) -> Team | None:
        return self.home if self.home.abbr == abbr else self.away if self.away.abbr == abbr else None

    def other(self, abbr: str) -> Team:
        return self.away if self.home.abbr == abbr else self.home

    @property
    def matchup(self) -> str:
        return f"{self.away.name} @ {self.home.name}"

    def score_line(self) -> str:
        lead, trail = (self.home, self.away) if self.home.score >= self.away.score else (self.away, self.home)
        return f"{lead.name} {lead.score}, {trail.name} {trail.score}"

    def winner(self) -> str | None:
        """Abbreviation of the winner, or None for a tie / unfinished game."""
        if not self.completed or self.home.score == self.away.score:
            return None
        return self.home.abbr if self.home.score > self.away.score else self.away.abbr


def parse_scoreboard(data: dict[str, Any]) -> list[Game]:
    week = int(((data.get("week") or {}).get("number")) or 0)
    season_type = int(((data.get("season") or {}).get("type")) or 2)
    games = []
    for ev in data.get("events") or []:
        try:
            c = ev["competitions"][0]
            st = ev.get("status") or c.get("status") or {}
            typ = st.get("type") or {}
            teams = {}
            for comp in c["competitors"]:
                t = comp["team"]
                teams[comp["homeAway"]] = Team(t["abbreviation"], t.get("name") or t.get("shortDisplayName") or t["abbreviation"],
                                               t.get("displayName") or t["abbreviation"], _int(comp.get("score")) or 0)
            odds = (c.get("odds") or [None])[0] or {}
            ml = odds.get("moneyline") if isinstance(odds.get("moneyline"), dict) else {}
            for side in ("home", "away"):
                line = (ml.get(side) or {})
                teams[side].moneyline = _int((line.get("close") or {}).get("odds")) or _int((line.get("open") or {}).get("odds"))
            games.append(Game(
                id=str(ev["id"]),
                start=datetime.fromisoformat(str(ev["date"]).replace("Z", "+00:00")),
                state=str(typ.get("state") or "pre"),
                detail=str(typ.get("shortDetail") or typ.get("detail") or ""),
                period=int(st.get("period") or 0),
                clock=str(st.get("displayClock") or ""),
                home=teams["home"], away=teams["away"], week=week, season_type=season_type,
                spread=str(odds.get("details") or ""), total=odds.get("overUnder"),
                completed=bool(typ.get("completed")),
            ))
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    return games


@dataclass
class ScoringPlay:
    id: str
    team: str  # abbreviation
    kind: str  # TD, FG, SF, ...
    text: str
    period: int
    clock: str
    home_score: int
    away_score: int


def parse_scoring_plays(data: dict[str, Any]) -> list[ScoringPlay]:
    out = []
    for p in data.get("scoringPlays") or []:
        try:
            kind = ((p.get("scoringType") or {}).get("abbreviation") or (p.get("type") or {}).get("abbreviation") or "").upper()
            out.append(ScoringPlay(str(p["id"]), str((p.get("team") or {}).get("abbreviation") or ""), kind,
                                   " ".join(str(p.get("text") or "").split())[:200], int((p.get("period") or {}).get("number") or 0),
                                   str((p.get("clock") or {}).get("displayValue") or ""), int(p.get("homeScore") or 0),
                                   int(p.get("awayScore") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def quarter(period: int) -> str:
    return {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}.get(period, "OT")


class ESPN:
    def __init__(self, timeout: float = 15.0, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "imessage-agent"})

    async def aclose(self) -> None:
        await self._client.aclose()

    async def scoreboard(self, week: int | None = None, season_type: int | None = None) -> list[Game]:
        params = {}
        if week:
            params = {"week": week, "seasontype": season_type or 2}
        r = await self._client.get(SCOREBOARD, params=params)
        r.raise_for_status()
        return parse_scoreboard(r.json())

    async def upcoming(self) -> list[Game]:
        """This week's games, plus next week's once this week is all finished (for betting)."""
        games = await self.scoreboard()
        if games and all(g.state == "post" for g in games):
            nxt = await self.scoreboard(games[0].week + 1, games[0].season_type)
            if nxt:
                return nxt
        return games

    async def scoring_plays(self, game_id: str) -> list[ScoringPlay]:
        r = await self._client.get(SUMMARY, params={"event": game_id})
        r.raise_for_status()
        return parse_scoring_plays(r.json())


# Abbreviations that are everyday words ("no", "ten", "min") don't count as team names.
_COMMON_WORDS = {"no", "car", "den", "min", "sea", "ten", "was", "ind", "la", "ny"}


@dataclass
class TeamIndex:
    """Find a team from what people type ('the jets', 'big blue', 'pats')."""

    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_games(cls, games: list[Game]) -> "TeamIndex":
        seen: dict[str, set[str]] = {}
        for g in games:
            for t in (g.home, g.away):
                for a in (t.abbr, t.name, t.full, t.full.rsplit(" ", 1)[0], *EXTRA_ALIASES.get(t.abbr, [])):
                    a = a.lower().strip()
                    if a and a not in _COMMON_WORDS:
                        seen.setdefault(a, set()).add(t.abbr)
        # "new york" or "los angeles" could be two teams: only keep names that point at one
        return cls({a: next(iter(t)) for a, t in seen.items() if len(t) == 1})

    def find(self, text: str) -> str | None:
        t = " " + re.sub(r"[^a-z0-9 -]", " ", (text or "").lower()) + " "
        best = None
        for alias, abbr in self.aliases.items():
            if len(alias) < 2:
                continue
            if f" {alias} " in t and (best is None or len(alias) > len(best[0])):
                best = (alias, abbr)
        return best[1] if best else None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
