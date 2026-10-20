"""Scores, scoring plays and odds from ESPN's public site API (no key, no account).

Leagues: NFL, college football, NBA, WNBA, NHL, MLB, men's college basketball.

Used by the sports alerts (alerts.py) and the bookie (bookie.py). Read-only: nothing
from the chat is ever sent to ESPN; requests are just "this week's scoreboard" and
"scoring plays for game N".
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

import httpx

BASE = "https://site.api.espn.com/apis/site/v2/sports"
SCOREBOARD = f"{BASE}/football/nfl/scoreboard"
SUMMARY = f"{BASE}/football/nfl/summary"

# key -> (ESPN path, label, "week" = scoreboard is a week of games, "day" = one date per request)
LEAGUES = {
    "nfl": ("football/nfl", "NFL", "week"),
    "cfb": ("football/college-football", "college football", "week"),
    "nba": ("basketball/nba", "NBA", "day"),
    "wnba": ("basketball/wnba", "WNBA", "day"),
    "nhl": ("hockey/nhl", "NHL", "day"),
    "mlb": ("baseball/mlb", "MLB", "day"),
    "cbb": ("basketball/mens-college-basketball", "college basketball", "day"),
}
# Words that pick a league: "/bet 50 nhl jets", "/odds nba".
LEAGUE_WORDS = {
    "nfl": "nfl", "football": "nfl", "cfb": "cfb", "ncaaf": "cfb", "college football": "cfb",
    "nba": "nba", "basketball": "nba", "wnba": "wnba", "nhl": "nhl", "hockey": "nhl",
    "mlb": "mlb", "baseball": "mlb", "cbb": "cbb", "ncaab": "cbb", "college basketball": "cbb",
}

# Nicknames people actually type, on top of ESPN's names (location, nickname, abbreviation).
EXTRA_ALIASES = {
    # NBA / NHL / MLB (keys are "league:ABBR"; plain keys are NFL)
    "nba:NY": ["knicks", "knickerbockers", "the garden"], "nba:BKN": ["nets"], "nba:PHI": ["sixers", "76ers"],
    "nba:BOS": ["celtics", "cs"], "nba:LAL": ["lakers", "lebron"], "nba:GS": ["dubs", "warriors"],
    "nhl:NYR": ["blueshirts", "rangers"], "nhl:NYI": ["isles", "islanders"], "nhl:NJ": ["devils"],
    "nhl:PHI": ["flyers"], "mlb:NYY": ["yanks", "bronx bombers", "yankees"], "mlb:NYM": ["amazins", "mets"],
    "mlb:PHI": ["phils", "phillies"], "mlb:BOS": ["sox", "red sox"],
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


ET = ZoneInfo("America/New_York")


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
    league: str = "nfl"

    @property
    def key(self) -> str:
        """Unique across leagues."""
        return f"{self.league}:{self.id}"

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


def parse_scoreboard(data: dict[str, Any], league: str = "nfl") -> list[Game]:
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
                completed=bool(typ.get("completed")), league=league,
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
    def __init__(self, timeout: float = 15.0, client: httpx.AsyncClient | None = None, clock=None):
        self._client = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "imessage-agent"})
        self._clock = clock or now_utc

    async def aclose(self) -> None:
        await self._client.aclose()

    async def scoreboard(self, week: int | None = None, season_type: int | None = None, league: str = "nfl",
                         date: str | None = None) -> list[Game]:
        params: dict[str, Any] = {}
        if week:
            params = {"week": week, "seasontype": season_type or 2}
        elif date:
            params = {"dates": date}
        r = await self._client.get(f"{BASE}/{LEAGUES[league][0]}/scoreboard", params=params)
        r.raise_for_status()
        return parse_scoreboard(r.json(), league)

    async def upcoming(self, league: str = "nfl") -> list[Game]:
        """Games to bet on: this week's (the next week's once it's all done), or today through two days out."""
        if LEAGUES[league][2] == "week":
            games = await self.scoreboard(league=league)
            if games and all(g.state == "post" for g in games):
                nxt = await self.scoreboard(games[0].week + 1, games[0].season_type, league=league)
                if nxt:
                    return nxt
            return games
        days = [(self._clock().astimezone(ET) + timedelta(days=d)).strftime("%Y%m%d") for d in range(3)]
        boards = await asyncio.gather(*(self.scoreboard(league=league, date=d) for d in days), return_exceptions=True)
        return [g for b in boards if isinstance(b, list) for g in b]

    async def upcoming_all(self, leagues: list[str] | None = None) -> list[Game]:
        leagues = leagues or list(LEAGUES)
        boards = await asyncio.gather(*(self.upcoming(lg) for lg in leagues), return_exceptions=True)
        return [g for b in boards if isinstance(b, list) for g in b]

    async def scoring_plays(self, game_id: str, league: str = "nfl") -> list[ScoringPlay]:
        r = await self._client.get(f"{BASE}/{LEAGUES[league][0]}/summary", params={"event": game_id})
        r.raise_for_status()
        return parse_scoring_plays(r.json())


CORE = "https://sports.core.api.espn.com/v2/sports"
LIVE_PROVIDER = "200"  # "DraftKings - Live Odds"
FINAL_PERIOD = {"nfl": 4, "cfb": 4, "nba": 4, "wnba": 4, "cbb": 2, "nhl": 3, "mlb": 9}


def prob_to_american(p: float, vig: float = 0.025) -> int:
    """A win probability as American odds, shaded a little in the house's favor."""
    p = min(0.99, max(0.01, p + vig))
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def clock_seconds(clock: str) -> int | None:
    m = re.fullmatch(r"\s*(\d+):(\d{2})(?:\.\d+)?\s*", clock or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def too_late_to_bet(g: Game) -> bool:
    """Book closes on a live game in the last two minutes of regulation, and in overtime / extras."""
    final = FINAL_PERIOD.get(g.league, 4)
    if g.period > final:
        return True
    if g.league == "mlb":
        return g.period >= final
    secs = clock_seconds(g.clock)
    return g.period == final and secs is not None and secs < 120


async def live_odds(espn: "ESPN", g: Game) -> tuple[int, int] | None:
    """(away, home) live moneylines: DraftKings' live line, else ESPN's live win probability."""
    sport, league = LEAGUES[g.league][0].split("/")
    try:
        r = await espn._client.get(f"{CORE}/{sport}/leagues/{league}/events/{g.id}/competitions/{g.id}/odds")
        r.raise_for_status()
        for item in r.json().get("items") or []:
            if str((item.get("provider") or {}).get("id")) == LIVE_PROVIDER:
                away = _int((item.get("awayTeamOdds") or {}).get("moneyLine"))
                home = _int((item.get("homeTeamOdds") or {}).get("moneyLine"))
                if away and home:
                    return away, home
    except (httpx.HTTPError, ValueError):
        pass
    try:
        r = await espn._client.get(f"{BASE}/{LEAGUES[g.league][0]}/summary", params={"event": g.id})
        r.raise_for_status()
        wp = r.json().get("winprobability") or []
        if wp:
            home_p = float(wp[-1]["homeWinPercentage"])
            return prob_to_american(1 - home_p), prob_to_american(home_p)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        pass
    return None


# Margins that make a late game a sweat or a blowout, per league.
CLOSE_MARGIN = {"nfl": 8, "cfb": 8, "nba": 5, "wnba": 5, "cbb": 5, "nhl": 1, "mlb": 2}
BLOWOUT_MARGIN = {"nfl": 21, "cfb": 24, "nba": 20, "wnba": 18, "cbb": 20, "nhl": 4, "mlb": 6}


def past_halfway(g: Game) -> bool:
    """Halftime or later: the half in football/basketball, after two periods in hockey, after five innings."""
    if g.state != "in":
        return False
    if g.league == "mlb":
        return g.period >= 6
    if g.league == "nhl":
        d = g.detail.lower()
        return g.period >= 3 or (g.period == 2 and ("end" in d or "int" in d))
    halves = 1 if g.league == "cbb" else 2
    return g.period > halves or "half" in g.detail.lower()


def late_game(g: Game) -> str | None:
    """'close' or 'blowout' once a game is in its last stretch (5 minutes left, or the 8th inning), else None."""
    if g.state != "in":
        return None
    final = FINAL_PERIOD.get(g.league, 4)
    if g.league == "mlb":
        late = g.period >= 8
    else:
        secs = clock_seconds(g.clock)
        late = g.period > final or (g.period == final and secs is not None and secs <= 300)
    if not late:
        return None
    margin = abs(g.home.score - g.away.score)
    if margin <= CLOSE_MARGIN.get(g.league, 5):
        return "close"
    if margin >= BLOWOUT_MARGIN.get(g.league, 20):
        return "blowout"
    return None


def et_date(dt: datetime) -> str:
    """The ESPN 'dates=' value for a game (US Eastern calendar day)."""
    return dt.astimezone(ET).strftime("%Y%m%d")


def league_hint(text: str) -> str | None:
    t = " " + re.sub(r"[^a-z ]", " ", (text or "").lower()) + " "
    for word in sorted(LEAGUE_WORDS, key=len, reverse=True):
        if f" {word} " in t:
            return LEAGUE_WORDS[word]
    return None


# Abbreviations that are everyday words ("no", "ten", "min") don't count as team names.
_COMMON_WORDS = {"no", "car", "den", "min", "sea", "ten", "was", "ind", "la", "ny"}


@dataclass
class TeamIndex:
    """Find a team from what people type ('the jets', 'big blue', 'knicks'). Teams are (league, abbr)."""

    aliases: dict[str, set[tuple[str, str]]] = field(default_factory=dict)

    @classmethod
    def from_games(cls, games: list[Game]) -> "TeamIndex":
        idx = cls()
        for g in games:
            for t in (g.home, g.away):
                extra = EXTRA_ALIASES.get(f"{g.league}:{t.abbr}", []) + (EXTRA_ALIASES.get(t.abbr, []) if g.league == "nfl" else [])
                for a in (t.abbr, t.name, t.full, t.full.rsplit(" ", 1)[0], *extra):
                    a = a.lower().strip()
                    if a and a not in _COMMON_WORDS:
                        idx.aliases.setdefault(a, set()).add((g.league, t.abbr))
        return idx

    def candidates(self, text: str) -> set[tuple[str, str]]:
        """Every team the longest matching name could mean ('jets' -> NFL and NHL Jets)."""
        t = " " + re.sub(r"[^a-z0-9 -]", " ", (text or "").lower()) + " "
        best: tuple[str, set[tuple[str, str]]] | None = None
        for alias, teams in self.aliases.items():
            if len(alias) >= 2 and f" {alias} " in t and (best is None or len(alias) > len(best[0])):
                best = (alias, teams)
        return set(best[1]) if best else set()

    def find(self, text: str) -> tuple[str, str] | None:
        """The one team the text means, or None (no match, or several teams/leagues)."""
        c = self.candidates(text)
        return next(iter(c)) if len(c) == 1 else None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
