"""Live NFL alerts for the teams the chat follows, and the poller the bookie settles from.

One loop polls ESPN's scoreboard: every ~45 s while a watched game (a followed team's,
or one with an open bet) is live, every minute just before kickoff, every 10 minutes
otherwise. For followed teams it posts kickoff, each scoring play, halftime and the
final; the handler adds the bot's one-line reaction. What was already announced is kept
in a small state file so a restart doesn't repeat (or backfill) alerts.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from .logsafe import exc_name, log_event
from .sports import ESPN, Game, ScoringPlay, now_utc, quarter

KIND_LABEL = {"TD": "TD", "FG": "FG", "SF": "SAFETY", "D2P": "DEFENSIVE 2PT"}


def parse_teams(raw: str) -> dict[str, list[str]]:
    """'NYG=Max,Joe;NYJ=Chris Oddo' -> {'NYG': ['Max', 'Joe'], 'NYJ': ['Chris Oddo']}."""
    out: dict[str, list[str]] = {}
    for part in (raw or "").split(";"):
        if "=" not in part:
            continue
        abbr, fans = part.split("=", 1)
        abbr = abbr.strip().upper()
        if abbr:
            out[abbr] = [f.strip() for f in fans.split(",") if f.strip()]
    return out


@dataclass
class Alert:
    kind: str  # kickoff | score | half | final
    text: str  # the fact line, posted as is
    game: Game
    scorer: str | None = None  # abbreviation of the team that scored / won
    fans: dict[str, list[str]] = field(default_factory=dict)  # abbr -> fans in this game


class SportsAlerts:
    def __init__(self, espn: ESPN, teams: dict[str, list[str]], state_path: str,
                 on_alert: Callable[[Alert], Awaitable[None]] | None = None,
                 on_games: Callable[[list[Game]], Awaitable[None]] | None = None,
                 watch_extra: Callable[[], set[str]] | None = None,
                 busy: Callable[[], bool] | None = None,
                 field_goals: bool = True):
        self.espn = espn
        self.teams = teams
        self.path = Path(state_path).expanduser()
        self.on_alert = on_alert
        self.on_games = on_games  # e.g. the bookie settling finished games
        self.watch_extra = watch_extra or (lambda: set())  # game keys with open bets
        self.busy = busy or (lambda: False)  # e.g. bets waiting on a result in another league
        self.field_goals = field_goals
        self.state: dict[str, dict[str, Any]] = self._load()
        self.last_error: str | None = None
        self.games: list[Game] = []  # the last scoreboard seen

    # -- state --
    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        cutoff = (now_utc() - timedelta(days=10)).isoformat()
        self.state = {k: v for k, v in self.state.items() if v.get("start", "") >= cutoff}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".sports-", suffix=".json", dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def followed(self, g: Game) -> bool:
        return g.home.abbr in self.teams or g.away.abbr in self.teams

    def _fans(self, g: Game) -> dict[str, list[str]]:
        return {t.abbr: self.teams[t.abbr] for t in (g.home, g.away) if self.teams.get(t.abbr)}

    # -- one poll --
    async def tick(self) -> list[Alert]:
        games = self.games = await self.espn.scoreboard()
        alerts: list[Alert] = []
        for g in games:
            if self.followed(g):
                alerts += await self._game(g)
        self._save()
        if self.on_games is not None:
            await self.on_games(games)
        for a in alerts:
            log_event("sports.alert", kind=a.kind)
            if self.on_alert is not None:
                await self.on_alert(a)
        return alerts

    async def _game(self, g: Game) -> list[Alert]:
        st = self.state.get(g.id)
        fans = self._fans(g)
        if st is None:
            # First time we see this game: never backfill what already happened.
            st = self.state[g.id] = {"start": g.start.isoformat(), "kickoff": g.state != "pre",
                                     "half": g.state == "post" or g.period > 2, "final": g.state == "post", "plays": []}
            if g.state == "in":
                st["plays"] = [p.id for p in await self._plays(g)]
            return []
        out: list[Alert] = []
        if g.state == "pre":
            return out
        if not st["kickoff"]:
            st["kickoff"] = True
            out.append(Alert("kickoff", f"🏈 {g.matchup} just kicked off.", g, fans=fans))
        if g.state == "in" or (g.state == "post" and not st["final"]):
            for p in await self._plays(g):
                if p.id in st["plays"]:
                    continue
                st["plays"].append(p.id)
                if p.kind == "FG" and not self.field_goals:
                    continue
                team = g.team(p.team)
                g.home.score, g.away.score = p.home_score, p.away_score  # the score right after this play
                text = (f"🏈 {KIND_LABEL.get(p.kind, p.kind)} {team.name if team else p.team}: {p.text.rstrip('. ')}. "
                        f"{g.score_line()} ({quarter(p.period)} {p.clock})")
                out.append(Alert("score", text, g, scorer=p.team, fans=fans))
        if not st["half"] and "half" in g.detail.lower():
            st["half"] = True
            out.append(Alert("half", f"Halftime: {g.score_line()}.", g, scorer=self._leader(g), fans=fans))
        if g.state == "post" and not st["final"]:
            st["final"] = True
            st["half"] = True
            ot = "/OT" if g.period > 4 else ""
            out.append(Alert("final", f"FINAL{ot}: {g.score_line()}.", g, scorer=g.winner(), fans=fans))
        return out

    @staticmethod
    def _leader(g: Game) -> str | None:
        if g.home.score == g.away.score:
            return None
        return g.home.abbr if g.home.score > g.away.score else g.away.abbr

    async def _plays(self, g: Game) -> list[ScoringPlay]:
        try:
            return await self.espn.scoring_plays(g.id)
        except Exception as exc:  # noqa: BLE001
            log_event("sports.plays_failed", error=exc_name(exc))
            return []

    # -- loop --
    def next_wait(self, games: list[Game]) -> float:
        watched = self.watch_extra()
        live = [g for g in games if (self.followed(g) or g.key in watched)]
        if any(g.state == "in" for g in live):
            return 45.0
        if self.busy():
            return 120.0
        soon = [g for g in live if g.state == "pre" and g.start - now_utc() < timedelta(minutes=20)]
        return 60.0 if soon else 600.0

    async def run_forever(self) -> None:
        while True:
            wait = 600.0
            try:
                await self.tick()
                wait = self.next_wait(self.games)
                self.last_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_error = exc_name(exc)
                log_event("sports.poll_failed", error=self.last_error)
            await asyncio.sleep(wait)
