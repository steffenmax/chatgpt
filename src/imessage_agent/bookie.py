"""Ricky the bookie: fake-money NFL moneyline bets, per chat.

Everyone starts with BOOKIE_START bucks. A bet locks the moneyline at the time it's
placed (DraftKings via ESPN; even money if there's no line) and can only be placed
before kickoff. Finished games settle automatically from the sports poller: win pays
stake + profit, a tie refunds. Broke players can take a weekly bailout. Nothing here
is real money and nothing ever leaves this Mac; the book is a 0600 JSON file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sports import Game, american_profit, fmt_odds, now_utc

_STAKE = r"(?:\bbet\b\s*)?(?:\$?(\d{1,7})|\b(all\s*in|everything|it\s*all)\b)\s*(?:bucks?|dollars?)?\s*"
_BET = re.compile(_STAKE + r"(?:on|for)\s+(.+)$", re.IGNORECASE)
_BET_LOOSE = re.compile(r"^\s*" + _STAKE + r"(?:(?:on|for)\s+)?([a-z].+)$", re.IGNORECASE)


def parse_bet(text: str, loose: bool = False) -> tuple[int | str, str] | None:
    """'50 on the jets' -> (50, 'the jets'); 'all in on pats' -> ('all', 'pats'). None if it isn't a bet.
    loose (for /bet's argument): '50 jets' works too."""
    m = (_BET_LOOSE if loose else _BET).search(" ".join((text or "").split()))
    if not m:
        return None
    stake: int | str = int(m.group(1)) if m.group(1) else "all"
    return stake, m.group(3).strip(" ?!.")


@dataclass
class Settled:
    name: str
    team: str
    stake: int
    result: str  # won | lost | push
    delta: int  # change to the balance relative to before the bet (profit, -stake or 0)
    balance: int


class Bookie:
    def __init__(self, path: str, start: int = 1000, bailout: int = 100, clock=time.time):
        self.path = Path(path).expanduser()
        self.start = start
        self.bailout_amount = bailout
        self._clock = clock
        self.book: dict[str, Any] = self._load()

    # -- storage --
    def _load(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {"chats": {}}
        except (OSError, ValueError):
            return {"chats": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".book-", suffix=".json", dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.book, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def _chat(self, chat: str) -> dict[str, Any]:
        return self.book.setdefault("chats", {}).setdefault(chat, {"players": {}, "bets": [], "next_id": 1})

    def _player(self, chat: str, who: str, name: str) -> dict[str, Any]:
        p = self._chat(chat)["players"].setdefault(who, {"name": name, "bucks": self.start, "won": 0, "lost": 0,
                                                         "bailouts": 0, "last_bailout": None})
        if name:
            p["name"] = name
        return p

    # -- reads --
    def balance(self, chat: str, who: str, name: str) -> int:
        b = self._player(chat, who, name)["bucks"]
        self._save()
        return b

    def open_bets(self, chat: str) -> list[dict[str, Any]]:
        return [b for b in self._chat(chat)["bets"] if b["status"] == "open"]

    def open_game_ids(self) -> set[str]:
        return {b["game_id"] for c in self.book.get("chats", {}).values() for b in c["bets"] if b["status"] == "open"}

    def open_weeks(self) -> set[tuple[int, int]]:
        return {(b["week"], b["season_type"]) for c in self.book.get("chats", {}).values()
                for b in c["bets"] if b["status"] == "open"}

    def leaderboard(self, chat: str) -> list[dict[str, Any]]:
        players = self._chat(chat)["players"].values()
        at_risk = {}
        for b in self.open_bets(chat):
            at_risk[b["who"]] = at_risk.get(b["who"], 0) + b["stake"]
        rows = [{**p, "at_risk": at_risk.get(w, 0)} for w, p in self._chat(chat)["players"].items()]
        return sorted(rows, key=lambda p: p["bucks"] + p["at_risk"], reverse=True) if players else []

    # -- writes --
    def place(self, chat: str, who: str, name: str, game: Game, team: str, stake: int | str) -> dict[str, Any] | str:
        """The new bet, or a short reason it was refused."""
        p = self._player(chat, who, name)
        if game.state != "pre" or game.start <= now_utc():
            return "started"
        side = game.team(team)
        if side is None:
            return "no_game"
        amount = p["bucks"] if stake == "all" else int(stake)
        if amount <= 0:
            return "broke" if p["bucks"] <= 0 else "zero"
        if amount > p["bucks"]:
            return "too_much"
        c = self._chat(chat)
        bet = {"id": c["next_id"], "who": who, "name": name, "game_id": game.id, "week": game.week,
               "season_type": game.season_type, "team": team, "team_name": side.name,
               "opp_name": game.other(team).name, "odds": side.moneyline, "stake": amount,
               "to_win": american_profit(amount, side.moneyline), "start": game.start.isoformat(),
               "placed": self._clock(), "status": "open"}
        c["next_id"] += 1
        c["bets"].append(bet)
        p["bucks"] -= amount
        self._save()
        return bet

    def settle(self, games: list[Game]) -> dict[str, list[Settled]]:
        """Pay out bets on finished games. Returns what changed, per chat."""
        done = {g.id: g for g in games if g.completed}
        out: dict[str, list[Settled]] = {}
        for chat, c in self.book.get("chats", {}).items():
            for b in c["bets"]:
                g = done.get(b["game_id"])
                if b["status"] != "open" or g is None:
                    continue
                p = c["players"][b["who"]]
                winner = g.winner()
                if winner is None:
                    b["status"], delta = "push", 0
                    p["bucks"] += b["stake"]
                elif winner == b["team"]:
                    b["status"], delta = "won", b["to_win"]
                    p["bucks"] += b["stake"] + b["to_win"]
                    p["won"] += 1
                else:
                    b["status"], delta = "lost", -b["stake"]
                    p["lost"] += 1
                b["final"] = g.score_line()
                out.setdefault(chat, []).append(Settled(p["name"], b["team_name"], b["stake"], b["status"], delta, p["bucks"]))
        if out:
            self._save()
        return out

    def bailout(self, chat: str, who: str, name: str) -> tuple[bool, int]:
        """(granted, balance). Only when broke (nothing left, nothing riding), once a week."""
        p = self._player(chat, who, name)
        riding = any(b["who"] == who for b in self.open_bets(chat))
        week = 7 * 86400
        last = p.get("last_bailout")
        if p["bucks"] > 0 or riding or (last is not None and self._clock() - last < week):
            return False, p["bucks"]
        p["bucks"] += self.bailout_amount
        p["bailouts"] += 1
        p["last_bailout"] = self._clock()
        self._save()
        return True, p["bucks"]


def odds_line(g: Game) -> str:
    when = g.start.astimezone().strftime("%a %-I:%M%p").lower()
    return (f"{g.away.name} {fmt_odds(g.away.moneyline)} @ {g.home.name} {fmt_odds(g.home.moneyline)}"
            f"  ({when}{', ' + g.spread if g.spread else ''})")
