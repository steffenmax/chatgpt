"""Reminders: deterministic time parsing, a small JSON store, and a scheduler.

Enabled with REMINDERS_FILE. The file holds reminder text (message-derived content)
so it is opt-in, 0600, and documented in the README privacy table. Times are parsed
by rules, never by a model, so a reminder fires when the person said, not when a
model guessed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

from .logsafe import exc_name, log_event

MAX_PER_CHAT = 50
MAX_DAYS_AHEAD = 60

_UNITS = {"m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
          "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
          "d": 86400, "day": 86400, "days": 86400}
_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_REL = r"in\s+(\d+)\s*(m|mins?|minutes?|h|hrs?|hours?|d|days?)\b"
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"
_ABS = (
    r"(?:(tomorrow|today|" + "|".join(w + "[a-z]*" for w in _WEEKDAYS) + r")\s+)?"
    r"(?:at\s+)?" + _CLOCK
)
_ABS_DAY_ONLY = r"(tomorrow|" + "|".join(w + "[a-z]*" for w in _WEEKDAYS) + r")\b"


@dataclass
class Reminder:
    id: int
    chat_guid: str
    who: str
    text: str
    due: float  # unix epoch seconds
    created: float


def _clock(now: datetime, day_word: str | None, hh: str, mm: str | None, ampm: str | None) -> datetime | None:
    hour, minute = int(hh), int(mm or 0)
    if ampm:
        if hour < 1 or hour > 12:
            return None
        hour = hour % 12 + (12 if ampm == "pm" else 0)
    elif hour > 23 or minute > 59:
        return None
    base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_word in (None, "today"):
        if base <= now:
            if not ampm and hour < 12:
                base += timedelta(hours=12)  # "at 7" said in the afternoon means 7pm
            if base <= now:
                base += timedelta(days=1)
        return base
    if day_word == "tomorrow":
        return base + timedelta(days=1)
    target = _WEEKDAYS.index(day_word[:3])
    delta = (target - now.weekday()) % 7
    if delta == 0 and base <= now:
        delta = 7
    return base + timedelta(days=delta)


def parse(arg: str, now: datetime) -> tuple[datetime | None, str]:
    """Return (due, what). due is None when no time could be read."""
    s = " ".join(arg.split())
    if not s:
        return None, ""
    low = s.lower()

    # "in 20m <what>" / "<what> in 20m"
    for pat in (r"^" + _REL + r"\s+(.+)$", r"^(.+?)\s+" + _REL + r"$"):
        m = re.match(pat, low)
        if m:
            g = m.groups()
            n, unit, what = (g[0], g[1], g[2]) if pat.startswith("^in") else (g[1], g[2], g[0])
            secs = int(n) * _UNITS[unit]
            if secs <= 0:
                return None, ""
            return now + timedelta(seconds=secs), s[m.start(3):m.end(3)] if pat.startswith("^in") else s[m.start(1):m.end(1)]

    # "[tomorrow|sat] [at] 7[:30][pm] <what>" / "<what> [tomorrow|sat] [at] 7pm"
    for pat, what_idx in ((r"^" + _ABS + r"\s+(.+)$", 5), (r"^(.+?)\s+" + _ABS + r"$", 1)):
        m = re.match(pat, low)
        if m:
            g = m.groups()
            day, hh, mm, ampm = (g[0], g[1], g[2], g[3]) if what_idx == 5 else (g[1], g[2], g[3], g[4])
            due = _clock(now, day, hh, mm, ampm)
            if due is None:
                continue
            what = s[m.start(what_idx):m.end(what_idx)]
            return due, what

    # "tomorrow <what>" / "<what> tomorrow" -> 9am that day
    for pat, what_idx in ((r"^" + _ABS_DAY_ONLY + r"\s+(.+)$", 2), (r"^(.+?)\s+" + _ABS_DAY_ONLY + r"$", 1)):
        m = re.match(pat, low)
        if m:
            day = m.group(1) if what_idx == 2 else m.group(2)
            due = _clock(now, day, "9", "00", "am")
            return due, s[m.start(what_idx):m.end(what_idx)]
    return None, ""


def describe(due: datetime, now: datetime) -> str:
    delta = due - now
    if delta < timedelta(hours=12) and due.date() == now.date():
        mins = int(delta.total_seconds() // 60)
        if mins < 60:
            return f"in {max(mins, 1)} min"
        return f"at {due.strftime('%-I:%M%p').lower()}"
    if due.date() == now.date() + timedelta(days=1):
        return f"tomorrow at {due.strftime('%-I:%M%p').lower()}"
    return f"{due.strftime('%a')} at {due.strftime('%-I:%M%p').lower()}"


class ReminderStore:
    def __init__(self, path: str, clock: Callable[[], float] = time.time):
        self.path = Path(path).expanduser()
        self._clock = clock
        self._items: list[Reminder] = []
        self._next_id = 1
        self._load()

    # -- persistence --
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log_event("reminders.load_failed", error=exc_name(exc))
            return
        self._items = [Reminder(**r) for r in raw.get("items", [])]
        self._next_id = int(raw.get("next_id", 1))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        fd, tmp = tempfile.mkstemp(prefix=".reminders-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"next_id": self._next_id, "items": [asdict(r) for r in self._items]}, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- API --
    def add(self, chat_guid: str, who: str, text: str, due: float) -> Reminder | None:
        if len(self.pending(chat_guid)) >= MAX_PER_CHAT:
            return None
        r = Reminder(self._next_id, chat_guid, who, text[:300], due, self._clock())
        self._next_id += 1
        self._items.append(r)
        self._save()
        log_event("reminder.set", chat_guid=chat_guid, count=len(self.pending(chat_guid)))
        return r

    def pending(self, chat_guid: str) -> list[Reminder]:
        return sorted((r for r in self._items if r.chat_guid == chat_guid), key=lambda r: r.due)

    def cancel(self, chat_guid: str, rid: int) -> bool:
        before = len(self._items)
        self._items = [r for r in self._items if not (r.chat_guid == chat_guid and r.id == rid)]
        if len(self._items) != before:
            self._save()
            log_event("reminder.cancelled", chat_guid=chat_guid)
            return True
        return False

    def pop_due(self) -> list[Reminder]:
        now = self._clock()
        due = [r for r in self._items if r.due <= now]
        if due:
            self._items = [r for r in self._items if r.due > now]
            self._save()
        return due

    def next_due(self) -> float | None:
        return min((r.due for r in self._items), default=None)

    async def run(self, fire: Callable[[Reminder], Awaitable[None]], poll_seconds: float = 30.0) -> None:
        while True:
            for r in self.pop_due():
                try:
                    await fire(r)
                    log_event("reminder.fired", chat_guid=r.chat_guid)
                except Exception as exc:  # noqa: BLE001
                    log_event("reminder.fire_failed", chat_guid=r.chat_guid, error=exc_name(exc))
            nxt = self.next_due()
            wait = poll_seconds if nxt is None else min(poll_seconds, max(0.05, nxt - self._clock()))
            await asyncio.sleep(wait)
