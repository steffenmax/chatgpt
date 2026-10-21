"""Drop watches: some places (4 Charles) release tables at a set time, and they're gone in minutes.

"ricky watch 4 charles drop at 9am" sets one up for the chat. From a minute before the drop
until DROP_WINDOW after it, the watcher checks Resy every few seconds for ANY open table: every
day on the calendar, every time, party sizes 2-6 (or just the size they named). Each new table is
posted once, with a link to grab it. Alert only: nothing is booked.

No day named: it runs at that time every day and only speaks when something opens. A day named
("9am tomorrow", "9am nov 1"): it runs that once, says how it went either way, then ends.
Watches are kept in a small 0600 JSON file so they survive restarts.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .logsafe import exc_name, log_event
from .resy import Resy, Slot, Venue, _fmt_day, _fmt_time, parse_parts

LEAD = 60.0  # start checking this many seconds before the drop
DROP_WINDOW = 15 * 60.0  # and keep checking this long after it
EVERY = 8.0  # seconds between checks inside the window
CALENDAR_DAYS = 45
MAX_FINDS = 8  # slot lookups per check
ANY_PARTY = [2, 3, 4, 5, 6]
MAX_WATCHES = 10

WATCH_ASK = re.compile(r"\b(watch|monitor|snipe|camp)\b.*\b(drop|drops|release|releases|opens?|opening)\b"
                       r"|\b(set|make|start)\b.*\bwatch\b|\bdrop watch\b", re.IGNORECASE)
LIST_ASK = re.compile(r"\b(what are you watching|what('?s| is| are) (you )?watch(ing)?|(list|show)( me)?( the| your| my)? watch(es)?|"
                      r"^\W*\w*\W*watches\W*$)", re.IGNORECASE)
STOP_ASK = re.compile(r"\b(stop|quit|cancel|kill|end|remove|delete|drop|unwatch)\b.*\bwatch(ing|es)?\b"
                      r"|\bwatch(ing|es)?\b.*\b(stop|cancel|off|remove)\b|\bunwatch\b", re.IGNORECASE)
_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_DAILY = re.compile(r"\b(every ?day|daily|each (day|morning)|every morning)\b", re.IGNORECASE)


def wants_watch(text: str) -> bool:
    return bool(WATCH_ASK.search(text or ""))


def wants_watch_list(text: str, bot_name: str = "") -> bool:
    t = re.sub(rf"\b{re.escape(bot_name)}\b,?", " ", text or "", flags=re.IGNORECASE) if bot_name else text or ""
    return bool(LIST_ASK.search(" ".join(t.split())))


def wants_watch_stop(text: str) -> bool:
    return bool(STOP_ASK.search(text or ""))


@dataclass
class DropWatch:
    id: int
    chat: str
    venue_id: int
    venue_name: str
    slug: str
    city_slug: str
    drop: int  # minutes after local midnight
    on: str = ""  # "" = every day, else an ISO date
    parties: list[int] = field(default_factory=lambda: list(ANY_PARTY))
    set_by: str = ""
    last_run: str = ""  # ISO date of the last drop it finished watching

    @property
    def venue(self) -> Venue:
        return Venue(self.venue_id, self.venue_name, "", self.slug, self.city_slug)

    def describe(self) -> str:
        when = f"{_fmt_day(date.fromisoformat(self.on))} at {_fmt_time(self.drop)}" if self.on else f"every day at {_fmt_time(self.drop)}"
        size = "any size" if self.parties == ANY_PARTY else "for " + "/".join(map(str, self.parties))
        return f"{self.venue_name}, {when}, {size}"

    def next_drop(self, now: datetime) -> datetime | None:
        """When the next drop it should watch starts (the drop time itself), or None if it's done."""
        hh, mm = divmod(self.drop, 60)
        if self.on:
            d = date.fromisoformat(self.on)
            return None if self.last_run == self.on else datetime.combine(d, datetime.min.time()).replace(hour=hh, minute=mm)
        today = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if self.last_run == now.date().isoformat() or now > today + timedelta(seconds=DROP_WINDOW):
            return today + timedelta(days=1)
        return today


@dataclass
class Spec:
    venue: str
    drop: int | None
    on: date | None
    parties: list[int]


def parse_watch(text: str, today: date, bot_name: str = "") -> Spec:
    """'watch 4 charles drop at 9am nov 1 for 4' -> venue, drop time, day (None = daily), sizes."""
    raw = " ".join((text or "").replace("’", "'").split())
    t = re.sub(rf"\b{re.escape(bot_name)}\b,?", " ", raw, flags=re.IGNORECASE) if bot_name else raw
    on = None
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t, re.IGNORECASE)
    if m:
        try:
            d = date(today.year, _MONTHS[m.group(1).lower()[:3]], int(m.group(2)))
            on = d if d >= today else date(today.year + 1, d.month, d.day)
        except ValueError:
            pass
        t = t[:m.start()] + " " + t[m.end():]
    daily = bool(_DAILY.search(t))
    t = _DAILY.sub(" ", t)
    p = parse_parts(t, today)
    if on is None and not daily:
        on = p.day
    drop = None
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a|p)\b|\b(\d{1,2}):(\d{2})\b", t, re.IGNORECASE)
    if m:
        hh = int(m.group(1) or m.group(4))
        mm = int(m.group(2) or m.group(5) or 0)
        ap = (m.group(3) or "").lower()[:1]
        if ap == "p" and hh < 12:
            hh += 12
        elif ap == "a" and hh == 12:
            hh = 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            drop = hh * 60 + mm
    venue = re.sub(r"\b(set|make|start|up|a|an|the|watch|watching|monitor|snipe|camp|drop|drops|dropping|release|releases|"
                   r"opens?|opening|when|it|they|for|at|on|any|size|sizes|time|times|date|dates|day|tables?|res|resy|"
                   r"reservations?|alert|me|us|pls|please|can|you|yo|of)\b", " ", p.venue, flags=re.IGNORECASE)
    return Spec(" ".join(venue.split()), drop, on, [p.party] if p.party else list(ANY_PARTY))


class DropWatcher:
    def __init__(self, resy: Resy, path: str, send: Callable[[str, str], Awaitable[None]],
                 now: Callable[[], datetime] = datetime.now, sleep=asyncio.sleep):
        self.resy = resy
        self.path = Path(path).expanduser()
        self.send = send
        self._now = now
        self._sleep = sleep
        self._wake = asyncio.Event()
        self._running: set[int] = set()

    # -- the saved watches --
    def watches(self) -> list[DropWatch]:
        try:
            return [DropWatch(**w) for w in json.loads(self.path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            return []

    def _save(self, items: list[DropWatch]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".watch-", suffix=".json", dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([asdict(w) for w in items], f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def add(self, chat: str, venue: Venue, drop: int, on: date | None, parties: list[int], set_by: str) -> DropWatch:
        items = [w for w in self.watches() if not (w.chat == chat and w.venue_id == venue.id)]  # one per place per chat
        w = DropWatch(max([x.id for x in self.watches()] + [0]) + 1, chat, venue.id, venue.name, venue.slug,
                      venue.city_slug, drop, on.isoformat() if on else "", parties, set_by)
        self._save((items + [w])[-MAX_WATCHES:])
        log_event("resy.watch_added", chat_guid=chat, kind="once" if on else "daily")
        self._wake.set()
        return w

    def remove(self, chat: str, name: str = "") -> list[DropWatch]:
        """Drop this chat's watches matching `name` (all of them if no name). Returns the removed ones."""
        words = [x for x in re.findall(r"[a-z0-9]+", name.lower()) if len(x) > 1]
        gone, keep = [], []
        for w in self.watches():
            hit = w.chat == chat and (not words or any(x in w.venue_name.lower() for x in words))
            (gone if hit else keep).append(w)
        if gone:
            self._save(keep)
            self._wake.set()
        return gone

    def _finish(self, watch: DropWatch, day: date) -> None:
        items = []
        for w in self.watches():
            if w.id == watch.id:
                if w.on:
                    continue  # a one-day watch is done
                w.last_run = day.isoformat()
            items.append(w)
        self._save(items)

    # -- the scheduler --
    async def run(self) -> None:
        while True:
            now = self._now()
            soonest = 3600.0
            for w in self.watches():
                at = w.next_drop(now)
                if at is None or w.id in self._running:
                    continue
                start = (at - now).total_seconds() - LEAD
                if start <= 0:
                    self._running.add(w.id)
                    asyncio.create_task(self._run_drop(w, at), name=f"drop-watch-{w.id}")
                else:
                    soonest = min(soonest, start)
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=max(1.0, min(soonest, 3600.0)))
            except asyncio.TimeoutError:
                pass

    async def _run_drop(self, w: DropWatch, at: datetime) -> None:
        seen: set[tuple[str, int, str]] = set()
        found_any = False
        log_event("resy.watch_started", chat_guid=w.chat)
        try:
            end = at + timedelta(seconds=DROP_WINDOW)
            if self._now() >= end:  # missed it (the Mac was off or asleep)
                log_event("resy.watch_missed", chat_guid=w.chat)
                if w.on:
                    await self.send(w.chat, f"missed the {w.venue_name} drop, i was knocked out. watch is off")
                self._finish(w, at.date())
                return
            while self._now() < end:
                if w.id not in {x.id for x in self.watches()}:
                    return  # stopped mid-drop
                try:
                    finds = await self.check(w)
                except httpx.HTTPError as exc:
                    log_event("resy.watch_check_failed", chat_guid=w.chat, error=exc_name(exc))
                    finds = []
                new = [(d, p, s) for d, p, s in finds if (d.isoformat(), p, s.start.isoformat()) not in seen]
                for d, p, s in new:
                    seen.add((d.isoformat(), p, s.start.isoformat()))
                if new:
                    found_any = True
                    await self.send(w.chat, alert_text(w, new))
                    log_event("resy.watch_found", chat_guid=w.chat, count=len(new))
                await self._sleep(EVERY)
            if not found_any and w.on:
                await self.send(w.chat, f"{w.venue_name} {_fmt_time(w.drop)} drop came and went, nothing opened up "
                                        f"for any size i was looking at. done watching it")
            log_event("resy.watch_done", chat_guid=w.chat, count=len(seen))
            self._finish(w, at.date())
        finally:
            self._running.discard(w.id)
            self._wake.set()

    async def check(self, w: DropWatch) -> list[tuple[date, int, Slot]]:
        """Every open table right now: (day, party size, slot)."""
        today = self._now().date()
        open_days: list[tuple[date, int]] = []
        for p in w.parties:
            for d in await self.resy.calendar(w.venue, p, today, days=CALENDAR_DAYS):
                open_days.append((d, p))
        out = []
        for d, p in sorted(open_days, reverse=True)[:MAX_FINDS]:  # the newly dropped day first
            for s in await self.resy.slots(w.venue, d, p):
                out.append((d, p, s))
        return out


def alert_text(w: DropWatch, finds: list[tuple[date, int, Slot]]) -> str:
    """'🚨 4 Charles just opened up' then a line per day: sizes and times, and a link to grab it."""
    by_day: dict[date, dict[int, list[Slot]]] = {}
    for d, p, s in finds:
        by_day.setdefault(d, {}).setdefault(p, []).append(s)
    lines = [f"🚨 {w.venue_name} just opened up, go go go"]
    for d in sorted(by_day)[:4]:
        sizes = by_day[d]
        bits = []
        for p in sorted(sizes):
            times = sorted({s.start for s in sizes[p]})
            shown = ", ".join(_fmt_time(t.hour * 60 + t.minute) for t in times[:4]) + (f" +{len(times) - 4}" if len(times) > 4 else "")
            bits.append(f"for {p}: {shown}")
        lines.append(f"{_fmt_day(d)}: " + " · ".join(bits))
        lines.append(w.venue.link(d, min(sizes)))
    if len(by_day) > 4:
        lines.append(f"(+{len(by_day) - 4} more days)")
    return "\n".join(lines)
