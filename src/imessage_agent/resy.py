"""Resy reservations: find a table, and book it once enough people in the chat say yes.

Uses Resy's own web API (not an official public API; their terms don't cover automated use,
so the account could be limited). Searching and availability need only Resy's public web key.
Booking uses the owner's login token (RESY_AUTH_TOKEN, copied by the owner from a logged-in
browser) and the card already on that Resy account: no password or card number ever passes
through the bot. The chat side (fee notice first, then N distinct yes's) lives in the handler.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .logsafe import log_event

API = "https://api.resy.com"
PUBLIC_KEY = "VbWk7s3L4KiK5fzlO7JD3Q5EYolJI7n5"  # the key resy.com's own web app sends
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")

RESY_ASK = re.compile(r"\b(resy|reservations?|rezzy|res for|res at|book (?:a |us a )?table|table for \d+|get (?:us )?a table)\b",
                      re.IGNORECASE)
CANCEL_ASK = re.compile(r"\bcancel\b.*\b(res|resy|reservation|table|booking)\b|\b(res|resy|reservation)\b.*\bcancel", re.IGNORECASE)
YES = re.compile(r"^\s*(y|ya|yes+|yea+h?|yeah+|yup|yep|yessir|i'?m in|im in|count me in|in|down|bet|let'?s go+|lets go+|"
                 r"ok(?:ay)?|sure|sounds good|👍|🙋‍♂️|🙋)\s*[!.]*\s*$", re.IGNORECASE)
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12}
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_ABBR = {d[:3]: i for i, d in enumerate(_DAYS)} | {"tues": 1, "thur": 3, "thurs": 3}


# A bare number is a time only if what follows isn't part of a name ("at 4 charles prime rib").
_TIME_END = (r"(?=\s*$|\s+(?:at|on|for|this|next|tonight|today|tomorrow|tmrw|with|please|pls|"
             + "|".join(_DAYS) + "|" + "|".join(_DAY_ABBR) + r")\b|\s*[,.!?])")


def wants_resy(text: str) -> bool:
    return bool(RESY_ASK.search(text or ""))


def wants_cancel(text: str) -> bool:
    return bool(CANCEL_ASK.search(text or ""))


def is_yes(text: str) -> bool:
    return bool(YES.match(text or ""))


@dataclass
class Request:
    venue: str
    day: date
    minutes: int  # minutes after midnight
    party: int

    @property
    def when(self) -> str:
        return f"{_fmt_day(self.day)} {_fmt_time(self.minutes)}"


def _fmt_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{(h % 12) or 12}:{m:02d}{'am' if h < 12 else 'pm'}"


def _fmt_day(d: date) -> str:
    return d.strftime("%a %b %-d")


def parse_request(text: str, today: date, bot_name: str = "") -> Request | str:
    """'get resy for carbone this saturday at 6 3 people' -> Request, or a short reason it can't be read."""
    t = " " + " ".join((text or "").replace("’", "'").split()) + " "
    if bot_name:
        t = re.sub(rf"\b{re.escape(bot_name)}\b,?", " ", t, flags=re.IGNORECASE)
    # party size
    party = None
    for pat in (r"\b(\d{1,2}|" + "|".join(_NUMBERS) + r")\s*(?:people|ppl|persons|guests|of us|heads|pax)\b",
                r"\b(?:party of|table for|for)\s+(\d{1,2}|" + "|".join(_NUMBERS) + r")\b(?!\s*(?:am|pm|:\d))"):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            party = int(_NUMBERS.get(m.group(1).lower(), m.group(1)) if not m.group(1).isdigit() else m.group(1))
            t = t[:m.start()] + " " + t[m.end():]
            break
    # time
    minutes = None
    # most explicit first: "8pm" / "7:30" beat "at 4" (which might be "4 Charles Prime Rib")
    m = (re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t, re.IGNORECASE)
         or re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm|a|p)?\b", t, re.IGNORECASE)
         or re.search(rf"\b(?:at|@|around)\s+(\d{{1,2}})\b{_TIME_END}()()", t, re.IGNORECASE)
         or re.search(rf"(?<![\d/])\b(1[0-2]|[1-9])\b(?![\d/]){_TIME_END}()()", t, re.IGNORECASE))  # a lone "7" = the time
    if m:
        groups = m.groups()
        hh = int(groups[0])
        mm = int(groups[1]) if groups[1] and groups[1].isdigit() else 0
        ap = next((g for g in groups[1:] if g and not g.isdigit()), "")
        ap = ap.lower()[:1]
        if ap == "p" and hh < 12:
            hh += 12
        elif ap == "a" and hh == 12:
            hh = 0
        elif not ap and 1 <= hh <= 11:
            hh += 12  # "at 6" for a dinner reservation means 6pm
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            minutes = hh * 60 + mm
            t = t[:m.start()] + " " + t[m.end():]
    # day
    day = None
    low = t.lower()
    if re.search(r"\b(tonight|today)\b", low):
        day = today
    elif re.search(r"\btomorrow|tmrw|tmr\b", low):
        day = today + timedelta(days=1)
    else:
        m = re.search(r"\b(?:(this|next)\s+)?(" + "|".join(_DAYS) + r"|" + "|".join(_DAY_ABBR) + r")\b", low)
        if m:
            name = m.group(2)
            idx = _DAYS.index(name) if name in _DAYS else _DAY_ABBR[name]
            ahead = (idx - today.weekday()) % 7  # "saturday"/"this saturday": the coming one (today if it's Saturday)
            if m.group(1) == "next" and ahead == 0:
                ahead = 7
            day = today + timedelta(days=ahead)
        else:
            m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", low)
            if m:
                try:
                    d = date(today.year, int(m.group(1)), int(m.group(2)))
                    day = d if d >= today else date(today.year + 1, d.month, d.day)
                except ValueError:
                    day = None
    t = re.sub(r"\b(this|next|tonight|today|tomorrow|tmrw|tmr|" + "|".join(_DAYS) + "|" + "|".join(_DAY_ABBR) + r")\b|\b\d{1,2}/\d{1,2}\b",
               " ", t, flags=re.IGNORECASE)
    # whatever's left is the restaurant
    t = re.sub(r"\b(can you|could you|pls|please|yo|get|grab|find|book|make|snag|lock in|me|us|a|an|the|resy|rezzy|reservations?|"
               r"res|table|for|at|on|in|tonight|around|dinner|lunch|brunch|spot)\b", " ", t, flags=re.IGNORECASE)
    venue = " ".join(re.sub(r"[^\w&' -]", " ", t).split()).strip(" -'")
    if not venue:
        return "where? like: ricky get resy for carbone saturday at 7 for 4"
    if party is None:
        return "how many people?"
    if minutes is None:
        return "what time?"
    if day is None:
        return "what day?"
    if not 1 <= party <= 20:
        return "party size has to be 1 to 20"
    return Request(venue, day, minutes, party)


@dataclass
class Venue:
    id: int
    name: str
    city: str
    slug: str
    city_slug: str

    def link(self, day: date, party: int) -> str:
        return f"https://resy.com/cities/{self.city_slug}/{self.slug}?date={day.isoformat()}&seats={party}"


@dataclass
class Slot:
    token: str
    start: datetime
    kind: str
    cancel_fee: float | None
    deposit: float | None
    cancel_cutoff_hours: int | None

    @property
    def minutes(self) -> int:
        return self.start.hour * 60 + self.start.minute

    def fee_line(self, party: int, owner: str) -> str:
        """The fee notice, said before anyone counts yes's. '' if there's no fee."""
        parts = []
        if self.deposit:
            parts.append(f"a ${self.deposit:g} per person deposit (${self.deposit * party:g} total) up front")
        if self.cancel_fee:
            cut = f" if it's cancelled within {self.cancel_cutoff_hours}h or nobody shows" if self.cancel_cutoff_hours else " to cancel or no-show"
            parts.append(f"a ${self.cancel_fee:g} per person fee (${self.cancel_fee * party:g} total){cut}")
        return f"⚠️ heads up: there's {' and '.join(parts)}, charged to {owner}'s card." if parts else ""


@dataclass
class Booking:
    chat: str
    venue: str
    when: str
    party: int
    resy_token: str
    reservation_id: str
    requested_by: str


class Resy:
    def __init__(self, auth_token: str = "", api_key: str = PUBLIC_KEY, lat: float = 40.7128, lon: float = -74.006,
                 bookings_path: str | None = None, client: httpx.AsyncClient | None = None):
        self.auth_token = auth_token
        self.lat, self.lon = lat, lon
        headers = {"Authorization": f'ResyAPI api_key="{api_key}"', "Origin": "https://resy.com",
                   "Referer": "https://resy.com/", "User-Agent": UA, "Accept": "application/json"}
        if auth_token:
            headers |= {"X-Resy-Auth-Token": auth_token, "X-Resy-Universal-Auth": auth_token}
        self._client = client or httpx.AsyncClient(timeout=20.0, headers=headers)
        if client is not None:
            self._client.headers.update(headers)
        self.bookings_path = Path(bookings_path).expanduser() if bookings_path else None

    @property
    def can_book(self) -> bool:
        return bool(self.auth_token)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def find_venue(self, query: str) -> Venue | None:
        r = await self._client.post(f"{API}/3/venuesearch/search", json={
            "query": query, "geo": {"latitude": self.lat, "longitude": self.lon}, "per_page": 5, "types": ["venue"]})
        r.raise_for_status()
        hits = (r.json().get("search") or {}).get("hits") or []
        if not hits:
            return None
        h = hits[0]
        loc = h.get("location") or {}
        return Venue(int((h.get("id") or {}).get("resy")), str(h.get("name") or query), str(loc.get("name") or h.get("locality") or ""),
                     str(h.get("url_slug") or ""), str(loc.get("code") or "ny"))

    async def slots(self, venue: Venue, day: date, party: int) -> list[Slot]:
        r = await self._client.get(f"{API}/4/find", params={"lat": 0, "long": 0, "day": day.isoformat(),
                                                             "party_size": party, "venue_id": venue.id})
        r.raise_for_status()
        out = []
        for v in (r.json().get("results") or {}).get("venues") or []:
            for s in v.get("slots") or []:
                try:
                    start = datetime.strptime(str((s.get("date") or {}).get("start")), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                pay = s.get("payment") or {}
                cut = pay.get("secs_cancel_cut_off")
                out.append(Slot(str((s.get("config") or {}).get("token") or ""), start, str((s.get("config") or {}).get("type") or ""),
                                _money(pay.get("cancellation_fee")), _money(pay.get("deposit_fee")),
                                int(cut) // 3600 if isinstance(cut, (int, float)) and cut else None))
        seen, unique = set(), []
        for s in sorted(out, key=lambda s: s.start):
            if s.token and (s.start, s.kind) not in seen:
                seen.add((s.start, s.kind))
                unique.append(s)
        return unique

    async def _payment_method(self) -> int | None:
        r = await self._client.get(f"{API}/2/user")
        r.raise_for_status()
        methods = r.json().get("payment_methods") or []
        default = next((m for m in methods if m.get("is_default")), methods[0] if methods else None)
        return int(default["id"]) if default and default.get("id") is not None else None

    async def book(self, slot: Slot, party: int) -> tuple[str, str]:
        """(resy_token, reservation_id). Raises on any failure."""
        day = slot.start.date().isoformat()
        r = await self._client.post(f"{API}/3/details", json={"commit": 1, "config_id": slot.token, "day": day, "party_size": party})
        if r.status_code >= 400:  # older form of the same call
            r = await self._client.get(f"{API}/3/details", params={"config_id": slot.token, "day": day, "party_size": party})
        r.raise_for_status()
        book_token = ((r.json().get("book_token") or {}).get("value"))
        if not book_token:
            raise RuntimeError("no book token")
        form = {"book_token": book_token, "source_id": "resy.com-venue-details"}
        pm = await self._payment_method()
        if pm is not None:
            form["struct_payment_method"] = json.dumps({"id": pm})
        r = await self._client.post(f"{API}/3/book", data=form)
        r.raise_for_status()
        d = r.json()
        token = str(d.get("resy_token") or "")
        if not token:
            raise RuntimeError("no reservation token")
        return token, str(d.get("reservation_id") or "")

    async def cancel(self, resy_token: str) -> None:
        r = await self._client.post(f"{API}/3/cancel", data={"resy_token": resy_token})
        r.raise_for_status()

    # -- bookings we made (so they can be cancelled), 0600 --
    def bookings(self) -> list[Booking]:
        if self.bookings_path is None:
            return []
        try:
            return [Booking(**b) for b in json.loads(self.bookings_path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            return []

    def save_bookings(self, items: list[Booking]) -> None:
        if self.bookings_path is None:
            return
        self.bookings_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".resy-", suffix=".json", dir=self.bookings_path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in items], f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.bookings_path)
        log_event("resy.bookings_saved", count=len(items))


def _money(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def pick_slot(slots: list[Slot], minutes: int, window: int = 45) -> tuple[Slot | None, list[Slot]]:
    """(the slot at or nearest the asked time within `window` minutes, a few other options)."""
    if not slots:
        return None, []
    ranked = sorted(slots, key=lambda s: (abs(s.minutes - minutes), s.minutes))
    best = ranked[0] if abs(ranked[0].minutes - minutes) <= window else None
    others = sorted([s for s in ranked[1:6] if s is not best], key=lambda s: s.minutes)[:3] if best else sorted(ranked[:3], key=lambda s: s.minutes)
    return best, others
