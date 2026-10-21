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

RESY_ASK = re.compile(r"\b(resy|reservations?|rezzy|a res|res for|res at|book (?:a |us a )?(?:table|spot)|table for \w+|"
                      r"get (?:us )?a table|find (?:us )?a table|next available|(?:dinner|lunch|brunch) for (?:\d+|two|three|"
                      r"four|five|six|seven|eight))\b", re.IGNORECASE)
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


@dataclass
class Parts:
    """A reservation request as far as it's known; filled in over a few messages."""

    venue: str = ""
    area: str = ""
    day: date | None = None
    minutes: int | None = None
    party: int | None = None
    wants_bar: bool = False  # "to watch the game"
    game: str = ""  # the "watch the pats game" bit, to look up kickoff
    guessed_time: str = ""  # "night" -> 7pm, said back so they can correct it
    cuisine: str = ""  # "sushi", "steakhouse", "sports bar"
    next_available: bool = False  # "next available", "whenever"

    def merge(self, newer: "Parts") -> "Parts":
        for f in ("venue", "area", "day", "minutes", "party", "game", "guessed_time", "cuisine"):
            v = getattr(newer, f)
            if v not in (None, ""):
                setattr(self, f, v)
        if newer.minutes is not None and not newer.guessed_time:
            self.guessed_time = ""
        self.wants_bar = self.wants_bar or newer.wants_bar
        self.next_available = self.next_available or newer.next_available
        return self

    def missing(self) -> str | None:
        """Only what can't be searched without: something to look for, and how many people.
        No day and no time means the next available; a day without a time means show that day's times."""
        if not (self.venue or self.area or self.cuisine):
            return "where or what? a spot (\"carbone\"), a kind of food (\"sushi\"), an area (\"in hoboken\"), or a mix"
        if self.party is None:
            return "how many people?"
        if not 1 <= self.party <= 20:
            return "party size has to be 1 to 20"
        return None

    @property
    def mode(self) -> str:
        if self.next_available or (self.day is None and self.minutes is None):
            return "next"
        return "exact" if self.minutes is not None else "day"

    def request(self) -> "Request":
        return Request(self.venue or self.area, self.day, self.minutes, self.party)  # type: ignore[arg-type]


_MEALS = {"brunch": 11 * 60 + 30, "lunch": 12 * 60 + 30, "dinner": 19 * 60, "night": 19 * 60, "tonight": 19 * 60,
          "evening": 19 * 60}


def parse_parts(text: str, today: date, bot_name: str = "", followup_for: str | None = None) -> Parts:
    """Whatever the message says about a reservation. followup_for: the field the bot just asked about,
    so a bare "4" answers "how many people?" and a bare "8" answers "what time?"."""
    raw = " ".join((text or "").replace("’", "'").split())
    if followup_for and re.fullmatch(r"\s*(\d{1,2}|" + "|".join(_NUMBERS) + r")\s*[.!]*\s*", raw, re.IGNORECASE):
        n = int(_NUMBERS.get(raw.strip(" .!").lower(), raw.strip(" .!") if raw.strip(" .!").isdigit() else 0))
        if followup_for == "party":
            return Parts(party=n)
        if followup_for == "time" and 1 <= n <= 12:
            return Parts(minutes=(n + 12 if n < 12 else n) * 60)
    parts = Parts()
    t = " " + raw + " "
    if bot_name:
        t = re.sub(rf"\b{re.escape(bot_name)}\b,?", " ", t, flags=re.IGNORECASE)
    # "to watch the pats game": a bar, and maybe the kickoff time
    m = re.search(r"\b(?:to\s+)?(?:watch|catch|for)\s+(?:the\s+)?([a-z0-9' ]{0,40}?\b(?:game|match|fight|kickoff|playoffs))\b",
                  t, re.IGNORECASE)
    if m:
        parts.wants_bar, parts.game = True, m.group(1).strip()
        t = t[:m.start()] + " " + t[m.end():]
    # an area: "in hoboken", "near the west village"
    stops = r"(?:on|this|next|tonight|today|tomorrow|tmrw|at|for|to|around|with|" + "|".join(_DAYS) + "|" + "|".join(_DAY_ABBR) + r")"
    m = re.search(rf"\b(?:in|near|around|by)\s+(?!\d)([a-z][a-z .'-]*?)(?=\s+{stops}\b|\s*$|\s*[,.!?])", t, re.IGNORECASE)
    if m and not re.fullmatch(r"(the )?(city|town|area|morning|evening|afternoon|night)", m.group(1).strip(), re.IGNORECASE):
        parts.area = m.group(1).strip(" .'-")
        t = t[:m.start()] + " " + t[m.end():]
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
    parts.party, parts.minutes, parts.day = party, minutes, day
    if parts.minutes is None:
        meal = next((w for w in _MEALS if re.search(rf"\b{w}\b", raw, re.IGNORECASE)), None)
        if meal:
            parts.minutes, parts.guessed_time = _MEALS[meal], meal
    # whatever's left is the restaurant
    t = re.sub(r"\b(can you|could you|pls|please|yo|get|grab|find|book|make|snag|lock in|me|us|a|an|the|resy|rezzy|"
               r"reservations?|res|table|for|at|on|in|tonight|around|dinner|lunch|brunch|spot|night|evening|place|"
               r"somewhere|something|good|nice|some|bar|restaurant)\b", " ", t, flags=re.IGNORECASE)
    parts.venue = " ".join(re.sub(r"[^\w&' -]", " ", t).split()).strip(" -'")
    return parts


def parse_request(text: str, today: date, bot_name: str = "") -> Request | str:
    """One-shot form: a full Request, or the question to ask next."""
    parts = parse_parts(text, today, bot_name)
    missing = parts.missing()
    return missing if missing else parts.request()


@dataclass
class Venue:
    id: int
    name: str
    city: str
    slug: str
    city_slug: str
    cuisine: str = ""
    neighborhood: str = ""

    @property
    def blurb(self) -> str:
        bits = [b for b in (self.cuisine.lower(), self.neighborhood) if b]
        return f" ({', '.join(bits)})" if bits else ""

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

    @staticmethod
    def _venue(h: dict[str, Any]) -> Venue:
        loc = h.get("location") or {}
        cuisine = h.get("cuisine")
        cuisine = (cuisine[0] if cuisine else "") if isinstance(cuisine, list) else str(cuisine or "")
        return Venue(int((h.get("id") or {}).get("resy")), str(h.get("name") or ""), str(loc.get("name") or h.get("locality") or ""),
                     str(h.get("url_slug") or ""), str(loc.get("code") or "ny"), cuisine, str(h.get("neighborhood") or "").split(",")[0].strip())

    async def search(self, query: str, per_page: int = 20) -> list[tuple[Venue, dict[str, Any]]]:
        r = await self._client.post(f"{API}/3/venuesearch/search", json={
            "query": query, "geo": {"latitude": self.lat, "longitude": self.lon}, "per_page": per_page, "types": ["venue"]})
        r.raise_for_status()
        return [(self._venue(h), h) for h in (r.json().get("search") or {}).get("hits") or [] if (h.get("id") or {}).get("resy")]

    async def candidates(self, cuisine: str, area: str, wants_bar: bool = False, limit: int = 10) -> list[Venue]:
        """Places matching a food type and/or an area (both if both given), best fits first."""
        queries = [q for q in (f"{cuisine} {area}".strip(), area, cuisine) if q]
        seen: dict[int, tuple[Venue, dict]] = {}
        for q in dict.fromkeys(queries):
            for v, h in await self.search(q, 30):
                seen.setdefault(v.id, (v, h))
        cw = [w for w in re.findall(r"[a-z]+", cuisine.lower()) if len(w) > 2]
        aw = [w for w in re.findall(r"[a-z]+", area.lower()) if len(w) > 2]
        out = []
        for v, h in seen.values():
            where = " ".join(str(x or "") for x in (h.get("locality"), h.get("neighborhood"), h.get("region"),
                                                  (h.get("location") or {}).get("name"), v.name)).lower()
            what = f"{v.name} {v.cuisine}".lower()
            if aw and not all(w in where for w in aw):
                continue
            if cw and not any(w in what or (w.rstrip("s") in what) for w in cw):
                continue
            rank = 0
            if wants_bar:
                rank = (0 if re.search(r"sports|pub|tap\b|taproom|tavern|beer|brew|saloon", what) else
                        1 if re.search(r"grill|\bbar\b", what) else 2 if "lounge" in what else 3)
            out.append((rank, v))
        out.sort(key=lambda x: x[0])
        return [v for _, v in out[:limit]]

    async def calendar(self, venue: Venue, party: int, start: date, days: int = 30) -> list[date]:
        """Days in the next `days` with reservations open (Resy's availability calendar)."""
        r = await self._client.get(f"{API}/4/venue/calendar", params={
            "venue_id": venue.id, "num_seats": party, "start_date": start.isoformat(),
            "end_date": (start + timedelta(days=days)).isoformat()})
        r.raise_for_status()
        out = []
        for d in r.json().get("scheduled") or []:
            if ((d.get("inventory") or {}).get("reservation")) == "available":
                try:
                    out.append(date.fromisoformat(d["date"]))
                except (KeyError, ValueError):
                    pass
        return out

    async def find_venue(self, query: str) -> Venue | None:
        r = await self._client.post(f"{API}/3/venuesearch/search", json={
            "query": query, "geo": {"latitude": self.lat, "longitude": self.lon}, "per_page": 5, "types": ["venue"]})
        r.raise_for_status()
        hits = (r.json().get("search") or {}).get("hits") or []
        if not hits:
            return None
        return self._venue(hits[0])

    async def area_options(self, area: str, day: date, party: int, minutes: int, wants_bar: bool = False,
                           limit: int = 3) -> list[tuple[Venue, Slot]]:
        """Up to `limit` places in an area with a table near the time; bars first when they want to watch a game."""
        r = await self._client.post(f"{API}/3/venuesearch/search", json={
            "query": area, "geo": {"latitude": self.lat, "longitude": self.lon}, "per_page": 20, "types": ["venue"]})
        r.raise_for_status()
        words = [w for w in re.findall(r"[a-z]+", area.lower()) if len(w) > 2]
        hits = []
        for h in (r.json().get("search") or {}).get("hits") or []:
            where = " ".join(str(h.get(k) or "") for k in ("locality", "neighborhood", "region")).lower()
            where += " " + str((h.get("location") or {}).get("name") or "").lower()
            if words and not all(w in where for w in words):
                continue
            cuisine = " ".join(h.get("cuisine") or []) if isinstance(h.get("cuisine"), list) else str(h.get("cuisine") or "")
            text = f"{h.get('name')} {cuisine}"
            # for a game: sports bars and pubs first, then grills/bars, then lounges, then everything else
            rank = (0 if re.search(r"sports|pub|tap\b|taproom|tavern|beer|brew|saloon", text, re.IGNORECASE) else
                    1 if re.search(r"grill|\bbar\b", text, re.IGNORECASE) else
                    2 if re.search(r"lounge", text, re.IGNORECASE) else 3) if wants_bar else 0
            loc = h.get("location") or {}
            hits.append((rank, Venue(int((h.get("id") or {}).get("resy")), str(h.get("name")),
                                                               str(loc.get("name") or h.get("locality") or ""),
                                                               str(h.get("url_slug") or ""), str(loc.get("code") or "ny"))))
        hits.sort(key=lambda x: x[0])
        out: list[tuple[Venue, Slot]] = []
        for _, venue in hits[:10]:
            try:
                best, _ = pick_slot(await self.slots(venue, day, party), minutes, window=60)
            except httpx.HTTPError:
                continue
            if best is not None:
                out.append((venue, best))
            if len(out) >= limit:
                break
        return out

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


# --- understanding the request: food, restaurant, area, day, time, people -------------------------
SPEC_SYSTEM = (
    "Read one message asking a group-chat bot to find or book a restaurant reservation. Pull out what it says, "
    "exactly as written, and answer ONLY with JSON matching the schema.\n"
    "restaurant: a specific place's name if one is named (\"carbone\", \"4 charles prime rib\"), else \"\".\n"
    "cuisine: a kind of food or place if asked for (\"sushi\", \"italian\", \"steakhouse\", \"sports bar\", \"rooftop\"), "
    "else \"\".\n"
    "area: a neighborhood, town or city if given (\"west village\", \"hoboken\", \"midtown\"), else \"\".\n"
    "day: the day words as written (\"friday\", \"this saturday\", \"tomorrow\", \"10/31\", \"tonight\"), else \"\".\n"
    "time: the time words as written (\"8pm\", \"7:30\", \"at 6\", \"dinner\", \"night\"), else \"\".\n"
    "party: number of people as a number, 0 if not said.\n"
    "next_available: true if they want the soonest open table (\"next available\", \"whenever\", \"soonest\", "
    "\"any day\"), else false.\n"
    "game: if they want to watch a game, the game words (\"pats game\"), else \"\"."
)
SPEC_SCHEMA = {"type": "object", "properties": {
    "restaurant": {"type": "string"}, "cuisine": {"type": "string"}, "area": {"type": "string"}, "day": {"type": "string"},
    "time": {"type": "string"}, "party": {"type": "integer"}, "next_available": {"type": "boolean"}, "game": {"type": "string"}},
    "required": ["restaurant", "cuisine", "area", "day", "time", "party", "next_available", "game"]}


def spec_from_model(raw: str, text: str, today: date, bot_name: str = "") -> Parts:
    """The model's reading of the request, with day/time/party checked by the plain parser."""
    base = parse_parts(text, today, bot_name)
    try:
        d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, TypeError):
        return base
    clean = lambda k: " ".join(str(d.get(k) or "").split())[:60]
    restaurant, cuisine, area = clean("restaurant"), clean("cuisine"), clean("area")
    if restaurant.lower() in (cuisine.lower(), area.lower()):
        restaurant = ""
    if not (restaurant or cuisine or area):  # the model missed it: trust the plain parser
        return base
    when = parse_parts(f"{clean('day')} {clean('time')}", today) if (clean("day") or clean("time")) else Parts()
    party = d.get("party") if isinstance(d.get("party"), int) and 0 < d.get("party") <= 20 else base.party
    game = clean("game") or base.game
    return Parts(venue=restaurant, area=area, day=when.day or base.day,
                 minutes=when.minutes if when.minutes is not None else base.minutes, party=party,
                 wants_bar=bool(game) or "bar" in cuisine.lower() or base.wants_bar, game=game,
                 guessed_time=when.guessed_time or base.guessed_time, cuisine=cuisine,
                 next_available=bool(d.get("next_available")))


async def find_options(resy: "Resy", p: Parts, now: datetime, limit: int = 4) -> list[tuple[Venue, Slot]]:
    """Open (venue, slot) choices for a request, in the order to offer them.
    exact: the slot nearest the time at each place; day: that day's times (one place) or the best time per
    place; next: the soonest open day per place (Resy's calendar), then its times."""
    today, mode = now.date(), p.mode
    target = p.minutes if p.minutes is not None else 19 * 60
    day = p.day
    if mode == "exact" and day is None:
        day = today if target > now.hour * 60 + now.minute + 30 else today + timedelta(days=1)
    if p.venue:
        v = await resy.find_venue(f"{p.venue} {p.area}".strip())
        venues = [v] if v else []
    else:
        venues = await resy.candidates(p.cuisine, p.area, p.wants_bar)
    single = bool(p.venue)
    out: list[tuple[Venue, Slot]] = []
    for v in venues[:8]:
        try:
            if mode == "next":
                days = [d for d in await resy.calendar(v, p.party or 2, today) if d >= today][:2]
                slots = []
                for d in days:
                    slots = [s for s in await resy.slots(v, d, p.party or 2) if s.start > now]
                    if slots:
                        break
            else:
                slots = [s for s in await resy.slots(v, day, p.party or 2) if s.start > now]  # type: ignore[arg-type]
        except httpx.HTTPError:
            continue
        if not slots:
            continue
        if single and (mode != "exact" or pick_slot(slots, target, window=60)[0] is None):
            # one place, flexible: several of its times, nearest the target first, shown in time order
            near = sorted(sorted(slots, key=lambda s: abs(s.minutes - target))[:limit + 1], key=lambda s: s.start)
            return [(v, s) for s in near]
        best, _ = pick_slot(slots, target, window=60 if mode == "exact" else 24 * 60)
        if best is not None:
            out.append((v, best))
        if len(out) >= limit:
            break
    if mode == "next":
        out.sort(key=lambda vs: vs[1].start)
    return out
