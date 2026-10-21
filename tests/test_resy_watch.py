import asyncio
from datetime import date, datetime, timedelta

from imessage_agent import resy_watch
from imessage_agent.resy import Slot, Venue
from imessage_agent.resy_watch import (ANY_PARTY, DropWatch, DropWatcher, alert_text, parse_watch, wants_watch,
                                       wants_watch_list, wants_watch_stop)

TODAY = date(2026, 10, 21)  # a Wednesday


def test_parse_daily_any_size():
    s = parse_watch("ricky watch 4 charles drop at 9am", TODAY, "ricky")
    assert (s.venue.lower(), s.drop, s.on, s.parties) == ("4 charles", 540, None, ANY_PARTY)


def test_parse_one_day_and_size():
    s = parse_watch("ricky set a watch for the carbone drop 10am tomorrow for 4", TODAY, "ricky")
    assert (s.venue.lower(), s.drop, s.on, s.parties) == ("carbone", 600, TODAY + timedelta(days=1), [4])
    s = parse_watch("ricky watch 4 charles release at 9:00 on nov 1", TODAY, "ricky")
    assert (s.venue.lower(), s.drop, s.on) == ("4 charles", 540, date(2026, 11, 1))
    assert parse_watch("ricky watch 4 charles drop every day at 9am", TODAY, "ricky").on is None
    assert parse_watch("ricky watch 4 charles drop", TODAY, "ricky").drop is None


def test_asks():
    assert wants_watch("ricky watch 4 charles drop at 9am")
    assert wants_watch("ricky set a watch on 4 charles 9am")
    assert not wants_watch("ricky what time is the game")
    assert wants_watch_list("ricky what are you watching", "ricky")
    assert wants_watch_list("ricky watches", "ricky")
    assert wants_watch_stop("ricky stop watching 4 charles")
    assert not wants_watch_stop("ricky book carbone")


def _w(**kw):
    base = dict(id=1, chat="c", venue_id=834, venue_name="4 Charles Prime Rib", slug="4-charles-prime-rib",
                city_slug="ny", drop=540)
    return DropWatch(**(base | kw))


def test_next_drop():
    w = _w()
    assert w.next_drop(datetime(2026, 10, 21, 8, 0)) == datetime(2026, 10, 21, 9, 0)
    assert w.next_drop(datetime(2026, 10, 21, 9, 5)) == datetime(2026, 10, 21, 9, 0)  # still in the window
    assert w.next_drop(datetime(2026, 10, 21, 10, 0)) == datetime(2026, 10, 22, 9, 0)
    assert _w(last_run="2026-10-21").next_drop(datetime(2026, 10, 21, 9, 5)) == datetime(2026, 10, 22, 9, 0)
    assert _w(on="2026-11-01").next_drop(datetime(2026, 10, 21, 8, 0)) == datetime(2026, 11, 1, 9, 0)


def _slot(day, hh, mm=0):
    return Slot(f"t{day}{hh}{mm}", datetime.combine(day, datetime.min.time()).replace(hour=hh, minute=mm), "Dining Room",
                None, None, None)


def test_alert_text():
    d = date(2026, 11, 11)
    text = alert_text(_w(), [(d, 2, _slot(d, 17)), (d, 2, _slot(d, 22, 30)), (d, 4, _slot(d, 22, 45))])
    assert "4 Charles Prime Rib just opened up" in text
    assert "Wed Nov 11: for 2: 5:00pm, 10:30pm · for 4: 10:45pm" in text
    assert "resy.com/cities/ny/4-charles-prime-rib?date=2026-11-11&seats=2" in text


class FakeResy:
    def __init__(self, clock):
        self.clock = clock

    async def calendar(self, venue, party, start, days=30):
        # tables show up 30 seconds after the 9am drop, for 2 and 4 only
        return [date(2026, 11, 11)] if self.clock[0] >= datetime(2026, 10, 21, 9, 0, 30) and party in (2, 4) else []

    async def slots(self, venue, day, party):
        return [_slot(day, 17 if party == 2 else 22)]


def test_drop_run(tmp_path, monkeypatch):
    clock = [datetime(2026, 10, 21, 8, 59, 0)]
    sent = []

    async def send(chat, text):
        sent.append((chat, text))

    async def sleep(seconds):
        clock[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(resy_watch, "DROP_WINDOW", 120.0)
    watcher = DropWatcher(FakeResy(clock), str(tmp_path / "w.json"), send, now=lambda: clock[0], sleep=sleep)
    watcher.add("c", Venue(834, "4 Charles Prime Rib", "", "4-charles-prime-rib", "ny"), 540, None, list(ANY_PARTY), "Max")
    w = watcher.watches()[0]
    asyncio.run(watcher._run_drop(w, datetime(2026, 10, 21, 9, 0)))
    assert len(sent) == 1  # posted once, not every check
    assert "for 2: 5:00pm · for 4: 10:00pm" in sent[0][1]
    assert watcher.watches()[0].last_run == "2026-10-21"  # a daily watch stays, done for today


def test_one_day_watch_reports_nothing_and_ends(tmp_path, monkeypatch):
    clock = [datetime(2026, 10, 21, 8, 59, 0)]
    sent = []

    async def send(chat, text):
        sent.append(text)

    async def sleep(seconds):
        clock[0] += timedelta(seconds=seconds)

    class Empty(FakeResy):
        async def calendar(self, *a, **k):
            return []

    monkeypatch.setattr(resy_watch, "DROP_WINDOW", 60.0)
    watcher = DropWatcher(Empty(clock), str(tmp_path / "w.json"), send, now=lambda: clock[0], sleep=sleep)
    watcher.add("c", Venue(834, "4 Charles Prime Rib", "", "4-charles-prime-rib", "ny"), 540, TODAY, [2], "Max")
    asyncio.run(watcher._run_drop(watcher.watches()[0], datetime(2026, 10, 21, 9, 0)))
    assert len(sent) == 1 and "nothing opened up" in sent[0]
    assert watcher.watches() == []


def test_remove(tmp_path):
    async def send(chat, text):
        pass
    watcher = DropWatcher(None, str(tmp_path / "w.json"), send)
    watcher.add("c", Venue(834, "4 Charles Prime Rib", "", "4-charles-prime-rib", "ny"), 540, None, [2], "Max")
    watcher.add("c", Venue(1, "Carbone", "", "carbone", "ny"), 600, None, [2], "Max")
    assert [w.venue_name for w in watcher.remove("c", "4 charles")] == ["4 Charles Prime Rib"]
    assert [w.venue_name for w in watcher.watches()] == ["Carbone"]
