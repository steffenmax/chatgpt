import asyncio
import json
from datetime import date, datetime, timedelta

import httpx

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.resy import Resy, Slot, is_yes, parse_request, pick_slot
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

DAY = (datetime.now().date() + timedelta(days=(5 - datetime.now().weekday()) % 7 or 7))  # the coming Saturday


def fake_resy(calls, fee=50.0, times=("17:45", "18:00", "19:30"), book_ok=True):
    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path))
        if req.url.path == "/3/venuesearch/search":
            return httpx.Response(200, json={"search": {"hits": [{"id": {"resy": 6194}, "name": "Carbone", "url_slug": "carbone",
                                                                   "location": {"code": "ny", "name": "New York"}}]}})
        if req.url.path == "/4/find":
            slots = [{"date": {"start": f"{DAY.isoformat()} {t}:00"}, "config": {"token": f"tok-{t}", "type": "Dining Room"},
                      "payment": {"cancellation_fee": fee, "deposit_fee": None, "secs_cancel_cut_off": 86400}} for t in times]
            return httpx.Response(200, json={"results": {"venues": [{"slots": slots}]}})
        if req.url.path == "/3/details":
            return httpx.Response(200, json={"book_token": {"value": "BT"}})
        if req.url.path == "/2/user":
            return httpx.Response(200, json={"payment_methods": [{"id": 77, "is_default": True}]})
        if req.url.path == "/3/book":
            assert b"struct_payment_method" in req.content and b"BT" in req.content
            return httpx.Response(200 if book_ok else 412, json={"resy_token": "RT1", "reservation_id": 555} if book_ok else {})
        if req.url.path == "/3/cancel":
            return httpx.Response(200, json={})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


NAMES = {"+15550001111": "Max", "+15550002222": "Tom", "+15550003333": "Joe", "+15550004444": "Dziuba"}


def _agent(clock, tmp_path, token="TOKEN", **kw):
    calls = []
    bb = FakeBB()
    agent = Agent(make_settings(), bb, FakeLLM(), ContextStore(15, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, NAMES, use_contacts=False),
                  resy=Resy(token, bookings_path=str(tmp_path / "b.json"), client=fake_resy(calls, **kw)))
    return agent, bb, calls


def test_parsing_and_yes():
    today = date(2026, 10, 21)
    r = parse_request("coco get resy for carbone this saturday at 6 3 people", today, "coco")
    assert (r.venue, r.day, r.minutes, r.party) == ("carbone", date(2026, 10, 24), 18 * 60, 3)
    assert parse_request("coco get resy for carbone saturday", today, "coco") == "how many people?"
    assert all(is_yes(t) for t in ("yes", "Yeah!", "im in", "down", "bet", "👍"))
    assert not any(is_yes(t) for t in ("yes but not saturday", "no", "maybe", "who's paying"))
    s = Slot("t", datetime(2026, 10, 24, 18), "x", 50.0, None, 24)
    assert s.fee_line(3, "Max") == "⚠️ heads up: there's a $50 per person fee ($150 total) if it's cancelled within 24h or nobody shows, charged to Max's card."
    best, others = pick_slot([Slot("a", datetime(2026, 10, 24, 17, 45), "", None, None, None),
                              Slot("b", datetime(2026, 10, 24, 21), "", None, None, None)], 18 * 60)
    assert best.token == "a" and [o.token for o in others] == ["b"]


async def test_three_yeses_book_it_and_the_fee_comes_first(clock, tmp_path):
    agent, bb, calls = _agent(clock, tmp_path)
    assert await agent.process(message("coco get resy for carbone this saturday at 6 3 people", guid="r1")) == "handled:resy"
    msg = bb.texts[-1][1]
    assert msg.startswith(f"Carbone, {DAY.strftime('%a %b %-d')} 6:00pm for 3 is open (dining room).")
    assert msg.index("heads up: there's a $50 per person fee") < msg.index("need 3 yes's in the next 15 min to lock it in (0/3)")
    await agent.process(message("yes", guid="y1", sender="+15550002222"))
    await agent.process(message("yes", guid="y1b", sender="+15550002222"))  # same person twice counts once
    assert bb.texts[-1][1] == "1/3" and ("POST", "/3/book") not in calls
    await agent.process(message("down", guid="y2", sender="+15550003333"))
    await agent.process(message("bet", guid="y3", sender="+15550004444"))
    assert ("POST", "/3/book") in calls
    assert bb.texts[-1][1].startswith(f"3/3, locked in 🔒 Carbone {DAY.strftime('%a %b %-d')} 6:00pm for 3, under Max's name.")
    assert json.loads((tmp_path / "b.json").read_text())[0]["resy_token"] == "RT1"
    assert await agent.process(message("coco cancel the carbone res", guid="c1")) == "handled:resy_cancel"
    assert bb.texts[-1][1].startswith("cancelled Carbone") and ("POST", "/3/cancel") in calls
    assert json.loads((tmp_path / "b.json").read_text()) == []


async def test_no_token_closest_time_and_failed_booking(clock, tmp_path):
    agent, bb, calls = _agent(clock, tmp_path, token="")
    await agent.process(message("coco get resy for carbone saturday at 6 for 2", guid="n1"))
    assert "i can't book without Max's resy login, grab it here: https://resy.com/cities/ny/carbone?date=" in bb.texts[-1][1]
    agent2, bb2, _ = _agent(clock, tmp_path / "x", times=("21:30", "22:00"))
    await agent2.process(message("coco get resy for carbone saturday at 6 for 2", guid="n2"))
    assert "nothing near 6:00pm" in bb2.texts[-1][1] and "9:30pm, 10:00pm" in bb2.texts[-1][1]
    agent3, bb3, _ = _agent(clock, tmp_path / "y", book_ok=False, fee=0)
    await agent3.process(message("coco get resy for carbone saturday at 6 for 1", guid="n3"))
    assert "heads up" not in bb3.texts[-1][1]  # no fee, no notice
    await agent3.process(message("yes", guid="n4"))
    assert "resy wouldn't take it" in bb3.texts[-1][1] and "resy.com/cities/ny/carbone" in bb3.texts[-1][1]


async def test_hold_expires(clock, tmp_path, monkeypatch):
    import imessage_agent.handler as h

    agent, bb, calls = _agent(clock, tmp_path)
    real_sleep = asyncio.sleep

    async def fast(sec):
        await real_sleep(0)

    monkeypatch.setattr(h.asyncio, "sleep", fast)
    await agent.process(message("coco get resy for carbone saturday at 6 for 4", guid="e1"))
    await agent.drain()
    assert bb.texts[-1][1] == "time's up, only 0/4 yes's. let the Carbone table go" and not agent._resy_pending
