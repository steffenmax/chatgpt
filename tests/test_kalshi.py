import httpx

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.kalshi import Kalshi, format_odds, wants_kalshi
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message


def mk(label, price, vol):
    return {"yes_sub_title": label, "last_price_dollars": price, "yes_bid_dollars": None, "yes_ask_dollars": None, "volume_fp": vol}


EVENTS = [
    {"event_ticker": "NBA", "title": "2027 Pro Basketball Champion", "sub_title": "2026-27", "category": "Sports",
     "markets": [mk("Oklahoma City", "0.23", 900000), mk("New York", "0.08", 500000), mk("Brooklyn", "0.003", 1000)]},
    {"event_ticker": "EBB", "title": "eBasketball: Lakers (Ray) vs New York Knicks (Melo)", "sub_title": "Oct 21", "category": "Sports",
     "markets": [mk("New York Knicks (Melo)", "0.01", 50), mk("Lakers (Ray)", "0.99", 50)]},
    {"event_ticker": "PO", "title": "Pro Football Playoff Qualifiers", "sub_title": "2026-27", "category": "Sports",
     "markets": [mk("New York G", "0.32", 300000), mk("New York J", "0.07", 300000), mk("Buffalo", "0.94", 300000)]},
    {"event_ticker": "BOTH", "title": "New York J and New York G: Both Make the Playoffs", "sub_title": "NYJNYG", "category": "Sports",
     "markets": [mk("Both", "0.02", 100)]},
]


def api(seen):
    def handler(req: httpx.Request):
        seen.append((req.method, req.url.path))
        if req.url.path.endswith("/events"):
            return httpx.Response(200, json={"events": EVENTS, "cursor": ""})
        ticker = req.url.path.rsplit("/", 1)[-1]
        ev = next(e for e in EVENTS if e["event_ticker"] == ticker)
        return httpx.Response(200, json={"event": ev, "markets": ev["markets"]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_search_ranks_focused_busy_markets_and_reads_only():
    seen = []
    k = Kalshi(client=api(seen))
    ev, outs, named = await k.search("knicks title odds")
    assert ev.ticker == "NBA" and named == ["New York"]  # the real market, not the esports game; Knicks, not Brooklyn
    assert format_odds(ev, outs, named).startswith("Kalshi: 2027 Pro Basketball Champion (2026-27)\n- New York: 8%\n- Oklahoma City: 23%")
    ev, outs, named = await k.search("will the jets make the playoffs")
    assert ev.ticker == "PO" and named == ["New York J"]
    assert (await k.search("ebasketball knicks melo"))[0].ticker == "EBB"  # esports only when asked
    assert await k.search("ricky kalshi") is None
    assert all(m == "GET" for m, _ in seen) and sum(p.endswith("/events") for _, p in seen) == 1  # one index build


def test_formatting_and_triggers():
    from imessage_agent.kalshi import EventDoc, Outcome

    ev = EventDoc("X", "Fed cut?", "", "", "")
    assert format_odds(ev, [Outcome("Yes", 0.004, 1)], []) == "Kalshi: Fed cut?: <1% yes"
    assert wants_kalshi("ricky what's kalshi say about the knicks") and not wants_kalshi("ricky odds on the knicks")


async def test_kalshi_in_the_chat(clock):
    bb, llm = FakeBB(), FakeLLM("knicks at 8, max you're dreaming")
    agent = Agent(make_settings(), bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {}, use_contacts=False), kalshi=Kalshi(client=api([])))
    assert await agent.process(message("coco what's kalshi say about the knicks title", guid="k1")) == "handled:kalshi"
    assert bb.texts[-1][1].startswith("Kalshi: 2027 Pro Basketball Champion (2026-27)\n- New York: 8%")
    assert bb.texts[-1][1].endswith("\nknicks at 8, max you're dreaming")
    await agent.process(message("/kalshi lunar colony by tuesday", guid="k2"))
    assert bb.texts[-1][1] == "kalshi doesn't have a market on that"
