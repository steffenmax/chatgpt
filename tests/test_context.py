from imessage_agent.context import ContextStore
from tests.conftest import FakeClock


def test_rolling_window():
    c = ContextStore(max_messages=3, ttl_seconds=100, clock=FakeClock())
    for i in range(5):
        c.add("chat", "a", f"m{i}")
    assert [m.text for m in c.get("chat")] == ["m2", "m3", "m4"]


def test_ttl_wipe_on_read():
    clk = FakeClock()
    c = ContextStore(max_messages=30, ttl_seconds=3600, clock=clk)
    c.add("chat", "a", "hello")
    clk.advance(3599)
    assert len(c.get("chat")) == 1
    clk.advance(2)  # now 3601 s idle
    assert c.get("chat") == []
    assert c.chat_count() == 0


def test_ttl_sweep():
    clk = FakeClock()
    c = ContextStore(max_messages=30, ttl_seconds=10, clock=clk)
    c.add("a", "x", "1")
    clk.advance(5)
    c.add("b", "x", "2")
    clk.advance(6)  # a idle 11s, b idle 6s
    assert c.sweep() == 1
    assert c.get("a") == [] and len(c.get("b")) == 1


def test_activity_refreshes_ttl():
    clk = FakeClock()
    c = ContextStore(max_messages=30, ttl_seconds=10, clock=clk)
    c.add("a", "x", "1")
    clk.advance(8)
    c.add("a", "x", "2")
    clk.advance(8)
    assert len(c.get("a")) == 2


def test_forget():
    c = ContextStore(max_messages=30, ttl_seconds=10, clock=FakeClock())
    c.add("a", "x", "1")
    assert c.forget("a") is True
    assert c.get("a") == []
    assert c.forget("a") is False
