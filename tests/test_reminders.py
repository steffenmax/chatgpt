import asyncio
import json
from datetime import datetime, timedelta

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.reminders import ReminderStore, describe, parse
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

NOW = datetime(2026, 10, 18, 14, 0).astimezone()  # a Sunday, 2pm


def test_parse_relative_and_clock_forms():
    due, what = parse("in 20m call the dentist", NOW)
    assert due == NOW + timedelta(minutes=20) and what == "call the dentist"
    due, what = parse("call the dentist in 2 hours", NOW)
    assert due == NOW + timedelta(hours=2) and what == "call the dentist"
    due, what = parse("at 7pm draft starts", NOW)
    assert due.hour == 19 and due.date() == NOW.date() and what == "draft starts"
    due, what = parse("draft starts at 7:30pm", NOW)
    assert (due.hour, due.minute) == (19, 30) and what == "draft starts"
    due, what = parse("at 9am standup", NOW)  # already passed today -> tomorrow
    assert due.hour == 9 and due.date() == NOW.date() + timedelta(days=1)
    due, what = parse("at 7 dinner", NOW)  # no am/pm in the afternoon -> 7pm today
    assert due.hour == 19 and due.date() == NOW.date()
    due, what = parse("tomorrow 9am venmo joe", NOW)
    assert due.hour == 9 and due.date() == NOW.date() + timedelta(days=1) and what == "venmo joe"
    due, what = parse("tomorrow pay rent", NOW)
    assert due.hour == 9 and due.date() == NOW.date() + timedelta(days=1) and what == "pay rent"
    due, what = parse("sat 3pm tee time", NOW)
    assert due.weekday() == 5 and due.hour == 15 and what == "tee time"
    due, what = parse("sun 1pm brunch", NOW)  # today is Sunday and 1pm passed -> next Sunday
    assert due.weekday() == 6 and (due - NOW).days == 6
    assert parse("", NOW) == (None, "")
    assert parse("just some words", NOW)[0] is None
    assert parse("at 25pm nope", NOW)[0] is None


def test_describe():
    assert describe(NOW + timedelta(minutes=20), NOW) == "in 20 min"
    assert describe(NOW + timedelta(hours=3), NOW) == "at 5:00pm"
    assert describe(NOW + timedelta(days=1), NOW).startswith("tomorrow at")
    assert describe(NOW + timedelta(days=4), NOW).startswith("Thu at")


def test_store_persists_and_pops_due(tmp_path):
    t = {"now": 1000.0}
    path = tmp_path / "r" / "reminders.json"
    st = ReminderStore(str(path), clock=lambda: t["now"])
    a = st.add("chatA", "Max", "call mom", due=1100.0)
    b = st.add("chatA", "Joe", "venmo", due=1050.0)
    st.add("chatB", "Sam", "other chat", due=1020.0)
    assert [r.id for r in st.pending("chatA")] == [b.id, a.id]
    assert (path.stat().st_mode & 0o777) == 0o600 and (path.parent.stat().st_mode & 0o777) == 0o700
    assert "call mom" in json.loads(path.read_text())["items"][0]["text"]
    # survives a restart
    st2 = ReminderStore(str(path), clock=lambda: t["now"])
    assert len(st2.pending("chatA")) == 2 and st2.next_due() == 1020.0
    assert st2.cancel("chatA", b.id) and not st2.cancel("chatA", 999)
    t["now"] = 1101.0
    fired = st2.pop_due()
    assert sorted(r.text for r in fired) == ["call mom", "other chat"]
    assert st2.pending("chatA") == [] and st2.next_due() is None


async def test_remind_command_round_trip(tmp_path):
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM()
    ctx = ContextStore(5, 3600)
    store = ReminderStore(str(tmp_path / "reminders.json"))
    agent = Agent(s, bb, llm, ctx, images=None,
                  resolver=SenderResolver(None, {"+15550001111": "Alex"}, use_contacts=False),  # type: ignore[arg-type]
                  reminders=store)
    assert await agent.process(message("/remind in 45m call the dentist", guid="r1")) == "handled:remind"
    assert bb.texts[-1][1].startswith("got it Alex, i'll remind you in 45 min (#1)")
    assert await agent.process(message("/reminders", guid="r2")) == "handled:reminders"
    assert "#1" in bb.texts[-1][1] and "call the dentist" in bb.texts[-1][1] and "(Alex)" in bb.texts[-1][1]
    assert await agent.process(message("/remind nonsense", guid="r3")) == "handled:remind"
    assert bb.texts[-1][1].startswith("say it like")
    assert await agent.process(message("/remind cancel 1", guid="r4")) == "handled:remind"
    assert bb.texts[-1][1] == "dropped it" and store.pending(ALLOWED) == []
    # firing posts in the chat with the asker's name
    r = store.add(ALLOWED, "Alex", "call the dentist", due=0)
    await agent.fire_reminder(r)
    assert bb.texts[-1] == (ALLOWED, "⏰ Alex: call the dentist")
    assert llm.calls == []
    # reminders are not stored as chat context
    assert all("dentist" not in m.text for m in ctx.get(ALLOWED) if not m.from_bot)


async def test_remind_without_store_says_off(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    await agent.process(message("/remind in 5m x", guid="r9"))
    assert bb.texts[-1][1] == "reminders are off on my end"
    await agent.process(message("/help", guid="h9"))
    assert "/remind" not in bb.texts[-1][1]


async def test_scheduler_fires_due_reminders(tmp_path):
    t = {"now": 1000.0}
    st = ReminderStore(str(tmp_path / "reminders.json"), clock=lambda: t["now"])
    st.add("c", "Max", "ping", due=1000.5)
    fired = []

    async def fire(r):
        fired.append(r.text)

    task = asyncio.create_task(st.run(fire, poll_seconds=0.05))
    await asyncio.sleep(0.02)
    assert fired == []
    t["now"] = 1001.0
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert fired == ["ping"]
