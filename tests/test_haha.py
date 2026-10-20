import asyncio

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from tests.conftest import ALLOWED, OTHER, FakeBB, make_settings, message


class SysLLM:
    name = "fake"

    def __init__(self):
        self.systems = []

    async def reply(self, system, messages, stop=None):
        self.systems.append(system)
        return "ok"

    async def healthy(self):
        return True

    async def aclose(self):
        pass


def laugh(sender, target="T1", kind="laugh", chat=ALLOWED, guid=None):
    return message("Laughed at “funny line”", chat=chat, sender=sender, guid=guid or f"r-{sender}-{target}-{kind}",
                   associatedMessageType=kind, associatedMessageGuid=f"p:0/{target}")


def _agent(clock, **over):
    s = make_settings(haha_ack_at=6, haha_praise_at=7, **over)
    bb, llm = FakeBB(), SysLLM()
    bb.known_messages = {"T1": {"guid": "T1", "text": "funny line", "isFromMe": False, "handle": {"address": "+15550009999"}},
                         "B1": {"guid": "B1", "text": "bot line", "isFromMe": True}}
    agent = Agent(s, bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {"+15550009999": "Dan"}, use_contacts=False))
    return agent, bb, llm


async def test_six_laughs_get_a_dumb_line_and_seven_get_praise(clock):
    agent, bb, llm = _agent(clock)
    for i in range(5):
        assert await agent.process(laugh(f"+1555000000{i}")) == "tapback:counted"
    await agent.process(laugh("+15550000001", guid="dup"))  # same person twice counts once
    assert await agent.process(laugh("+15550000005")) == "tapback:ack"
    await agent.drain()
    assert len(bb.texts) == 1 and 'Dan\'s message "funny line" just got 6 haha reactions' in llm.systems[-1]
    assert await agent.process(laugh("+15550000006")) == "tapback:praise"
    await agent.drain()
    assert len(bb.texts) == 2 and "got a haha from every single person" in llm.systems[-1] and "no roast" in llm.systems[-1]
    assert await agent.process(laugh("+15550000007")) == "tapback:counted"  # each milestone once
    await agent.drain()
    assert len(bb.texts) == 2


async def test_removed_laughs_self_laughs_bot_lines_other_chats_and_off(clock):
    agent, bb, llm = _agent(clock)
    for i in range(6):
        await agent.process(laugh(f"+1555000000{i}"))
    await agent.drain()
    assert len(bb.texts) == 1
    agent, bb, llm = _agent(clock)
    for i in range(5):
        await agent.process(laugh(f"+1555000000{i}"))
    assert await agent.process(laugh("+15550000004", kind="-laugh")) == "tapback:removed"
    assert await agent.process(laugh("+15550000005")) == "tapback:counted"  # back to 5
    # the author laughing at their own message doesn't count
    assert await agent.process(laugh("+15550009999")) == "tapback:ack"
    await agent.drain()
    assert bb.texts == []
    # the bot's own line: no victory lap
    for i in range(6):
        await agent.process(laugh(f"+1555000000{i}", target="B1"))
    await agent.drain()
    assert bb.texts == []
    # other chats and other tapbacks are ignored
    assert await agent.process(laugh("+15550000001", chat=OTHER, target="X")) == "ignored:reaction_or_system"
    assert await agent.process(laugh("+15550000001", kind="love", target="X")) == "ignored:reaction_or_system"
    off = Agent(make_settings(), FakeBB(), SysLLM(), ContextStore(5, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                resolver=SenderResolver(None, {}, use_contacts=False))
    assert await off.process(laugh("+15550000001")) == "ignored:reaction_or_system"


async def test_muted_chat_stays_quiet(clock):
    agent, bb, llm = _agent(clock)
    await agent.process(message("/mute 10m", guid="m1"))
    for i in range(7):
        await agent.process(laugh(f"+1555000000{i}"))
    await agent.drain()
    assert len(bb.texts) == 1  # just the mute ack
