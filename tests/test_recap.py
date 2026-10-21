import time

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver, _norm_addr
from imessage_agent.recap import gather, wants_recap, window_from
from tests.conftest import ALLOWED, FakeBB, make_settings, message

NAMES = {"+15550001111": "Max", "+15550002222": "Joe", "+15550003333": "Dziuba"}


def msg(i, who, text, me=False, **extra):
    m = {"guid": f"m{i}", "text": text, "isFromMe": me, "dateCreated": 1_000_000 + i * 1000,
         "handle": None if me else {"address": who}}
    m.update(extra)
    return m


class HistoryBB(FakeBB):
    def __init__(self, history):
        super().__init__()
        self.history = history  # oldest first
        self.queries = []

    async def recent_messages(self, chat_guid, limit=200, after_ms=None):
        self.queries.append((chat_guid, limit, after_ms))
        return list(reversed(self.history))[:limit]


class SysLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply_text, self.calls = reply, []

    async def reply(self, system, messages, stop=None):
        self.calls.append((system, messages))
        return self.reply_text

    async def healthy(self):
        return True

    async def aclose(self):
        pass


def test_recap_phrases_and_windows():
    assert wants_recap("coco catch me up") and wants_recap("coco what'd i miss") and not wants_recap("coco what happened to the jets")
    assert window_from("catch me up on the last 2 hours") == 7200 and window_from("catch me up") is None


async def test_gather_since_their_last_message():
    history = [msg(0, "+15550001111", "heading out, later"),
               *[msg(i, "+15550002222", f"line {i}") for i in range(1, 14)],
               msg(14, "", "dziuba lost his phone again", me=True),
               msg(15, "+15550003333", "", associatedMessageType="laugh"),
               msg(16, "+15550003333", "pic", attachments=[{"mimeType": "image/jpeg"}]),
               msg(17, "+15550001111", "coco what did i miss")]
    bb = HistoryBB(history)
    lines, since = await gather(bb, SenderResolver(None, NAMES, use_contacts=False), "Coco", ALLOWED, "+15550001111",
                                None, norm=_norm_addr, now=time.time())
    assert since == " since they last said something"
    assert lines[0] == "Joe: line 1" and lines[-2] == "Coco (you): dziuba lost his phone again" and lines[-1] == "Dziuba: pic [photo]"
    assert all("Laughed" not in l and "what did i miss" not in l and "heading out" not in l for l in lines)


async def test_recap_in_the_chat(clock):
    history = [msg(i, "+15550002222" if i % 2 else "+15550003333", f"we're doing wings at 7, line {i}") for i in range(30)]
    bb = HistoryBB(history)
    llm = SysLLM("ok max, here's the damage:\n- wings at 7, joe's driving\n- dziuba lost his phone again")
    agent = Agent(make_settings(), bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, NAMES, use_contacts=False))
    assert await agent.process(message("coco catch me up", guid="r1")) == "handled:recap"
    assert bb.texts[-1][1] == "ok max, here's the damage:\n- wings at 7, joe's driving\n- dziuba lost his phone again"
    system, msgs = llm.calls[-1]
    assert "You're catching Max up on the group chat" in system and "only what's in the messages, never invent" in system
    assert msgs[0]["content"].startswith("Dziuba: we're doing wings at 7, line 0\nJoe:") and msgs[0]["content"].endswith("Now catch Max up.")
    await agent.process(message("/recap 3h", guid="r2"))
    assert bb.queries[-1][2] is not None and " in the last 3 hours" in llm.calls[-1][0]
    bb.history = history[:1]
    await agent.process(message("/recap", guid="r3"))
    assert bb.texts[-1][1] == "nothing happened. you didn't miss a thing"
