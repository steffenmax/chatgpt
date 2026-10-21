from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from tests.conftest import ALLOWED, FakeBB, make_settings, message

NAMES = {"+15550001111": "Max", "+15550002222": "Tom O'Brien", "+15550003333": "Joe"}


class Cloud:
    name = "anthropic"

    def __init__(self):
        self.calls = []

    async def reply(self, system, messages):
        self.calls.append((system, messages))
        return "- Literally: he's 'fine'.\n- Actually: he's furious about the 20 bucks.\n- Subtext: passive-aggressive shot at Max."

    async def healthy(self):
        return True

    async def aclose(self):
        pass


class Local:
    name = "fake"

    def __init__(self, reply):
        self.reply_text, self.systems = reply, []

    async def reply(self, system, messages, stop=None):
        self.systems.append(system)
        return self.reply_text

    async def healthy(self):
        return True

    async def aclose(self):
        pass


def _agent(clock, local_reply="ENHANCE. tom says he's fine, he is NOT fine: furious about the 20 bucks, and that's a "
                               "passive-aggressive shot at max dressed up as a joke"):
    bb, cloud, local = FakeBB(), Cloud(), Local(local_reply)
    agent = Agent(make_settings(), bb, local, ContextStore(15, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, NAMES, use_contacts=False), cloud=cloud)
    return agent, bb, cloud, local


async def test_enhance_last_message_by_default(clock):
    agent, bb, cloud, local = _agent(clock)
    await agent.process(message("i'm fine. totally fine.", guid="e1", sender="+15550002222"))
    assert await agent.process(message("coco enhance this", guid="e2")) == "handled:enhance"
    system, msgs = cloud.calls[-1]
    assert "enhance" in system and 'Target message, from Tom O\'Brien: "i\'m fine. totally fine."' in msgs[0]["content"]
    assert agent.s.persona.split(".")[0] not in system  # the analyst gets no persona
    assert "Your analyst's breakdown" in local.systems[-1] and bb.texts[-1][1].startswith("ENHANCE. tom says he's fine")


async def test_enhance_named_person_and_swipe_reply(clock):
    agent, bb, cloud, local = _agent(clock)
    await agent.process(message("i'm fine. totally fine.", guid="n1", sender="+15550002222"))
    await agent.process(message("anyway who's driving", guid="n2", sender="+15550003333"))
    await agent.process(message("coco enhance tom's message", guid="n3"))
    assert 'from Tom O\'Brien: "i\'m fine. totally fine."' in cloud.calls[-1][1][0]["content"]
    bb.known_messages = {"old-1": {"guid": "old-1", "text": "nah i'm good on wings", "isFromMe": False,
                                   "handle": {"address": "+15550003333"}}}
    await agent.process(message("coco enhance this", guid="n4", threadOriginatorGuid="old-1"))
    assert 'from Joe: "nah i\'m good on wings"' in cloud.calls[-1][1][0]["content"]


async def test_enhance_falls_back_to_the_breakdown_when_the_voice_garbles(clock):
    agent, bb, cloud, local = _agent(clock, local_reply="tom owes max 500 dollars lol")  # invented number
    await agent.process(message("i'm fine.", guid="g1", sender="+15550002222"))
    await agent.process(message("coco enhance", guid="g2"))
    assert bb.texts[-1][1].startswith("🔍 enhancing Tom...\n- Literally: he's 'fine'.")


async def test_long_breakdowns_are_cut_between_bullets(clock):
    agent, bb, cloud, local = _agent(clock, local_reply="")  # no voice: the breakdown goes out
    bullets = "\n".join(f"- point {i}: " + "word " * 30 for i in range(8))

    async def long_reply(system, messages):
        return bullets

    cloud.reply = long_reply
    await agent.process(message("i'm fine.", guid="l1", sender="+15550002222"))
    await agent.process(message("coco enhance", guid="l2"))
    out = bb.texts[-1][1]
    assert out.rstrip().endswith("word") and not out.rstrip().endswith("-") and len(out) < 1000
