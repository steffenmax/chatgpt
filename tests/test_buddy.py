import asyncio
import json

from imessage_agent.context import ContextStore
from imessage_agent.handler import BUDDY_COMEBACKS, BUDDY_THIS_U, Agent, SenderResolver, _norm_addr
from imessage_agent.likeness import Likenesses
from tests.conftest import ALLOWED, FakeBB, make_settings, message

DZ = "+15550003333"


class MoodLLM:
    name = "fake"

    def __init__(self, label, reply="you got this dziuba, rough days end"):
        self.label, self.reply_text, self.systems = label, reply, []

    async def reply(self, system, messages, stop=None):
        self.systems.append(system)
        return json.dumps({"label": self.label}) if "Pick ONE label" in system else self.reply_text

    async def healthy(self):
        return True

    async def aclose(self):
        pass


class Q:
    def __init__(self):
        self.jobs = []

    def qsize(self):
        return 0

    async def submit(self, job):
        self.jobs.append(job)


def _agent(clock, tmp_path, label, opted_in=True, **over):
    d = tmp_path / "likeness" / "dz"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dz.safetensors").write_bytes(b"x")
    (d / "likeness.json").write_text(json.dumps({"name": "dziuba", "aliases": ["dziuba"], "lora": "dz.safetensors",
                                                  "owner": DZ, "opted_in": opted_in}))
    bb, llm, q = FakeBB(), MoodLLM(label), Q()
    agent = Agent(make_settings(buddy_address=DZ, **over), bb, llm, ContextStore(10, 3600, clock=clock), images=q,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {DZ: "Dziuba", "+15550001111": "Max"}, use_contacts=False),
                  likenesses=Likenesses(str(tmp_path / "likeness"), norm=_norm_addr))
    return agent, bb, llm, q


async def test_down_gets_encouragement_once_per_cooldown(clock, tmp_path):
    agent, bb, llm, q = _agent(clock, tmp_path, "down")
    await agent.process(message("man today has been rough", guid="d1", sender=DZ))
    await agent.drain()
    assert bb.texts == [(ALLOWED, "you got this dziuba, rough days end")]
    assert "He sounds down" in llm.systems[-1] and "no roasting this time" in llm.systems[-1]
    await agent.process(message("still rough", guid="d2", sender=DZ))
    await agent.drain()
    assert len(bb.texts) == 1  # cooldown


async def test_jab_at_the_bot_gets_a_this_u_picture(clock, tmp_path):
    agent, bb, llm, q = _agent(clock, tmp_path, "jab")
    await agent.process(message("ricky is the dumbest bot ever made", guid="j1", sender=DZ))
    await agent.drain()
    job = q.jobs[-1]
    assert job.prompt.startswith("dziuba, a man. dziuba ") and job.loras and job.caption in [t.replace("{name}", "dziuba") for t in BUDDY_THIS_U]
    # the model says "jab" but it's aimed at a person, not the bot: nothing
    agent2, bb2, _, q2 = _agent(clock, tmp_path / "b", "jab")
    await agent2.process(message("mike you're an idiot", guid="j2", sender=DZ))
    await agent2.drain()
    assert q2.jobs == [] and bb2.texts == []


async def test_others_fed_up_opted_out_and_muted(clock, tmp_path):
    agent, bb, llm, q = _agent(clock, tmp_path, "jab")
    await agent.process(message("ricky is trash", guid="o1", sender="+15550001111"))  # not Dziuba
    await agent.drain()
    assert q.jobs == [] and not any("Pick ONE label" in s for s in llm.systems)
    fed = _agent(clock, tmp_path / "f", "fed_up")
    await fed[0].process(message("i'm so tired of ricky", guid="o2", sender=DZ))
    await fed[0].drain()
    assert fed[3].jobs == [] and fed[1].texts == []  # no special reaction
    out = _agent(clock, tmp_path / "x", "jab", opted_in=False)
    await out[0].process(message("ai is so cringe", guid="o3", sender=DZ))
    await out[0].drain()
    assert out[3].jobs == [] and out[1].texts[-1][1] in [c.replace("{name}", "dziuba") for c in BUDDY_COMEBACKS]
    muted = _agent(clock, tmp_path / "m", "down")
    await muted[0].process(message("/mute 10m", guid="o4"))
    await muted[0].process(message("rough day", guid="o5", sender=DZ))
    await muted[0].drain()
    assert len(muted[1].texts) == 1  # just the mute ack
    _ = asyncio
