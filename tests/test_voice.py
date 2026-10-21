from pathlib import Path

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.voice import parse_pronunciations, repeat_last, speakable, wants_voice
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message


class FakeVoice:
    def __init__(self, fail=False):
        self.fail, self.said = fail, []

    async def synthesize(self, text, workdir: Path):
        self.said.append(text)
        if self.fail:
            return None
        out = workdir / "voice-note.m4a"
        out.write_bytes(b"m4a")
        return out


def test_speakable_and_triggers():
    p = parse_pronunciations("dziuba=ʤˈubə, coco=/kˈOkO/")
    assert p == {"dziuba": "ʤˈubə", "coco": "kˈOkO"}
    assert speakable("Dziuba 💀 lost https://x.com/a [sent a gif: x] again", p) == "[Dziuba](/ʤˈubə/) lost again"
    assert wants_voice("coco send a voice note") and wants_voice("coco say that out loud") and not wants_voice("coco who won")
    assert repeat_last("coco say that out loud", "coco") and not repeat_last("coco send a voice note roasting mike", "coco")
    # something to actually say is a new reply, not a re-read of the last one
    assert not repeat_last("coco say happy birthday to tom, say it out loud", "coco")
    assert repeat_last("coco say it again pls", "coco")


def _agent(clock, voice):
    bb, llm = FakeBB(), FakeLLM("mike you absolute clown")
    agent = Agent(make_settings(), bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {}, use_contacts=False), voice=voice)
    return agent, bb, llm


async def test_voice_note_reply_and_repeat(clock):
    v = FakeVoice()
    agent, bb, llm = _agent(clock, v)
    assert await agent.process(message("coco send a voice note roasting mike", guid="v1")) == "handled:voice"
    assert bb.attachments[-1][2] == b"m4a" and bb.mimes[-1] == "audio/x-m4a" and bb.texts == []
    assert v.said == ["mike you absolute clown"] and agent.context.get(ALLOWED)[-1].text == "mike you absolute clown"
    assert not bb.attachments[-1][1].exists()  # temp audio cleaned up
    await agent.process(message("/voice", guid="v2"))
    await agent.process(message("coco say that out loud", guid="v3"))
    assert v.said[-1] == "mike you absolute clown" and len(bb.attachments) == 3


async def test_voice_failure_falls_back_to_text(clock):
    agent, bb, llm = _agent(clock, FakeVoice(fail=True))
    await agent.process(message("coco voice note please", guid="v4"))
    assert bb.texts == [(ALLOWED, "mike you absolute clown")] and getattr(bb, "attachments", []) == []


def test_orpheus_text_keeps_sound_tags_and_respells():
    from imessage_agent.voice import strip_tags

    t = speakable("<Laugh> yo dziuba <script>x</script> you bet <sigh> again", respell={"dziuba": "Jooba"}, tags=True)
    assert t == "<laugh> yo Jooba x you bet <sigh> again"
    assert speakable("<laugh> yo", {"dziuba": "ʤˈubə"}) == "yo"  # Kokoro reads tags literally: dropped
    assert strip_tags("<laugh> no way <sigh>") == "no way"


async def test_expressive_voice_gets_tags_and_memory_stays_clean(clock):
    v = FakeVoice()
    v.expressive = True
    bb, llm = FakeBB(), FakeLLM("<laugh> mike you absolute clown")
    llm_systems = []
    orig = llm.reply

    async def spy(system, messages):
        llm_systems.append(system)
        return await orig(system, messages)

    llm.reply = spy
    agent = Agent(make_settings(), bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {}, use_contacts=False), voice=v)
    await agent.process(message("/voice roast mike", guid="t1"))
    assert "<laugh>" in llm_systems[-1] and "sound tags" in llm_systems[-1]
    assert v.said[-1] == "<laugh> mike you absolute clown"
    assert agent.context.get(ALLOWED)[-1].text == "mike you absolute clown"



async def test_say_something_out_loud_is_a_new_reply_not_a_repeat(clock):
    v = FakeVoice()
    agent, bb, llm = _agent(clock, v)
    agent.context.add(ALLOWED, "Coco", "yesterday's take", from_bot=True)
    await agent.process(message("coco say happy birthday to tom, say it out loud", guid="hb"))
    assert v.said[-1] == "mike you absolute clown"  # the fake model's new reply, spoken
    assert "yesterday's take" not in v.said
