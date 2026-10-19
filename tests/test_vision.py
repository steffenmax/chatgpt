import asyncio

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from tests.conftest import ALLOWED, OTHER, FakeBB, FakeLLM, make_settings, message


class FakeVision:
    def __init__(self, caption="two guys holding a trophy at a bar"):
        self.caption_text = caption
        self.calls = []

    async def caption(self, data):
        self.calls.append(data)
        return self.caption_text


def _agent(clock, vision):
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM("nice pic")
    ctx = ContextStore(5, 3600, clock=clock)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {"+15550001111": "Alex"}, use_contacts=False), vision=vision)  # type: ignore[arg-type]
    return agent, bb, llm, ctx


async def test_photo_only_message_is_captioned_into_context(clock):
    vision = FakeVision()
    agent, bb, llm, ctx = _agent(clock, vision)
    payload = message("", guid="p1", attachments=[{"guid": "att-1", "mimeType": "image/heic"}])
    assert await agent.process(payload) == "stored"
    assert vision.calls == [b"\xff\xd8fakejpeg"] and bb.downloads == ["att-1"]
    stored = ctx.get(ALLOWED)
    assert len(stored) == 1 and stored[0].text == "[photo: two guys holding a trophy at a bar]" and stored[0].sender == "Alex"


async def test_photo_with_mention_reaches_the_model(clock):
    vision = FakeVision("a corgi on a surfboard")
    agent, bb, llm, ctx = _agent(clock, vision)
    payload = message("coco rate this", guid="p2", attachments=[{"guid": "att-2", "mimeType": "image/jpeg"}])
    assert await agent.process(payload) == "handled:mention"
    seen = llm.calls[-1][-1]["content"]
    assert "coco rate this" in seen and "[photo: a corgi on a surfboard]" in seen
    assert bb.texts == [(ALLOWED, "nice pic")]


async def test_non_image_attachments_and_other_chats_are_not_captioned(clock):
    vision = FakeVision()
    agent, bb, llm, ctx = _agent(clock, vision)
    await agent.process(message("", guid="p3", attachments=[{"guid": "att-3", "mimeType": "application/pdf"}]))
    await agent.process(message("", guid="p4", chat=OTHER, attachments=[{"guid": "att-4", "mimeType": "image/png"}]))
    assert vision.calls == [] and not getattr(bb, "downloads", [])


async def test_download_failure_leaves_message_unseen_for_redelivery(clock, monkeypatch):
    vision = FakeVision()
    agent, bb, llm, ctx = _agent(clock, vision)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    payload = message("", guid="p5", attachments=[{"guid": "missing", "mimeType": "image/jpeg"}])
    assert await agent.process(payload) == "ignored:empty"
    assert vision.calls == []
    # a later redelivery (attachment now downloadable) is handled
    payload["data"]["attachments"][0]["guid"] = "att-5"
    assert await agent.process(payload) == "stored"
    assert len(ctx.get(ALLOWED)) == 1


async def test_vision_off_by_default(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert agent.vision is None and s.vision_model == ""
    await agent.process(message("", guid="p6", attachments=[{"guid": "att-6", "mimeType": "image/jpeg"}]))
    assert ctx.get(ALLOWED) == [] and not getattr(bb, "downloads", [])
