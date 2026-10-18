import asyncio
from pathlib import Path

from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.context import ContextStore
from imessage_agent.imagegen import ImageQueue, MfluxGenerator
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message


class FakeGenerator(MfluxGenerator):
    def __init__(self):
        super().__init__()
        self.prompts = []
        self.dirs = []

    async def generate(self, prompt: str, workdir: Path) -> Path:
        self.prompts.append(prompt)
        self.dirs.append(workdir)
        await asyncio.sleep(0.01)
        out = workdir / "image.png"
        out.write_bytes(b"\x89PNG-fake")
        return out


async def test_image_flow_queue_and_cleanup():
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM()
    ctx = ContextStore(5, 3600)
    holder = {}

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    gen = FakeGenerator()
    q = ImageQueue(gen, send, fail)
    agent = Agent(s, bb, llm, ctx, images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    holder["a"] = agent
    q.start()
    try:
        assert await agent.process(message("/image a corgi surfing", guid="i1")) == "handled:image"
        assert await agent.process(message("/image a cat", guid="i2")) == "handled:image"
        assert bb.texts[0][1].startswith("on it")
        assert "ahead" in bb.texts[1][1]
        await asyncio.wait_for(q._queue.join(), timeout=5)
    finally:
        await q.stop()

    assert gen.prompts == ["a corgi surfing", "a cat"]
    assert len(bb.attachments) == 2
    assert bb.attachments[0][0] == ALLOWED and bb.attachments[0][2] == b"\x89PNG-fake"
    # temp dirs (image + prompt file) are gone after sending
    for d in gen.dirs:
        assert not d.exists()
    # prompt never enters chat context
    assert all("corgi" not in m.text for m in ctx.get(ALLOWED))


async def test_image_without_prompt():
    s = make_settings()
    bb = FakeBB()
    gen = FakeGenerator()
    q = ImageQueue(gen, lambda j, p: None, lambda j, r: None)  # type: ignore[arg-type]
    agent = Agent(s, bb, FakeLLM(), ContextStore(5, 3600), images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    await agent.process(message("/image", guid="e1"))
    assert "give me something" in bb.texts[0][1]
    assert q.qsize() == 0


async def test_image_failure_notifies():
    s = make_settings()
    bb = FakeBB()

    class Broken(FakeGenerator):
        async def generate(self, prompt, workdir):
            raise RuntimeError("boom: " + prompt)

    holder = {}

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    q = ImageQueue(Broken(), send, fail)
    agent = Agent(s, bb, FakeLLM(), ContextStore(5, 3600), images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    holder["a"] = agent
    q.start()
    try:
        await agent.process(message("/image a dragon", guid="f1"))
        await asyncio.wait_for(q._queue.join(), timeout=5)
    finally:
        await q.stop()
    assert any("didn't come out" in t for _, t in bb.texts)
    assert bb.attachments == []
