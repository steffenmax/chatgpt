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
    # the bot remembers what it posted, as its own line; the human request itself is not stored
    sent = [m for m in ctx.get(ALLOWED) if m.from_bot and m.text.startswith("[sent a picture:")]
    assert [m.text for m in sent] == ["[sent a picture: a corgi surfing]", "[sent a picture: a cat]"]
    assert all("corgi" not in m.text for m in ctx.get(ALLOWED) if not m.from_bot)


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


async def test_mention_waits_while_image_generates():
    """With a shared gate, the LLM is not called while mflux is running (RAM safety)."""
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM()
    ctx = ContextStore(5, 3600)
    holder = {}
    events = []
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowGenerator(FakeGenerator):
        async def generate(self, prompt, workdir):
            events.append("gen.start")
            started.set()
            await release.wait()
            events.append("gen.end")
            return await super().generate(prompt, workdir)

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    async def unload():
        events.append("unload")

    gate = asyncio.Semaphore(1)
    q = ImageQueue(SlowGenerator(), send, fail, before_generate=unload, gate=gate)
    agent = Agent(
        s, bb, llm, ctx, images=q,
        resolver=SenderResolver(None, {}, use_contacts=False),  # type: ignore[arg-type]
        llm_gate=gate,
    )
    holder["a"] = agent
    q.start()
    try:
        await agent.process(message("/image a corgi surfing", guid="i1"))
        await asyncio.wait_for(started.wait(), timeout=5)
        mention = asyncio.create_task(agent.process(message(f"{s.bot_name} you there?", guid="m1")))
        await asyncio.sleep(0.05)
        assert len(llm.calls) == 0, "LLM must not run while an image is generating"
        release.set()
        await asyncio.wait_for(mention, timeout=5)
        await asyncio.wait_for(q._queue.join(), timeout=5)
    finally:
        await q.stop()

    assert len(llm.calls) == 2  # dispatcher + reply, both after the image finished
    assert events == ["unload", "gen.start", "gen.end"]


async def test_stop_tells_the_chat_about_lost_image_jobs():
    """A restart mid-generation (and anything still queued) must not fail silently."""
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM()
    ctx = ContextStore(5, 3600)
    holder = {}
    started = asyncio.Event()

    class HangingGenerator(FakeGenerator):
        async def generate(self, prompt, workdir):
            started.set()
            await asyncio.sleep(3600)

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    q = ImageQueue(HangingGenerator(), send, fail)
    agent = Agent(s, bb, llm, ctx, images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    holder["a"] = agent
    q.start()
    await agent.process(message("/image one", guid="s1"))
    await agent.process(message("/image two", guid="s2"))
    await asyncio.wait_for(started.wait(), timeout=5)
    await q.stop()
    notices = [t for _, t in bb.texts if "restarted" in t]
    assert len(notices) == 2, bb.texts
    assert q.qsize() == 0

