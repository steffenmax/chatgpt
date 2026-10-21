import asyncio
import base64
import json

import httpx

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.imagegen import ImageGenError, ImageQueue, OpenAIImageGenerator
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

PNG = b"\x89PNG\r\n\x1a\nfake"


def _gen(handler):
    return OpenAIImageGenerator("sk-test-key", transport=httpx.MockTransport(handler))


async def test_generate_writes_png_from_b64(tmp_path):
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    gen = _gen(handler)
    out = await gen.generate("a corgi surfing", tmp_path)
    assert out.read_bytes() == PNG and (out.stat().st_mode & 0o777) == 0o600
    assert seen["auth"] == "Bearer sk-test-key"
    assert seen["body"]["prompt"] == "a corgi surfing" and seen["body"]["model"] == "gpt-image-1" and seen["body"]["n"] == 1
    await gen.aclose()


async def test_moderation_block_is_a_rejection(tmp_path):
    def handler(request):
        return httpx.Response(400, json={"error": {"code": "moderation_blocked", "message": "nope"}})

    gen = _gen(handler)
    try:
        await gen.generate("something", tmp_path)
        assert False, "expected ImageGenError"
    except ImageGenError as exc:
        assert str(exc) == "rejected"
    await gen.aclose()


async def test_bad_key_and_check(tmp_path):
    def handler(request):
        if request.url.path.startswith("/v1/models/"):
            return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

    gen = _gen(handler)
    assert await gen.check() is False
    try:
        await gen.generate("x", tmp_path)
        assert False
    except ImageGenError as exc:
        assert "401" in str(exc)
    await gen.aclose()


async def test_rejected_image_tells_the_chat_nicely():
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM()
    ctx = ContextStore(5, 3600)
    holder = {}

    def handler(request):
        return httpx.Response(400, json={"error": {"code": "moderation_blocked"}})

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    q = ImageQueue(_gen(handler), send, fail)
    agent = Agent(s, bb, llm, ctx, images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    holder["a"] = agent
    q.start()
    try:
        await agent.process(message("/image something awful", guid="o1"))
        await asyncio.wait_for(q._queue.join(), timeout=5)
    finally:
        await q.stop()
    assert bb.texts[-1][1].startswith("not drawing that one")
    assert bb.attachments == []


def test_openai_backend_config():
    s = make_settings(image_backend="openai", openai_api_key="")
    assert any("OPENAI_API_KEY" in p for p in s.validate_runtime())
    s = make_settings(image_backend="openai", openai_api_key="sk-x")
    assert not any("OPENAI" in p for p in s.validate_runtime())
    assert dict(s.outbound_hosts())["OpenAI API (image generation)"] == "api.openai.com:443"
    assert "OpenAI API (image generation)" not in dict(make_settings().outbound_hosts())
    assert "OpenAI API (image generation)" not in dict(make_settings(image_backend="openai", openai_api_key="k", image_enabled=False).outbound_hosts())
