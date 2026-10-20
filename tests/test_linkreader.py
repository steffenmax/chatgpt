import asyncio
import json

import httpx

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.linkreader import LinkReader, _syndication_token, find_urls, strip_cards
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

PUBLIC = {h: False for h in ("api.fxtwitter.com", "cdn.syndication.twimg.com", "www.youtube.com", "news.example",
                             "pbs.twimg.com", "short.example")}

CONVERSATION = {
    "status": {
        "text": "the jets just traded for a punter", "author": {"name": "Jets Fan", "screen_name": "jetsfan"},
        "likes": 12345, "replies": 410, "media": {"photos": [{"url": "https://pbs.twimg.com/media/a.jpg"}]},
    },
    "replies": [
        {"text": "@jetsfan classic jets", "likes": 90, "author": {"screen_name": "a"}},
        {"text": "@jetsfan @nyjets same as it ever was", "likes": 2500, "author": {"screen_name": "b"}},
        {"text": "more in my thread", "likes": 99999, "author": {"screen_name": "jetsfan"}},
        {"text": "@jetsfan", "likes": 5000, "author": {"screen_name": "c"}},
    ],
}

PAGE = b"""<html><head><title>Big Story - News</title><meta property="og:site_name" content="Example News">
<meta property="og:description" content="Something happened today."><script>var x = "not text";</script></head>
<body><nav><p>Home Sports Weather and a long navigation paragraph that should be skipped entirely</p></nav>
<article><p>The opening paragraph explains what happened in enough words to count as text.[1]</p></article></body></html>"""


class FakeVision:
    def __init__(self):
        self.calls = 0

    async def caption(self, data):
        self.calls += 1
        return "a guy kicking a football"


def _reader(handler, **kw):
    lr = LinkReader(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)
    lr._hosts.update(PUBLIC)
    return lr


def _routes(req: httpx.Request) -> httpx.Response:
    url = str(req.url)
    if url.startswith("https://api.fxtwitter.com/2/conversation/123"):
        return httpx.Response(200, json=CONVERSATION)
    if url.startswith("https://cdn.syndication.twimg.com/tweet-result?id=123"):
        return httpx.Response(200, json={"text": "post only", "user": {"name": "Jets Fan", "screen_name": "jetsfan"},
                                         "favorite_count": 7})
    if url.startswith("https://pbs.twimg.com/"):
        return httpx.Response(200, content=b"\xff\xd8jpg", headers={"content-type": "image/jpeg"})
    if url.startswith("https://www.youtube.com/oembed"):
        return httpx.Response(200, json={"title": "Top 10 fumbles", "author_name": "NFL"})
    if url == "https://short.example/x":
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8766/health"})
    if url == "https://news.example/story":
        return httpx.Response(200, content=PAGE, headers={"content-type": "text/html; charset=utf-8"})
    return httpx.Response(404)


def test_find_urls_and_strip_cards():
    assert find_urls("lol https://x.com/a/status/1?s=20, and (https://b.example/p). https://c.example") == [
        "https://x.com/a/status/1?s=20", "https://b.example/p"]
    assert strip_cards('look at this https://x.com/a [link: X post by A (@a): "hi"]') == "look at this https://x.com/a"


def test_syndication_token_is_base36_without_zeros_or_dot():
    t = _syndication_token("1850000000000000000")
    assert t and t.isalnum() and "0" not in t and t.startswith("4hf")  # 5811.9... is 4hf.x in base 36


async def test_x_post_with_top_replies_and_photo_caption():
    vision = FakeVision()
    lr = _reader(_routes, vision=vision)
    card = await lr.read("https://x.com/jetsfan/status/123?s=46")
    assert card.startswith('[link: X post by Jets Fan (@jetsfan), 12K likes, 410 replies: "the jets just traded for a punter"')
    assert "photo: a guy kicking a football" in card and vision.calls == 1
    # most liked first, the author's own thread skipped, leading @mentions dropped, bare mentions skipped
    assert 'top replies: @b (2.5K): "same as it ever was"; @a (90): "classic jets"' in card
    assert "my thread" not in card and "@c" not in card
    assert await lr.read("https://x.com/jetsfan/status/123?s=46") == card  # cached


async def test_x_falls_back_to_the_embed_endpoint():
    def handler(req):
        if req.url.host == "api.fxtwitter.com":
            return httpx.Response(503)
        return _routes(req)

    card = await _reader(handler).read("https://twitter.com/jetsfan/status/123")
    assert card == '[link: X post by Jets Fan (@jetsfan), 7 likes: "post only"]'
    card = await _reader(_routes, x_replies=False).read("https://x.com/jetsfan/status/123")
    assert "post only" in card


async def test_youtube_and_plain_pages():
    lr = _reader(_routes)
    assert await lr.read("https://www.youtube.com/watch?v=abc") == '[link: YouTube video by NFL: "Top 10 fumbles"]'
    card = await lr.read("https://news.example/story")
    assert card.startswith('[link: Example News page: "Big Story - News" | Something happened today. | The opening paragraph')
    assert "navigation" not in card and "not text" not in card and "(1)" not in card and "[1]" not in card


async def test_local_addresses_are_never_fetched():
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return _routes(req)

    lr = _reader(handler)
    assert await lr.read("http://127.0.0.1:8766/health") == "[link: couldn't open it]"
    assert await lr.read("https://short.example/x") == "[link: couldn't open it]"  # redirect into loopback
    assert seen == ["https://short.example/x"]


# -- handler --
def _agent(clock, links):
    s = make_settings()
    bb, llm = FakeBB(), FakeLLM("haha the jets")
    ctx = ContextStore(10, 3600, clock=clock)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {"+15550001111": "Alex"}, use_contacts=False),  # type: ignore[arg-type]
                  links=links)
    return agent, bb, llm, ctx


class SlowLinks:
    def __init__(self):
        self.release = asyncio.Event()
        self.urls = []

    async def read(self, url):
        self.urls.append(url)
        await self.release.wait()
        return '[link: X post by A (@a): "the jets traded for a punter"]'


async def test_link_card_lands_in_context_and_a_later_mention_waits_for_it(clock):
    links = SlowLinks()
    agent, bb, llm, ctx = _agent(clock, links)
    assert await agent.process(message("bro https://x.com/a/status/1", guid="l1")) == "stored"
    mention = asyncio.create_task(agent.process(message("coco thoughts?", guid="l2")))
    await asyncio.sleep(0.05)
    assert not mention.done() and bb.texts == []  # waiting on the link
    links.release.set()
    assert await mention == "handled:mention"
    seen = json.dumps(llm.calls[-1])
    assert "the jets traded for a punter" in seen and "When someone shares a link" in agent._system_for(ALLOWED)
    assert ctx.get(ALLOWED)[0].text.endswith('[link: X post by A (@a): "the jets traded for a punter"]')
    assert links.urls == ["https://x.com/a/status/1"]


async def test_links_off_by_default(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert agent.links is None and s.links_enabled is False
    await agent.process(message("https://x.com/a/status/1", guid="l3"))
    assert ctx.get(ALLOWED)[0].text == "https://x.com/a/status/1"


class InstantLinks:
    def __init__(self, card='[link: X post by A (@a): "the jets traded for a punter"]'):
        self.card = card

    async def read(self, url):
        return self.card


def _quiet_agent(clock, links, monkeypatch):
    s = make_settings(link_quiet_seconds=90, reaction_chance=1.0, reaction_cooldown_seconds=0)
    bb, llm = FakeBB(), FakeLLM("a punter?? jets gonna jets")
    ctx = ContextStore(10, 3600, clock=clock)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {}, use_contacts=False), links=links)  # type: ignore[arg-type]
    gate = asyncio.Event()
    real_sleep = asyncio.sleep

    async def sleep(sec):
        if sec == 90:
            await gate.wait()
        else:
            await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return agent, bb, llm, gate


async def test_unanswered_link_gets_a_take(clock, monkeypatch):
    agent, bb, llm, gate = _quiet_agent(clock, InstantLinks(), monkeypatch)
    assert await agent.process(message("https://x.com/a/status/1", guid="q1")) == "stored"
    await asyncio.sleep(0)
    assert bb.texts == []  # no instant reaction to a link
    gate.set()
    await agent.drain()
    assert bb.texts == [(ALLOWED, "a punter?? jets gonna jets")]


async def test_link_take_skipped_when_someone_replies_or_it_failed(clock, monkeypatch):
    agent, bb, llm, gate = _quiet_agent(clock, InstantLinks(), monkeypatch)
    agent.s.reaction_chance = 0
    await agent.process(message("https://x.com/a/status/1", guid="q2"))
    await agent.process(message("lol saw this", guid="q3"))
    gate.set()
    await agent.drain()
    assert bb.texts == []
    agent, bb, llm, gate = _quiet_agent(clock, InstantLinks("[link: couldn't open it]"), monkeypatch)
    await agent.process(message("https://www.reddit.com/r/nfl/x", guid="q4"))
    gate.set()
    await agent.drain()
    assert bb.texts == []
