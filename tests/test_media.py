import json
from urllib.parse import parse_qs, urlparse

import httpx

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.llm import parse_intent
from imessage_agent.media import GifSearch, TweetFinder, clean_query, gif_from_reaction
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

GIF_BYTES = b"GIF89a" + b"\x00" * 100


def giphy(results=1, ctype="image/gif", size=None, seen=None):
    def handler(req: httpx.Request):
        if req.url.host == "api.giphy.com":
            if seen is not None:
                seen.append({k: v[0] for k, v in parse_qs(urlparse(str(req.url)).query).items()})
            data = [{"title": f"g{i}", "images": {"downsized": {"url": f"https://media.giphy.com/{i}.gif",
                                                                 "size": str(size or len(GIF_BYTES))}}} for i in range(results)]
            return httpx.Response(200, json={"data": data})
        return httpx.Response(200, content=GIF_BYTES, headers={"content-type": ctype})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def xapi(posts, seen=None, status=200):
    def handler(req: httpx.Request):
        if seen is not None:
            seen.append({k: v[0] for k, v in parse_qs(urlparse(str(req.url)).query).items()})
        if status != 200:
            return httpx.Response(status, json={})
        return httpx.Response(200, json={"data": posts})
    return httpx.MockTransport(handler)


def tw(i, text, likes, sensitive=False):
    return {"id": str(i), "text": text, "possibly_sensitive": sensitive,
            "public_metrics": {"like_count": likes, "retweet_count": 0, "reply_count": 0, "quote_count": 0}}


def test_query_cleaning_and_gif_chime_parse():
    assert clean_query("crying jordan https://x.com/a @bob from:me #lol") == "crying jordan lol"
    assert gif_from_reaction("GIF: michael scott no") == "michael scott no"
    assert gif_from_reaction("gif:   ") is None and gif_from_reaction("lmaooo") is None


async def test_gif_search_downloads_a_gif():
    seen = []
    g = GifSearch("key", rating="pg-13", client=giphy(seen=seen))
    gif = await g.find("ricky crying jordan!!")
    assert gif is not None and gif.path.read_bytes() == GIF_BYTES and gif.query == "ricky crying jordan"
    assert seen[0]["rating"] == "pg-13" and seen[0]["q"] == "ricky crying jordan"
    gif.path.unlink()
    assert await GifSearch("key", client=giphy(results=0)).find("x y") is None
    assert await GifSearch("key", client=giphy(ctype="text/html")).find("x y") is None
    assert await GifSearch("key", client=giphy(size=50_000_000)).find("x y") is None  # too big


async def test_tweet_finder_picks_top_liked_safe_post_and_caps_per_hour():
    seen = []
    posts = [tw(1, "decent knicks joke here", 900), tw(2, "nsfw knicks thing here", 50000, sensitive=True),
             tw(3, "the best knicks joke ever", 4000), tw(4, "tiny knicks joke here", 10)]
    now = [0.0]
    f = TweetFinder("tok", min_likes=200, per_hour=2, transport=xapi(posts, seen), clock=lambda: now[0])
    t = await f.find("knicks @someone")
    assert t.id == "3" and t.url == "https://x.com/i/status/3" and t.likes == 4000
    assert seen[0]["query"] == "knicks lang:en -is:retweet -is:reply" and "possibly_sensitive" in seen[0]["tweet.fields"]
    assert (await f.find("knicks")).id == "1"  # never the same post twice
    assert f.allowed() is False and await f.find("knicks") is None and len(seen) == 2  # capped, no third search
    now[0] = 3601
    assert f.allowed() is True
    assert f.spent_usd == round(8 * 0.005, 4)


def test_router_knows_gif_and_tweet():
    assert parse_intent('{"action": "gif", "prompt": "crying jordan"}').action == "gif"
    assert parse_intent('{"action": "tweet", "prompt": ""}').action == "chat"  # no topic, no search


# -- handler --
def _agent(clock, llm_reply="yo", gifs=True, posts=None, **over):
    s = make_settings(**over)
    bb, llm = FakeBB(), FakeLLM(llm_reply)
    agent = Agent(s, bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {}, use_contacts=False),
                  gifs=GifSearch("key", client=giphy()) if gifs else None,
                  tweets=TweetFinder("tok", transport=xapi(posts or [])) if posts is not None else None)
    return agent, bb, llm


async def test_gif_command_sends_a_gif_and_remembers_it(clock):
    agent, bb, llm = _agent(clock)
    assert await agent.process(message("/gif crying jordan", guid="g1")) == "handled:gif"
    assert bb.attachments[0][2] == GIF_BYTES and bb.mimes == ["image/gif"]
    assert not bb.attachments[0][1].exists()  # temp file cleaned up
    assert agent.context.get(ALLOWED)[-1].text == "[sent a gif: crying jordan]"


async def test_gif_and_tweet_off_say_so(clock):
    agent, bb, llm = _agent(clock, gifs=False)
    await agent.process(message("/gif x", guid="g2"))
    await agent.process(message("/tweet knicks", guid="g3"))
    assert [t for _, t in bb.texts] == ["no gifs, nobody set me up with giphy", "can't pull tweets, that's not switched on"]


async def test_tweet_command_sends_the_link(clock):
    agent, bb, llm = _agent(clock, posts=[tw(7, "the best knicks joke ever", 4000)])
    assert await agent.process(message("/tweet knicks", guid="t1")) == "handled:tweet"
    assert bb.texts == [(ALLOWED, "https://x.com/i/status/7")]
    assert agent.context.get(ALLOWED)[-1].text.startswith('[sent a tweet: "the best knicks joke ever" (4000 likes)')


async def test_natural_language_gif_request_and_veto(clock):
    agent, bb, llm = _agent(clock, llm_reply=json.dumps({"action": "gif", "prompt": "dancing dog"}))
    await agent.process(message("coco send a gif of a dancing dog", guid="n1"))
    assert len(bb.attachments) == 1
    # the router says gif but nobody asked for one: vetoed, normal reply
    await agent.process(message("coco what do you think", guid="n2"))
    assert len(bb.attachments) == 1 and bb.texts


async def test_chime_in_can_be_a_gif(clock):
    agent, bb, llm = _agent(clock, llm_reply="GIF: michael scott no", reaction_chance=1.0, reaction_cooldown_seconds=0)
    await agent.process(message("dan fell off the boat again", guid="c1"))
    await agent.drain()
    assert len(bb.attachments) == 1 and bb.texts == []


async def test_unanswered_message_gets_a_gif_after_the_wait(clock, monkeypatch):
    import asyncio

    import imessage_agent.handler as h

    agent, bb, llm = _agent(clock, llm_reply="hello is anyone there", unanswered_gif_minutes=30)
    now = [1000.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: now[0])
    real_sleep = asyncio.sleep

    async def fast_sleep(sec):
        now[0] += sec
        await real_sleep(0)

    monkeypatch.setattr(h.asyncio, "sleep", fast_sleep)
    await agent.process(message("anyone want to grab food", guid="u1"))
    for _ in range(5):
        await real_sleep(0)
    assert len(bb.attachments) == 1 and agent.context.get(ALLOWED)[-1].text == "[sent a gif: hello is anyone there]"


async def test_no_gif_when_someone_answers_or_off(clock, monkeypatch):
    import asyncio

    import imessage_agent.handler as h

    agent, bb, llm = _agent(clock, llm_reply="crickets", unanswered_gif_minutes=30)
    now = [1000.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: now[0])
    real_sleep = asyncio.sleep
    first, never = asyncio.Event(), asyncio.Event()
    calls = []

    async def held_sleep(sec):
        calls.append(sec)
        if len(calls) == 1:
            await first.wait()
            now[0] = 1000.0 + 1800  # wake exactly at the first message's deadline
        else:
            await never.wait()

    monkeypatch.setattr(h.asyncio, "sleep", held_sleep)
    m1 = await agent.process(message("anyone want to grab food", guid="u2"))
    now[0] += 60
    await agent.process(message("yeah im down", guid="u3", sender="+15550002222"))  # answered
    first.set()
    for _ in range(5):
        await real_sleep(0)
    assert m1 == "stored" and bb.attachments == []
    assert agent._unanswered[ALLOWED][0].text == "yeah im down" and calls[-1] == 60  # waiting on the newer one
    await agent.drain()  # cancels the timer, doesn't wait for it
    off, bb2, _ = _agent(clock, llm_reply="crickets")
    await off.process(message("anyone", guid="u4"))
    assert off._unanswered_tasks == {}


def test_quiet_hours_parse():
    s = make_settings(unanswered_gif_quiet_hours="1-9")
    assert [h for h in range(24) if s.in_quiet_hours(h)] == list(range(1, 9))
    wrap = make_settings(unanswered_gif_quiet_hours="23-7")
    assert [h for h in range(24) if wrap.in_quiet_hours(h)] == [0, 1, 2, 3, 4, 5, 6, 23]
    assert not make_settings().in_quiet_hours(3) and not make_settings(unanswered_gif_quiet_hours="junk").in_quiet_hours(3)


async def test_no_unanswered_gif_in_quiet_hours(clock, monkeypatch):
    import asyncio

    import imessage_agent.handler as h

    agent, bb, llm = _agent(clock, llm_reply="crickets", unanswered_gif_minutes=30, unanswered_gif_quiet_hours="1-9")
    now = [1000.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(h, "_local_hour", lambda: 4)
    real_sleep = asyncio.sleep

    async def fast_sleep(sec):
        now[0] += sec
        await real_sleep(0)

    monkeypatch.setattr(h.asyncio, "sleep", fast_sleep)
    await agent.process(message("anyone up", guid="q1"))
    for _ in range(5):
        await real_sleep(0)
    assert bb.attachments == [] and agent._unanswered == {}
