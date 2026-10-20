import json
from urllib.parse import parse_qs, urlparse

import httpx

from imessage_agent.brief import CultureBrief, DEFAULT_TOPICS, X_GROUNDING
from imessage_agent.task import TaskResult
from imessage_agent.xsearch import (
    DEFAULT_QUERIES,
    POST_READ_USD,
    XSearch,
    full_query,
    parse_queries,
    rank,
    topics_to_queries,
)
from tests.conftest import make_settings


def post(i, text, likes=0, rts=0, replies=0, quotes=0):
    return {"id": str(i), "text": text, "public_metrics": {"like_count": likes, "retweet_count": rts, "reply_count": replies, "quote_count": quotes}}


def fake_api(pages: dict[str, list[dict]] | None = None, status=200, seen: list | None = None):
    """httpx transport pretending to be /2/tweets/search/recent. Keyed by the bare query text."""
    def handler(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)
        if seen is not None:
            seen.append({k: v[0] for k, v in q.items()} | {"auth": request.headers.get("authorization", "")})
        if status != 200:
            return httpx.Response(status, json={"title": "nope"})
        bare = q["query"][0].split(" lang:")[0]
        data = (pages or {}).get(bare, [])
        return httpx.Response(200, json={"data": data, "meta": {"result_count": len(data)}})
    return httpx.MockTransport(handler)


# -- queries --

def test_default_topics_map_to_hand_tuned_queries_and_custom_topics_split():
    assert topics_to_queries("") == DEFAULT_QUERIES
    assert topics_to_queries("NY sports (Jets, Giants, Knicks); NFL; general memes") == ["(Jets OR Giants OR Knicks)", "NY sports", "NFL", "memes"]
    assert topics_to_queries("the Bills, Sabres and Buffalo food") == ["Bills", "Sabres", "Buffalo food"]
    assert topics_to_queries("Taylor Swift (eras tour, new album)") == ['("eras tour" OR "new album")', "Taylor Swift"]
    assert len(topics_to_queries(";".join(f"t{i}" for i in range(30)))) == 12
    assert parse_queries(" a OR b ; \"c d\" ;; ") == ["a OR b", '"c d"']
    assert full_query("NFL") == "NFL lang:en -is:retweet -is:reply"
    assert full_query("NFL lang:es is:reply") == "NFL lang:es is:reply -is:retweet"
    assert full_query("NFL", "-has:links is:verified") == "NFL -has:links is:verified"
    assert XSearch("t", ["a"], filters="  ").filters == "lang:en -is:retweet -is:reply"


def test_default_prose_topics_derive_sanely_too():
    qs = topics_to_queries(DEFAULT_TOPICS)
    assert qs[0] == "(Jets OR Giants OR Knicks OR Nets OR Yankees OR Mets OR Rangers OR Islanders)"
    assert "New York sports" in qs and "internet culture" in qs and "memes" in qs


# -- budget --

def test_plan_thins_then_drops_queries_to_fit_the_cap():
    x = XSearch("tok", ["a", "b", "c", "d"], posts_per_query=20, max_usd_per_refresh=0.40)  # 80 allowed
    assert x.plan() == [("a", 20), ("b", 20), ("c", 20), ("d", 20)] and x.estimate_usd() == 0.40
    x = XSearch("tok", ["a", "b", "c", "d"], posts_per_query=20, max_usd_per_refresh=0.30)  # 60: thin to 15
    assert x.plan() == [("a", 15), ("b", 15), ("c", 15), ("d", 15)]
    x = XSearch("tok", ["a", "b", "c", "d"], posts_per_query=20, max_usd_per_refresh=0.15)  # 30: min 10, drop to 3
    assert x.plan() == [("a", 10), ("b", 10), ("c", 10)]
    x = XSearch("tok", ["a"], posts_per_query=20, max_usd_per_refresh=0.02)  # 4 posts: can't even do one
    assert x.plan() == [] and x.estimate_usd() == 0
    assert XSearch("tok", [], 15).plan() == []


def test_settings_derive_queries_and_resolve_source():
    s = make_settings(brief_enabled=True)
    assert s.resolved_brief_source == "web" and "X API (brief post pulls)" not in dict(s.outbound_hosts())
    assert any("ANTHROPIC_API_KEY" in p for p in s.validate_runtime())
    s = make_settings(brief_enabled=True, x_bearer_token="t")
    assert s.resolved_brief_source == "x" and s.x_query_list() == DEFAULT_QUERIES
    hosts = dict(s.outbound_hosts())
    assert hosts["X API (brief post pulls)"] == "api.x.com:443" and "Anthropic API (daily brief)" not in hosts
    assert not any("BRIEF" in p or "X_" in p for p in s.validate_runtime())
    s = make_settings(brief_enabled=True, x_bearer_token="t", anthropic_api_key="k", x_queries="Jets; Knicks")
    assert s.resolved_brief_source == "both" and s.x_query_list() == ["Jets", "Knicks"]
    assert "Anthropic API (daily brief)" in dict(s.outbound_hosts())
    s = make_settings(brief_enabled=True, brief_source="x")
    assert any("X_BEARER_TOKEN is empty" in p for p in s.validate_runtime())
    s = make_settings(brief_enabled=True, x_bearer_token="t", x_posts_per_query=100, x_max_usd_per_refresh=0.2)
    assert any("X_MAX_USD_PER_REFRESH" in p for p in s.validate_runtime())


# -- pulling --

def test_rank_orders_by_engagement_strips_links_and_dedupes():
    posts = [post(1, "meh again https://t.co/x", likes=1), post(2, "BIG one", likes=100, rts=50), post(3, "big one!!", likes=90),
             post(4, "x", likes=999), post(5, "word " * 100, likes=5)]
    out = rank(posts)
    assert [p["id"] for p in out] == ["2", "5", "1"]  # 3 is a dup of 2, 4 is too short
    assert "http" not in out[2]["text"] and out[1]["text"].endswith("…") and len(out[1]["text"]) <= 241


async def test_pull_builds_a_digest_and_records_spend(tmp_path):
    seen: list = []
    pages = {"(Jets OR Giants) NFL": [post(1, "Jets cooked", likes=2500, rts=300), post(2, "Giants fine", likes=10)],
             "NBA": [post(3, "Knicks in 5", likes=12000)]}
    x = XSearch("tok", ["(Jets OR Giants) NFL", "NBA", "empty"], posts_per_query=10, lookback_hours=24,
                ledger_path=str(tmp_path / "x-spend.json"), transport=fake_api(pages, seen=seen), clock=lambda: 1_700_000_000.0)
    res = await x.pull()
    assert res.posts == 3 and res.queries_ok == 3 and res.queries_failed == 0 and res.cost_usd == round(3 * POST_READ_USD, 4)
    assert res.digest == ("## (Jets OR Giants) NFL\n- [2.5k likes, 300 reposts] Jets cooked\n- [10 likes, 0 reposts] Giants fine\n\n"
                          "## NBA\n- [12k likes, 0 reposts] Knicks in 5")
    assert seen[0]["auth"] == "Bearer tok" and seen[0]["max_results"] == "10" and seen[0]["sort_order"] == "relevancy"
    assert seen[0]["query"] == "(Jets OR Giants) NFL lang:en -is:retweet -is:reply" and seen[0]["start_time"].endswith("Z")
    assert "expansions" not in seen[0] and "public_metrics" in seen[0]["tweet.fields"]
    led = json.loads((tmp_path / "x-spend.json").read_text())
    assert led["refreshes"] == 1 and led["posts"] == 3 and led["usd"] == 0.015 and led["last"]["queries_ok"] == 3
    assert (tmp_path / "x-spend.json").stat().st_mode & 0o777 == 0o600
    await x.pull()
    assert x.ledger()["refreshes"] == 2 and x.ledger()["usd"] == 0.03 and x.status()["spent_usd"] == 0.03
    await x.aclose()


async def test_pull_stops_on_auth_or_credit_errors_and_cuts_digest_between_lines(tmp_path):
    seen: list = []
    x = XSearch("bad", ["a", "b", "c"], transport=fake_api(status=402, seen=seen))
    res = await x.pull()
    assert res.posts == 0 and res.queries_failed == 1 and res.errors == ["http_402"] and len(seen) == 1 and res.digest == ""
    pages = {"a": [post(i, f"post number {i} " + "blah " * 10, likes=100 - i) for i in range(20)]}
    x = XSearch("tok", ["a"], posts_per_query=20, digest_max_chars=300, transport=fake_api(pages))
    res = await x.pull()
    assert len(res.digest) <= 300 and res.digest.count("\n") >= 2 and not res.digest.endswith("bl")
    assert res.cost_usd == 0.1


# -- the brief, with X --

class Writer:
    name = "fakewriter"

    def __init__(self, text):
        self.text, self.calls = text, []

    async def reply(self, system, messages):
        self.calls.append((system, messages))
        return self.text


class Runner:
    def __init__(self, result):
        self.result, self.prompts = result, []

    async def run(self, request, requester, ctx, ask=None):
        self.prompts.append(request)
        return self.result


GOOD = "\n".join(f"- thing {i}: why it's funny" for i in range(12))
PAGES = {"NBA": [post(1, "Knicks in 5", likes=12000)], "NFL": [post(2, "Jets cooked", likes=400)]}


async def test_source_x_writes_from_the_digest_with_the_writer_only(tmp_path):
    x = XSearch("tok", ["NBA", "NFL"], transport=fake_api(PAGES), ledger_path=str(tmp_path / "x.json"))
    w = Writer(GOOD)
    b = CultureBrief(str(tmp_path / "brief.md"), None, x=x, source="x", writer=w)
    assert await b.refresh() and "thing 3" in b.raw() and b.last_cost_usd == 0.01
    system, messages = w.calls[0]
    assert DEFAULT_TOPICS in system and "48 hours" in system and "Knicks in 5" in messages[0]["content"]
    st = b.status()
    assert st["source"] == "x" and st["writer"] == "fakewriter" and st["x"]["spent_usd"] == 0.01
    await b.aclose()


async def test_source_x_fails_cleanly_when_nothing_comes_back(tmp_path):
    x = XSearch("tok", ["NBA"], transport=fake_api(status=401))
    b = CultureBrief(str(tmp_path / "brief.md"), None, x=x, source="x", writer=Writer(GOOD))
    assert not await b.refresh() and b.last_error == "http_401" and not (tmp_path / "brief.md").exists()
    x = XSearch("tok", ["NBA"], transport=fake_api({}))
    b = CultureBrief(str(tmp_path / "brief.md"), None, x=x, source="x", writer=Writer(GOOD))
    assert not await b.refresh() and b.last_error == "x_no_posts"


async def test_source_both_grounds_the_task_agent_and_falls_back_to_web(tmp_path):
    x = XSearch("tok", ["NBA", "NFL"], transport=fake_api(PAGES))
    r = Runner(TaskResult(True, text=GOOD, cost_usd=0.08))
    b = CultureBrief(str(tmp_path / "brief.md"), r, x=x, source="both")
    assert await b.refresh() and b.last_cost_usd == 0.09
    assert "Knicks in 5" in r.prompts[0] and X_GROUNDING.split("{")[0] in r.prompts[0] and "Focus: " + DEFAULT_TOPICS[:20] in r.prompts[0]
    x2 = XSearch("tok", ["NBA"], transport=fake_api(status=429))
    r2 = Runner(TaskResult(True, text=GOOD, cost_usd=0.08))
    b2 = CultureBrief(str(tmp_path / "brief2.md"), r2, x=x2, source="both")
    assert await b2.refresh() and "Start from the posts" not in r2.prompts[0] and b2.last_cost_usd == 0.08


def test_build_brief_wires_the_source(monkeypatch, tmp_path):
    from imessage_agent.main import build_brief

    path = str(tmp_path / "brief.md")
    assert build_brief(make_settings(brief_enabled=False)) is None
    assert build_brief(make_settings(brief_enabled=True, brief_file=path)) is None  # web needs a key
    b = build_brief(make_settings(brief_enabled=True, brief_file=path, x_bearer_token="t"))
    assert b.source == "x" and b.runner is None and b.writer.name == "ollama" and b.x.digest_max_chars == 10000
    assert b.x.ledger_path == tmp_path / "x-spend.json" and b.x.queries == DEFAULT_QUERIES
    b = build_brief(make_settings(brief_enabled=True, brief_file=path, x_bearer_token="t", anthropic_api_key="k",
                                  x_queries="Jets;Knicks", x_posts_per_query=30))
    assert b.source == "both" and b.runner is not None and b.writer is None and b.x.plan() == [("Jets", 30), ("Knicks", 30)]
    b = build_brief(make_settings(brief_enabled=True, brief_file=path, anthropic_api_key="k", brief_source="web"))
    assert b.source == "web" and b.x is None


async def test_per_search_results_reach_the_ledger_and_status(tmp_path):
    pages = {"NBA": [post(1, "Knicks in 5", likes=12000), post(2, "Knicks in 5", likes=3)]}
    x = XSearch("tok", ["NBA", "empty", "c", "d"], posts_per_query=10, max_usd_per_refresh=0.15, origin="X_QUERIES",
                ledger_path=str(tmp_path / "x-spend.json"), transport=fake_api(pages))
    st = x.status()
    assert st["origin"] == "X_QUERIES" and [p["query"] for p in st["plan"]] == ["NBA", "empty", "c"]
    assert st["dropped"] == ["d"] and st["plan"][0]["full"] == "NBA lang:en -is:retweet -is:reply" and st["last"] is None
    await x.pull()
    rows = x.status()["last"]["per_query"]
    assert rows[0] == {"query": "NBA", "requested": 10, "posts": 2, "kept": 1, "status": "ok"}
    assert rows[1]["posts"] == 0 and rows[1]["status"] == "ok"
    assert "Knicks" not in json.dumps(x.ledger())  # counts only, never post text
    await x.aclose()
    bad = XSearch("bad", ["a", "b", "c"], transport=fake_api(status=402))
    await bad.pull()
    assert [r["status"] for r in bad.last.per_query] == ["http_402", "skipped", "skipped"]
    await bad.aclose()


def test_x_query_origin():
    from tests.conftest import make_settings

    assert make_settings(x_queries="from:a OR from:b; NBA").x_query_origin == "X_QUERIES"
    assert make_settings(x_queries=" ", brief_topics="nfl").x_query_origin == "BRIEF_TOPICS"
    assert make_settings().x_query_origin == "defaults"


async def test_last_pull_posts_are_saved_privately_for_the_dashboard(tmp_path):
    pages = {"NBA": [post(1, "Knicks in 5", likes=12000, rts=40), post(2, "Knicks in 5", likes=3)]}
    path = tmp_path / "x-posts.json"
    x = XSearch("tok", ["NBA", "empty"], posts_per_query=10, posts_path=str(path), transport=fake_api(pages),
                clock=lambda: 1_700_000_000.0)
    assert x.last_posts() == {}
    await x.pull()
    d = x.last_posts()
    assert path.stat().st_mode & 0o777 == 0o600 and d["at"].startswith("2023-11-14")
    nba = d["searches"][0]
    assert nba["query"] == "NBA" and nba["returned"] == 2 and len(nba["posts"]) == 1  # the copy was dropped
    assert nba["posts"][0]["text"] == "Knicks in 5" and nba["posts"][0]["likes"] == 12000 and nba["posts"][0]["reposts"] == 40
    assert d["searches"][1] == {"query": "empty", "returned": 0, "posts": []}
    await x.aclose()
    # a pull where every search fails keeps the last good posts
    bad = XSearch("bad", ["NBA"], posts_path=str(path), transport=fake_api(status=402))
    await bad.pull()
    assert bad.last_posts()["searches"][0]["posts"][0]["text"] == "Knicks in 5"
    await bad.aclose()
