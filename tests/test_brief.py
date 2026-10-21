import asyncio
import os
from datetime import datetime

from fastapi.testclient import TestClient

from imessage_agent.brief import BRIEF_PROMPT, CultureBrief, DEFAULT_TOPICS
from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.main import create_app
from imessage_agent.task import TaskResult
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

GOOD = "\n".join(f"- thing {i}: why it's funny https://x.com/{i}" for i in range(18))


class Runner:
    def __init__(self, result):
        self.result, self.prompts = result, []

    async def run(self, request, requester, ctx, ask=None):
        self.prompts.append(request)
        return self.result


def test_clean_formats_caps_and_strips_links():
    out = CultureBrief.clean("intro line\n" + GOOD + "\n• bullet style: ok\noutro")
    lines = out.splitlines()
    assert len(lines) == 15 and all(l.startswith("- ") for l in lines) and "http" not in out


async def test_refresh_writes_a_private_file_and_failures_keep_the_old_one(tmp_path):
    path = tmp_path / "b" / "brief.md"
    b = CultureBrief(str(path), Runner(TaskResult(True, text=GOOD, cost_usd=0.08)))
    assert await b.refresh()
    text = path.read_text()
    assert text.startswith("## What people are talking about (updated ") and "- thing 0: why it's funny" in text
    assert (path.stat().st_mode & 0o777) == 0o600 and b.load() == text.strip()
    assert DEFAULT_TOPICS in b.runner.prompts[0] and "Skip deaths" in BRIEF_PROMPT
    b.runner = Runner(TaskResult(True, text="- just one line"))
    assert not await b.refresh() and path.read_text() == text and b.last_error == "too_short"
    b.runner = Runner(TaskResult(False, reason="error_max_budget_usd"))
    assert not await b.refresh() and b.last_error == "error_max_budget_usd"


def test_stale_brief_is_not_used_and_schedule(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text("## x\n- a: b\n")
    old = path.stat().st_mtime - 4 * 86400
    os.utime(path, (old, old))
    b = CultureBrief(str(path), Runner(None), at="08:00")
    assert b.load() == "" and b.raw().startswith("## x")
    assert b.next_run(datetime(2026, 10, 20, 7, 0)) == datetime(2026, 10, 20, 8, 0)
    assert b.next_run(datetime(2026, 10, 20, 9, 0)) == datetime(2026, 10, 21, 8, 0)


async def test_brief_goes_into_the_prompt(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text("## What people are talking about\n- the jets: cooked again\n")

    class LLM(FakeLLM):
        async def reply(self, system, messages):
            self.systems = getattr(self, "systems", []) + [system]
            return await super().reply(system, messages)

    llm = LLM("lol")
    agent = Agent(make_settings(), FakeBB(), llm, ContextStore(5, 3600), images=None,
                  resolver=SenderResolver(None, {}, use_contacts=False), brief=CultureBrief(str(path), Runner(None)))  # type: ignore[arg-type]
    await agent.process(message("coco thoughts on sunday", guid="b1"))
    assert "the jets: cooked again" in llm.systems[-1] and "never force" in llm.systems[-1]


def test_dashboard_brief_view_edit_refresh(tmp_path):
    s = make_settings(dashboard_enabled=True)
    path = tmp_path / "brief.md"
    brief = CultureBrief(str(path), Runner(TaskResult(True, text=GOOD)))
    agent = Agent(s, FakeBB(), FakeLLM(), ContextStore(5, 3600), images=None,
                  resolver=SenderResolver(None, {}, use_contacts=False), brief=brief)  # type: ignore[arg-type]
    with TestClient(create_app(s, agent)) as c:
        d = c.get("/dashboard/api/brief").json()
        assert d["text"] == "" and d["age_hours"] is None and d["in_prompt"] is False
        assert c.put("/dashboard/api/brief", json={"text": "## mine\n- edited: by hand"}).json()["ok"]
        assert c.get("/dashboard/api/brief").json()["in_prompt"] is True and "edited: by hand" in path.read_text()
        assert c.post("/dashboard/api/brief/refresh").json()["ok"]
        for _ in range(50):
            if "thing 0" in path.read_text():
                break
            import time; time.sleep(0.02)
        assert "thing 0" in path.read_text()


async def test_long_lines_are_trimmed_and_cap_cuts_between_lines(tmp_path):
    long = "\n".join(f"- item {i}: " + "word " * 80 for i in range(15))
    b = CultureBrief(str(tmp_path / "brief.md"), Runner(TaskResult(True, text=long)), max_chars=700)
    assert await b.refresh()
    lines = b.raw().strip().splitlines()
    assert all(len(l) <= 181 for l in lines[1:]) and all(l.endswith("…") for l in lines[1:])
    assert len(b.raw().strip()) <= 700

