import asyncio
from pathlib import Path

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.task import TaskResult, TaskRunner, guard_decision
from tests.conftest import ALLOWED, OTHER, FakeBB, FakeLLM, make_settings, message


class ResultMessage:  # the runner matches on the class name, like the SDK's
    def __init__(self, result="done", subtype="success", is_error=False, cost=0.12, turns=4):
        self.result, self.subtype, self.is_error, self.total_cost_usd, self.num_turns = result, subtype, is_error, cost, turns


def test_guard_allows_web_and_sandboxed_files_only(tmp_path):
    root = tmp_path / "tasks"
    (root / "task-1").mkdir(parents=True)
    assert guard_decision("WebSearch", {"query": "x"}, root) is None
    assert guard_decision("WebFetch", {"url": "https://a.com"}, root) is None
    assert guard_decision("Write", {"file_path": str(root / "task-1" / "plan.md")}, root) is None
    assert guard_decision("Read", {"file_path": "notes.md"}, root) is None
    assert guard_decision("Read", {"file_path": str(Path.home() / ".ssh" / "id_rsa")}, root)
    assert guard_decision("Read", {"file_path": "~/Documents/x"}, root)
    assert guard_decision("Read", {"file_path": "../../.env"}, root)
    assert guard_decision("Write", {"file_path": str(root / ".." / "escape.txt")}, root)
    assert guard_decision("Grep", {"pattern": "x", "path": "/Users"}, root)
    assert guard_decision("Glob", {"pattern": "/Users/**/*.md"}, root)
    assert guard_decision("Glob", {"pattern": "*.md"}, root) is None
    for tool in ("Bash", "Task", "NotebookEdit", "mcp__x__y", "KillShell"):
        assert guard_decision(tool, {}, root), tool


async def test_runner_collects_result_files_and_cleans_transcripts(tmp_path):
    seen = {}

    async def fake_query(prompt, options):
        seen["prompt"], seen["cwd"] = prompt, options["cwd"]
        Path(options["cwd"], "itinerary.md").write_text("day 1 ...")
        (tmp_path / "tasks" / ".claude-config" / "projects" / "p").mkdir(parents=True, exist_ok=True)
        yield object()
        yield ResultMessage("here's the plan, itinerary.md has details")

    r = TaskRunner("sk-test", "claude-opus-5-5", str(tmp_path / "tasks"), query_fn=fake_query)
    res = await r.run("plan vegas", "Max", ["Mike: we going in march?"])
    assert res.ok and res.text.startswith("here's the plan") and res.cost_usd == 0.12 and res.turns == 4
    assert [f.name for f in res.files] == ["itinerary.md"]
    assert "Task from Max: plan vegas" in seen["prompt"] and "Mike: we going in march?" in seen["prompt"]
    assert Path(seen["cwd"]).parent == tmp_path / "tasks" and (Path(seen["cwd"]).stat().st_mode & 0o777) == 0o700
    assert not (tmp_path / "tasks" / ".claude-config" / "projects").exists()


async def test_runner_reports_limits_and_timeouts(tmp_path):
    async def capped(prompt, options):
        yield ResultMessage(None, subtype="error_max_budget_usd", is_error=True)

    res = await TaskRunner("k", "m", str(tmp_path / "a"), query_fn=capped).run("x", "Max", [])
    assert not res.ok and res.reason == "error_max_budget_usd"

    async def slow(prompt, options):
        await asyncio.sleep(5)
        yield ResultMessage()

    res = await TaskRunner("k", "m", str(tmp_path / "b"), timeout=0.05, query_fn=slow).run("x", "Max", [])
    assert not res.ok and res.reason == "timeout"


class FakeRunner:
    def __init__(self, result: TaskResult, delay=0.0):
        self.result, self.delay, self.calls = result, delay, []

    async def run(self, request, requester, ctx):
        self.calls.append((request, requester, ctx))
        await asyncio.sleep(self.delay)
        return self.result


def _agent(runner, **over):
    s = make_settings(task_chat_guids=[ALLOWED], **over)
    bb = FakeBB()
    agent = Agent(s, bb, FakeLLM(), ContextStore(10, 3600), images=None,
                  resolver=SenderResolver(None, {"+15550001111": "Max"}, use_contacts=False), tasks=runner)  # type: ignore[arg-type]
    return agent, bb


async def test_task_command_delivers_text_and_files(tmp_path):
    doc = tmp_path / "plan.md"
    doc.write_text("details")
    runner = FakeRunner(TaskResult(True, text="3 spots: a, b, c", files=[doc], cost_usd=0.3, turns=6))
    agent, bb = _agent(runner)
    await agent.process(message("we should do steak saturday", guid="t0"))
    assert await agent.process(message("/task find steakhouses near msg", guid="t1")) == "handled:task"
    await agent._task_job
    assert bb.texts[0][1].startswith("on it")
    assert bb.texts[-1] == (ALLOWED, "3 spots: a, b, c")
    assert [a[1].name for a in bb.attachments] == ["plan.md"]
    req, who, ctx = runner.calls[0]
    assert req == "find steakhouses near msg" and who == "Max" and ctx == ["Max: we should do steak saturday"]
    assert any("[finished a task for Max" in m.text for m in agent.context.get(ALLOWED))


async def test_task_only_from_task_chats_and_one_at_a_time():
    runner = FakeRunner(TaskResult(True, text="ok"), delay=0.2)
    agent, bb = _agent(runner, allowed_chat_guids=[ALLOWED, OTHER])
    await agent.process(message("/task x", chat=OTHER, guid="u1"))
    assert bb.texts[-1] == (OTHER, "tasks only work in the owner's dm for now") and runner.calls == []
    await agent.process(message("/task first", guid="u2"))
    await agent.process(message("/task second", guid="u3"))
    assert bb.texts[-1][1] == "already working on one, hold on"
    await agent._task_job
    assert [c[0] for c in runner.calls] == ["first"]


async def test_task_failure_and_shutdown_messages():
    agent, bb = _agent(FakeRunner(TaskResult(False, text="partial notes", reason="error_max_turns")))
    await agent.process(message("/task big job", guid="v1"))
    await agent._task_job
    assert bb.texts[-1][1].startswith("couldn't finish that one, it took too many steps. here's what i got: partial notes")
    agent, bb = _agent(FakeRunner(TaskResult(True, text="never"), delay=5))
    await agent.process(message("/task slow job", guid="v2"))
    await asyncio.sleep(0.02)
    await agent.stop_task()
    assert bb.texts[-1][1] == "got restarted mid-task, send that /task again"


async def test_task_off_by_default(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert agent.tasks is None and s.task_enabled is False
    await agent.process(message("/task anything", guid="w1"))
    assert bb.texts[-1][1] == "tasks are off on my end"
