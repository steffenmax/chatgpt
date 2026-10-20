from fastapi.testclient import TestClient

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.main import create_app
from imessage_agent.memory import MemoryStore
from imessage_agent.reminders import ReminderStore
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message


def _app(tmp_path, **over):
    s = make_settings(dashboard_enabled=True, memory_dir=str(tmp_path / "mem"), reminders_file=str(tmp_path / "r.json"), **over)
    bb, llm = FakeBB(), FakeLLM()
    ctx = ContextStore(5, 3600)
    mem = MemoryStore(s.memory_dir, 3000, 15)
    rem = ReminderStore(s.reminders_file)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {"+15550001111": "Alex"}, use_contacts=False), memory=mem, reminders=rem)  # type: ignore[arg-type]
    return s, bb, agent, mem, rem, create_app(s, agent)


def test_dashboard_off_by_default(tmp_path):
    s = make_settings()
    assert s.dashboard_enabled is False
    agent = Agent(s, FakeBB(), FakeLLM(), ContextStore(5, 3600), images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    with TestClient(create_app(s, agent)) as c:
        assert c.get("/dashboard").status_code == 404
        assert c.get("/dashboard/api/state").status_code == 404


async def test_dashboard_state_memory_reminders_and_events(tmp_path):
    s, bb, agent, mem, rem, app = _app(tmp_path)
    mem.save(ALLOWED, "## People\n- Alex: dry\n")
    r1 = rem.add(ALLOWED, "Alex", "call mom", due=4102444800.0)
    with TestClient(app) as c:
        page = c.get("/dashboard")
        assert page.status_code == 200 and "dashboard" in page.text.lower() and "/dashboard/api/state" in page.text
        st = c.get("/dashboard/api/state").json()
        assert st["bot_name"] == "Coco" and st["bluebubbles"] is True
        assert [h["purpose"] for h in st["hosts"]][:2] == ["BlueBubbles", "Ollama"] and all(h["local"] for h in st["hosts"])
        assert st["chats"][0]["memory_chars"] > 0 and st["chats"][0]["reminders"][0]["text"] == "call mom"
        assert st["features"]["memory"] and st["features"]["reminders"]
        # memory read + edit
        m = c.get("/dashboard/api/memory/0").json()
        assert "Alex: dry" in m["notes"] and m["path"].endswith(".md")
        assert c.put("/dashboard/api/memory/0", json={"notes": "## People\n- Alex: dry, punctual\n"}).json()["ok"]
        assert "punctual" in mem.load(ALLOWED)
        assert c.get("/dashboard/api/memory/9").status_code == 404
        # reminders cancel
        assert c.delete(f"/dashboard/api/reminders/0/{r1.id}").json()["ok"] and rem.pending(ALLOWED) == []
        assert c.delete("/dashboard/api/reminders/0/999").status_code == 404
        # events feed carries the redacted log, never message text
        c.post(f"/webhook?secret={s.webhook_secret}", json=message("coco secret banana text", guid="d1"))
        lines = c.get("/dashboard/api/events?n=50").json()["lines"]
        assert any("event=trigger" in l for l in lines)
        joined = "\n".join(lines)
        assert "banana" not in joined and s.webhook_secret not in joined and s.bluebubbles_password not in joined


def test_pulled_posts_endpoint(tmp_path):
    from types import SimpleNamespace

    s, bb, agent, mem, rem, app = _app(tmp_path)
    with TestClient(app) as c:
        assert c.get("/dashboard/api/brief/posts").status_code == 404  # brief off
        agent.brief = SimpleNamespace(x=None)
        assert c.get("/dashboard/api/brief/posts").status_code == 404  # brief without X
        agent.brief = SimpleNamespace(x=SimpleNamespace(last_posts=lambda: {}))
        assert c.get("/dashboard/api/brief/posts").json() == {"at": None, "searches": []}
        saved = {"at": "2026-10-20T13:42:00", "searches": [{"query": "NBA", "returned": 1, "posts": [{"id": "1", "text": "x"}]}]}
        agent.brief = SimpleNamespace(x=SimpleNamespace(last_posts=lambda: saved))
        assert c.get("/dashboard/api/brief/posts").json() == saved
        page = c.get("/dashboard").text
        assert 'id="xpsec"' in page and "/dashboard/api/brief/posts" in page


def test_persona_panel_reads_and_saves(tmp_path, monkeypatch):
    import dotenv

    from imessage_agent import dashboard
    from imessage_agent.main import create_app

    monkeypatch.setattr(dashboard, "PERSONA_BACKUPS", tmp_path / "persona-backups")

    s, bb, agent, mem, rem, _ = _app(tmp_path)
    env = tmp_path / ".env"
    env.write_text("BOT_NAME=Coco\nBOT_PERSONA=old persona text for {bot_name}\nPORT=1\n")
    app = create_app(s, agent, env_path=env)
    new = 'You are {bot_name}. He says "yo" a lot,\nit\'s his thing # not a comment'
    before = agent.s.bot_persona
    with TestClient(app) as c:
        assert c.get("/dashboard/api/persona").json()["saves_to_env"] is True
        assert c.put("/dashboard/api/persona", json={"persona": "short"}).status_code == 400
        r = c.put("/dashboard/api/persona", json={"persona": new})
        assert r.status_code == 200 and r.json()["saved_to_env"] is True
        assert agent.s.persona == new.replace("{bot_name}", "Coco")  # live, no restart
        assert dotenv.dotenv_values(env)["BOT_PERSONA"] == new and dotenv.dotenv_values(env)["PORT"] == "1"
        assert c.get("/dashboard/api/persona").json()["persona"] == new
        assert 'id="personatext"' in c.get("/dashboard").text
    backups = list((tmp_path / "persona-backups").glob("persona-*.txt"))
    assert len(backups) == 1 and backups[0].read_text() == before
    assert backups[0].stat().st_mode & 0o777 == 0o600


def test_persona_not_written_without_an_env_path(tmp_path):
    s, bb, agent, mem, rem, app = _app(tmp_path)
    with TestClient(app) as c:
        r = c.put("/dashboard/api/persona", json={"persona": "You are {bot_name}, a test persona."})
        assert r.json()["saved_to_env"] is False and agent.s.bot_persona == "You are {bot_name}, a test persona."
