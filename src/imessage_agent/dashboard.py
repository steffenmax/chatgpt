"""Owner dashboard: a page on the loopback port showing what the bot is doing.

Everything here reads the running process' own state. It is reachable only from
this Mac (the server refuses non-loopback binds) and is off unless DASHBOARD_ENABLED.
Memory notes and reminder text are shown because the owner asked to see them; the
event feed is the same redacted, content-free log the file gets.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Settings, is_loopback_host
from .logsafe import exc_name, log_event, recent_lines

_PAGE = Path(__file__).with_name("dashboard.html")
_STARTED = time.time()


def _label(chat: dict | None, guid: str) -> str:
    if chat and chat.get("displayName"):
        return str(chat["displayName"]).strip()
    n = len((chat or {}).get("participants") or [])
    return "Direct chat" if guid.split(";")[1:2] == ["-"] else f"Group chat ({n + 1} people)"


def build_router(settings: Settings) -> APIRouter:
    r = APIRouter(prefix="/dashboard")
    chat_cache: dict[str, dict] = {}

    async def chat_info(ag, guid: str) -> dict:
        if guid not in chat_cache:
            try:
                chat = await ag.bb.get_chat(guid)
            except Exception as exc:  # noqa: BLE001
                log_event("dashboard.chat_lookup_failed", error=exc_name(exc))
                chat = {}
            names = []
            for p in (chat or {}).get("participants") or []:
                n = await ag.resolver.name_for(p.get("address"))
                names.append(n if (n and not n.startswith("+") and "@" not in n) else "(unknown)")
            chat_cache[guid] = {"label": _label(chat, guid), "members": names}
        return chat_cache[guid]

    @r.get("", response_class=HTMLResponse)
    async def page():
        return HTMLResponse(_PAGE.read_text(encoding="utf-8"))

    @r.get("/api/state")
    async def state(request: Request):
        ag = request.app.state.agent
        try:
            bb_ok = await ag.bb.ping()
        except Exception:  # noqa: BLE001
            bb_ok = False
        loaded = []
        ollama_ok = None
        try:
            async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=5) as c:
                ollama_ok = (await c.get("/api/tags")).status_code == 200
                for m in (await c.get("/api/ps")).json().get("models", []):
                    loaded.append({"name": m.get("name"), "gb": round((m.get("size_vram") or m.get("size") or 0) / 1e9, 1),
                                   "until": m.get("expires_at")})
        except Exception:  # noqa: BLE001
            pass
        chats = []
        for i, guid in enumerate(settings.allowed_chat_guids):
            info = await chat_info(ag, guid)
            notes = ag.memory.load(guid) if ag.memory else ""
            chats.append({
                "index": i, "label": info["label"], "members": info["members"],
                "context_messages": len(ag.context.get(guid)),
                "memory_chars": len(notes),
                "reminders": [
                    {"id": x.id, "who": x.who, "text": x.text, "due": x.due} for x in (ag.reminders.pending(guid) if ag.reminders else [])
                ],
            })
        hosts = [{"purpose": p, "host": h, "local": is_loopback_host(h)} for p, h in settings.outbound_hosts()]
        return {
            "bot_name": settings.bot_name,
            "started": _STARTED,
            "bluebubbles": bb_ok,
            "ollama": ollama_ok,
            "loaded_models": loaded,
            "chat_model": settings.ollama_model,
            "vision_model": settings.vision_model or None,
            "image": {"enabled": settings.image_enabled and ag.images is not None, "backend": settings.image_backend,
                      "model": settings.openai_image_model if settings.image_backend == "openai" else settings.mflux_model,
                      "queue": ag.images.qsize() if ag.images else 0},
            "cloud": {"facts": ag.cloud is not None, "all_text": settings.cloud_llm_enabled,
                      "model": settings.anthropic_model, "searches": settings.cloud_facts_max_searches},
            "hosts": hosts,
            "features": {
                "quiet_question_seconds": settings.quiet_question_seconds,
                "reaction_chance": settings.reaction_chance,
                "reaction_cooldown_seconds": settings.reaction_cooldown_seconds,
                "memory": bool(ag.memory), "memory_dir": settings.memory_dir or None,
                "memory_update_every": settings.memory_update_every,
                "reminders": bool(ag.reminders), "reminders_file": settings.reminders_file or None,
                "voice_acks": settings.voice_acks,
                "task": bool(ag.tasks), "task_chats": len(settings.task_chat_guids),
                "task_budget_usd": settings.task_max_budget_usd,
                "context_max_messages": settings.context_max_messages,
                "context_ttl_hours": round(settings.context_ttl_seconds / 3600, 1),
            },
            "chats": chats,
        }

    def _chat_guid(index: int) -> str:
        try:
            return settings.allowed_chat_guids[index]
        except IndexError:
            raise HTTPException(status_code=404, detail="no such chat")

    @r.get("/api/memory/{index}")
    async def get_memory(index: int, request: Request):
        ag = request.app.state.agent
        if ag.memory is None:
            raise HTTPException(status_code=404, detail="memory off")
        guid = _chat_guid(index)
        return {"index": index, "notes": ag.memory.load(guid), "path": str(ag.memory.path_for(guid))}

    @r.put("/api/memory/{index}")
    async def put_memory(index: int, request: Request):
        ag = request.app.state.agent
        if ag.memory is None:
            raise HTTPException(status_code=404, detail="memory off")
        guid = _chat_guid(index)
        body = await request.json()
        notes = str(body.get("notes", ""))[: ag.memory.max_chars * 2]
        ag.memory.save(guid, notes)
        log_event("memory.edited", chat_guid=guid, chars=len(notes))
        return {"ok": True, "chars": len(notes)}

    @r.delete("/api/reminders/{index}/{rid}")
    async def cancel_reminder(index: int, rid: int, request: Request):
        ag = request.app.state.agent
        if ag.reminders is None:
            raise HTTPException(status_code=404, detail="reminders off")
        ok = ag.reminders.cancel(_chat_guid(index), rid)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)

    @r.get("/api/brief")
    async def get_brief(request: Request):
        ag = request.app.state.agent
        if ag.brief is None:
            raise HTTPException(status_code=404, detail="brief off")
        age = ag.brief.age_hours()
        return {"text": ag.brief.raw(), "age_hours": None if age is None else round(age, 1),
                "in_prompt": bool(ag.brief.load()), "topics": ag.brief.topics, "at": ag.brief.at,
                "last_error": ag.brief.last_error, "refreshing": ag.brief._lock.locked()}

    @r.put("/api/brief")
    async def put_brief(request: Request):
        ag = request.app.state.agent
        if ag.brief is None:
            raise HTTPException(status_code=404, detail="brief off")
        body = await request.json()
        text = str(body.get("text", ""))[:6000]
        ag.brief.save(text)
        log_event("brief.edited", chars=len(text))
        return {"ok": True}

    @r.post("/api/brief/refresh")
    async def refresh_brief(request: Request):
        import asyncio

        ag = request.app.state.agent
        if ag.brief is None:
            raise HTTPException(status_code=404, detail="brief off")
        if ag.brief._lock.locked():
            return {"ok": False, "detail": "already refreshing"}
        asyncio.create_task(ag.brief.refresh())
        return {"ok": True, "detail": "refreshing, takes a minute or two"}

    @r.get("/api/events")
    async def events(n: int = 200):
        return {"lines": recent_lines(max(1, min(n, 400)))}

    return r
