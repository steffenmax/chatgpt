"""FastAPI webhook server. Binds to loopback only."""

from __future__ import annotations

import asyncio
import hmac
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .bluebubbles import BlueBubblesClient
from .config import Settings
from .memory import MemoryStore
from .reminders import ReminderStore
from .vision import OllamaVision
from .context import ContextStore
from .egress import print_egress_report
from .handler import Agent, SenderResolver
from .imagegen import OpenAIImageGenerator, ImageQueue, MfluxGenerator
from .llm import AnthropicBackend, OllamaBackend, TextBackend
from .logsafe import exc_name, install_logging, log_event

MAX_BODY_BYTES = 1_000_000


def build_llm(settings: Settings) -> TextBackend:
    if settings.cloud_llm_enabled:
        return AnthropicBackend(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            timeout=settings.anthropic_timeout_seconds,
        )
    return OllamaBackend(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
        timeout=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
    )


def build_cloud(settings: Settings) -> TextBackend | None:
    """Second backend for fact lookups. None unless CLOUD_FACTS_ENABLED (and not full cloud mode)."""
    if settings.cloud_llm_enabled or not settings.cloud_facts_enabled:
        return None
    return AnthropicBackend(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        timeout=settings.anthropic_timeout_seconds,
        web_search=True,
        max_searches=settings.cloud_facts_max_searches,
    )


def create_app(settings: Settings | None = None, agent: Agent | None = None) -> FastAPI:
    """Build the app. Tests pass a pre-built Agent with fake clients."""
    settings = settings or Settings()
    install_logging(
        settings.log_level,
        secrets=[settings.bluebubbles_password, settings.webhook_secret, settings.anthropic_api_key, settings.openai_api_key],
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        fatal, warnings = _split_problems(settings)
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)
        if fatal:
            for f in fatal:
                print(f"FATAL: {f}", file=sys.stderr)
            raise RuntimeError("fatal configuration problems; see FATAL lines above")
        print_egress_report(settings)

        if agent is None:
            app.state.agent = _build_agent(settings)
            app.state.owns_agent = True
        else:
            app.state.agent = agent
            app.state.owns_agent = False
        ag: Agent = app.state.agent
        if ag.images is not None:
            ag.images.start()
        reminder_task = (
            asyncio.create_task(ag.reminders.run(ag.fire_reminder), name="reminders")
            if ag.reminders is not None
            else None
        )
        sweeper = asyncio.create_task(
            ag.context.run_sweeper(
                settings.context_sweep_seconds,
                on_sweep=lambda n: log_event("context.swept", count=n, ttl_seconds=settings.context_ttl_seconds),
            )
        )
        log_event("startup", host=f"{settings.host}:{settings.port}", backend=ag.llm.name)
        try:
            yield
        finally:
            sweeper.cancel()
            if reminder_task is not None:
                reminder_task.cancel()
            if ag.images is not None:
                await ag.images.stop()
            await ag.drain()
            ag.context.forget_all()  # context never survives a restart
            if app.state.owns_agent:
                await ag.bb.aclose()
                await ag.llm.aclose()
            log_event("shutdown")

    app = FastAPI(title="imessage-agent", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings

    def _check_secret(secret_q: str | None, secret_h: str | None) -> None:
        provided = secret_q or secret_h or ""
        if not provided or not hmac.compare_digest(provided.encode(), settings.webhook_secret.encode()):
            log_event("webhook.rejected", reason="bad_secret")
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post(settings.webhook_path)
    async def webhook(
        request: Request,
        secret: str | None = Query(default=None),
        x_webhook_secret: str | None = Header(default=None),
    ):
        _check_secret(secret, x_webhook_secret)
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="too large")
        try:
            import json

            payload: Any = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=400, detail="bad json")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="bad payload")
        etype = payload.get("type")
        log_event("webhook.received", kind=str(etype)[:40] if etype else "unknown")
        request.app.state.agent.schedule(payload)
        return JSONResponse({"ok": True})

    if settings.dashboard_enabled:
        from .dashboard import build_router

        app.include_router(build_router(settings))

    @app.get("/health")
    async def health():
        ag: Agent = app.state.agent
        ollama_ok: bool | None = None
        if isinstance(ag.llm, OllamaBackend):
            ollama_ok = await ag.llm.healthy()
        try:
            bb_ok = await ag.bb.ping()
        except Exception as exc:  # noqa: BLE001
            log_event("health.bluebubbles_error", error=exc_name(exc))
            bb_ok = False
        return {
            "status": "ok" if bb_ok and (ollama_ok is not False) else "degraded",
            "bluebubbles": bb_ok,
            "ollama": ollama_ok,
            "llm_backend": ag.llm.name,
            "cloud_llm_enabled": settings.cloud_llm_enabled,
            "cloud_facts_enabled": ag.cloud is not None,
            "memory_enabled": ag.memory is not None,
            "reminders_enabled": ag.reminders is not None,
            "vision_enabled": ag.vision is not None,
            "image_backend": settings.image_backend if ag.images else None,
            "image_queue": ag.images.qsize() if ag.images else None,
            "chats_with_context": ag.context.chat_count(),
            "allowlisted_chats": len(settings.allowed_chat_guids),
        }

    return app


def _split_problems(settings: Settings) -> tuple[list[str], list[str]]:
    problems = settings.validate_runtime()
    warnings = [p for p in problems if p.startswith("ALLOWED_CHAT_GUIDS")]
    fatal = [p for p in problems if p not in warnings]
    return fatal, warnings


def _build_agent(settings: Settings) -> Agent:
    bb = BlueBubblesClient(
        settings.bluebubbles_url,
        settings.bluebubbles_password,
        send_method=settings.bluebubbles_send_method,
        timeout=settings.bluebubbles_timeout_seconds,
    )
    llm = build_llm(settings)
    context = ContextStore(max_messages=settings.context_max_messages, ttl_seconds=settings.context_ttl_seconds)
    images: ImageQueue | None = None
    llm_gate = asyncio.Semaphore(1)
    agent_holder: dict[str, Agent] = {}

    if settings.image_enabled:
        gen: MfluxGenerator | OpenAIImageGenerator
        if settings.image_backend == "openai":
            gen = OpenAIImageGenerator(
                api_key=settings.openai_api_key,
                model=settings.openai_image_model,
                size=settings.openai_image_size,
                quality=settings.openai_image_quality,
                timeout=settings.openai_timeout_seconds,
            )
        else:
            gen = MfluxGenerator(
                command=settings.mflux_command,
                model=settings.mflux_model,
                steps=settings.mflux_steps,
                quantize=settings.mflux_quantize,
                width=settings.mflux_width,
                height=settings.mflux_height,
                extra_args=settings.mflux_extra_args,
                timeout=settings.image_timeout_seconds,
            )
        if not gen.available():
            print(
                f"WARNING: IMAGE_ENABLED=true but image backend `{settings.image_backend}` is not usable "
                f"({settings.mflux_command} not on PATH, or OPENAI_API_KEY empty); /image will answer that images are off.",
                file=sys.stderr,
            )
        else:
            async def _send(job, png):
                await agent_holder["a"].send_image(job, png)

            async def _fail(job, reason):
                await agent_holder["a"].image_failed(job, reason)

            before = None
            # Only the local generator competes with the LLM for RAM.
            if settings.unload_llm_before_image and isinstance(llm, OllamaBackend) and isinstance(gen, MfluxGenerator):
                before = llm.unload
            images = ImageQueue(gen, _send, _fail, before_generate=before, gate=llm_gate)

    agent = Agent(
        settings, bb, llm, context, images,
        resolver=SenderResolver(bb, settings.sender_names),
        llm_gate=llm_gate,
        cloud=build_cloud(settings),
        memory=MemoryStore(settings.memory_dir, settings.memory_max_chars, settings.memory_update_every)
        if settings.memory_dir
        else None,
        reminders=ReminderStore(settings.reminders_file) if settings.reminders_file else None,
        vision=OllamaVision(settings.ollama_url, settings.vision_model, settings.vision_max_width, settings.vision_keep_alive)
        if settings.vision_model
        else None,
    )
    agent_holder["a"] = agent
    return agent


def run() -> None:
    import uvicorn

    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning", access_log=False)


app = None  # built lazily by `imessage-agent serve`; avoids import-time env requirements
