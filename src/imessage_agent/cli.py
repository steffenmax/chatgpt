"""Command-line entry point: imessage-agent <command>."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys

import httpx

from . import __version__
from .bluebubbles import BlueBubblesClient
from .config import Settings, is_loopback_host
from .egress import egress_report
from .llm import OllamaBackend


def _bb(s: Settings) -> BlueBubblesClient:
    return BlueBubblesClient(s.bluebubbles_url, s.bluebubbles_password, s.bluebubbles_send_method, s.bluebubbles_timeout_seconds)


async def cmd_send(s: Settings, args) -> int:
    bb = _bb(s)
    try:
        await bb.send_text(args.chat, args.text)
        print("sent")
        return 0
    finally:
        await bb.aclose()


async def cmd_chats(s: Settings, args) -> int:
    bb = _bb(s)
    try:
        chats = await bb.list_chats(limit=args.limit)
    finally:
        await bb.aclose()
    allowed = set(s.allowed_chat_guids)
    print(f"{'allowed':7}  {'participants':12}  {'name':30}  guid")
    for c in chats:
        guid = c.get("guid", "")
        name = (c.get("displayName") or "").strip() or "(no name)"
        parts = c.get("participants") or []
        mark = "yes" if guid in allowed else ""
        print(f"{mark:7}  {len(parts):<12}  {name[:30]:30}  {guid}")
    print("\nGroup chats have GUIDs like iMessage;+;chat123456789. Add them to ALLOWED_CHAT_GUIDS.")
    return 0


async def cmd_register_webhook(s: Settings, args) -> int:
    url = f"http://{s.host}:{s.port}{s.webhook_path}?secret={s.webhook_secret}"
    bb = _bb(s)
    try:
        existing = await bb.list_webhooks()
        for w in existing:
            if w.get("url") == url:
                print(f"webhook already registered (id={w.get('id')})")
                return 0
        stale = [w for w in existing if str(w.get("url", "")).startswith(f"http://{s.host}:{s.port}{s.webhook_path}")]
        for w in stale:
            if args.replace:
                await bb.delete_webhook(int(w["id"]))
                print(f"deleted stale webhook id={w.get('id')} (old secret/port)")
            else:
                print(f"note: another webhook for this path exists (id={w.get('id')}); use --replace to remove it")
        created = await bb.create_webhook(url, ["new-message"])
        print(f"registered webhook id={created.get('id')} for new-message -> {s.host}:{s.port}{s.webhook_path}")
        return 0
    finally:
        await bb.aclose()


async def cmd_check(s: Settings, args) -> int:
    ok = True
    for line in egress_report(s):
        print(line)
    for purpose, host in s.outbound_hosts():
        if not is_loopback_host(host) and not purpose.startswith(("Anthropic API", "OpenAI API", "X API", "Web pages fetched")):
            print(f"!! {purpose} is not on loopback ({host}); message content would leave this Mac.")
            ok = False
    print()
    for p in s.validate_runtime():
        print(f"config: {p}")
        if not p.startswith("ALLOWED_CHAT_GUIDS"):
            ok = False

    bb = _bb(s)
    try:
        info = await bb.server_info()
        print(f"BlueBubbles: reachable, server {info.get('server_version')} on macOS {info.get('os_version')}, "
              f"private_api={info.get('private_api')}, imessage={info.get('detected_imessage')}")
    except Exception as e:  # noqa: BLE001
        print(f"BlueBubbles: NOT reachable ({type(e).__name__}). Is BlueBubbles Server running? Password correct?")
        ok = False
    finally:
        await bb.aclose()

    if s.cloud_llm_enabled:
        print("Ollama: skipped (cloud text mode on)")
    else:
        ol = OllamaBackend(s.ollama_url, s.ollama_model)
        try:
            if await ol.healthy():
                has = await ol.has_model()
                print(f"Ollama: reachable, model {s.ollama_model} {'present' if has else 'MISSING - run: ollama pull ' + s.ollama_model}")
                ok = ok and bool(has)
            else:
                print("Ollama: NOT reachable. Start the Ollama app or `ollama serve`.")
                ok = False
        finally:
            await ol.aclose()

    if s.vision_model:
        from .vision import OllamaVision

        v = OllamaVision(s.ollama_url, s.vision_model)
        try:
            present = await v.has_model()
        finally:
            await v.aclose()
        print(f"vision: model {s.vision_model} {'present' if present else 'NOT pulled (ollama pull ' + s.vision_model + ')'}")
        ok = ok and bool(present)
    if s.image_enabled:
        if s.image_backend == "openai":
            from .imagegen import OpenAIImageGenerator

            gen = OpenAIImageGenerator(s.openai_api_key, s.openai_image_model, timeout=15)
            try:
                good = bool(s.openai_api_key) and await gen.check()
            finally:
                await gen.aclose()
            print(f"OpenAI images: {'key OK, model ' + s.openai_image_model if good else 'NOT usable (key missing/invalid or model unavailable)'}")
            ok = ok and good
        else:
            path = shutil.which(s.mflux_command)
            print(f"mflux: {'found at ' + path if path else 'NOT found (uv tool install mflux); /image will be disabled'}")
    else:
        print("mflux: image generation disabled")
    if s.brief_enabled:
        print(_brief_plan_line(s))
    print("\nRESULT:", "OK" if ok else "problems found")
    return 0 if ok else 1


def _brief_plan_line(s: Settings) -> str:
    src = s.resolved_brief_source
    if src == "web":
        return f"brief: source=web (task agent web search, up to ${s.brief_max_budget_usd:.2f} per refresh)"
    from .xsearch import POST_READ_USD, XSearch

    x = XSearch(s.x_bearer_token, s.x_query_list(), s.x_posts_per_query, s.x_lookback_hours, s.x_max_usd_per_refresh)
    plan = x.plan()
    per = x.estimate_usd()
    writer = "cloud model" if s.anthropic_api_key else f"local model {s.ollama_model}"
    how = f"written by the {writer}" if src == "x" else f"+ task agent web search (up to ${s.brief_max_budget_usd:.2f})"
    dropped = len(x.queries) - len(plan)
    note = f"; {dropped} quer{'y' if dropped == 1 else 'ies'} dropped to fit X_MAX_USD_PER_REFRESH" if dropped else ""
    return (f"brief: source={src}, {how}. X: {len(plan)} queries × {plan[0][1] if plan else 0} posts "
            f"≈ ${per:.2f} per refresh (${POST_READ_USD}/post), ≈ ${per * 30:.2f}/month daily{note}")


async def cmd_brief(s: Settings, args) -> int:
    """Plan, pull or fully refresh the brief from the command line."""
    if not s.brief_enabled:
        print("BRIEF_ENABLED is false; nothing to do.")
        return 1
    src = s.resolved_brief_source
    print(_brief_plan_line(s))
    if src != "web":
        from .xsearch import XSearch, full_query

        x = XSearch(s.x_bearer_token, s.x_query_list(), s.x_posts_per_query, s.x_lookback_hours, s.x_max_usd_per_refresh)
        for q, n in x.plan():
            print(f"  {n:3d} posts  {full_query(q)}")
        await x.aclose()
    if args.plan:
        return 0
    from .main import build_brief

    brief = build_brief(s)
    if brief is None:
        print("brief: not configured (see `imessage-agent check`).")
        return 1
    try:
        if args.pull:
            if brief.x is None:
                print("brief: source=web has nothing to pull.")
                return 1
            res = await brief.x.pull()
            print(f"\npulled {res.posts} posts from {res.queries_ok} queries ({res.queries_failed} failed"
                  f"{': ' + ', '.join(res.errors) if res.errors else ''}), est. ${res.cost_usd:.2f}, {res.duration_ms} ms\n")
            print(res.digest or "(no posts)")
            return 0 if res.posts else 1
        print("\nrefreshing…")
        ok = await brief.refresh()
        if ok:
            print(f"updated {brief.path} (est. ${brief.last_cost_usd:.2f})\n")
            print(brief.raw())
            return 0
        print(f"failed: {brief.last_error}")
        return 1
    finally:
        await brief.aclose()


async def cmd_health(s: Settings, args) -> int:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"http://{s.host}:{s.port}/health")
        print(json.dumps(r.json(), indent=2))
        return 0 if r.status_code == 200 else 1


def cmd_serve(s: Settings, args) -> int:
    from .main import run

    run()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="imessage-agent", description="Private iMessage group-chat agent")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the webhook server (foreground)")
    sp = sub.add_parser("send", help="send a test message to a chat GUID")
    sp.add_argument("--chat", required=True, help="chat GUID, e.g. iMessage;+;chat123")
    sp.add_argument("text")
    cp = sub.add_parser("chats", help="list chats BlueBubbles knows about (to find GUIDs)")
    cp.add_argument("--limit", type=int, default=100)
    wp = sub.add_parser("register-webhook", help="register this server as a BlueBubbles webhook")
    wp.add_argument("--replace", action="store_true", help="delete other webhooks pointing at this path first")
    sub.add_parser("check", help="print outbound hosts and test local services")
    sub.add_parser("health", help="query the running server's /health")
    bp = sub.add_parser("brief", help="refresh the daily culture brief now (writes BRIEF_FILE)")
    bp.add_argument("--plan", action="store_true", help="only print the X queries and cost estimate; no network")
    bp.add_argument("--pull", action="store_true", help="pull the X posts and print the digest; don't write the brief")

    args = p.parse_args(argv)
    s = Settings()
    if args.cmd == "serve":
        return cmd_serve(s, args)
    fn = {
        "send": cmd_send,
        "chats": cmd_chats,
        "register-webhook": cmd_register_webhook,
        "check": cmd_check,
        "health": cmd_health,
        "brief": cmd_brief,
    }[args.cmd]
    try:
        return asyncio.run(fn(s, args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
