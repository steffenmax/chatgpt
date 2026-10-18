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
        if not is_loopback_host(host) and not purpose.startswith("Anthropic API"):
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

    if s.image_enabled:
        path = shutil.which(s.mflux_command)
        print(f"mflux: {'found at ' + path if path else 'NOT found (uv tool install mflux); /image will be disabled'}")
    else:
        print("mflux: image generation disabled")
    print("\nRESULT:", "OK" if ok else "problems found")
    return 0 if ok else 1


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
    }[args.cmd]
    try:
        return asyncio.run(fn(s, args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
