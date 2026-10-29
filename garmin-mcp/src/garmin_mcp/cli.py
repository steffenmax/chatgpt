"""`garmin-mcp` command: sign in once, then run the MCP server."""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys

from garminconnect import Garmin

from . import __version__
from .config import Settings
from .garmin import GarminError, GarminSession, interactive_login


def cmd_login(settings: Settings, args: argparse.Namespace) -> int:
    token_file = settings.token_dir / "garmin_tokens.json"
    if token_file.exists() and not args.force:
        print(f"Tokens already saved at {token_file}. Use --force to sign in again.")
        return cmd_status(settings, args)
    if args.force:
        Garmin().logout(tokenstore=str(settings.token_dir))

    email = settings.email or input("Garmin email: ").strip()
    password = settings.password or getpass.getpass("Garmin password: ")
    try:
        api = interactive_login(
            settings, email, password, prompt_mfa=lambda: input("MFA code (from email/app): ").strip()
        )
    except Exception as e:  # noqa: BLE001 - surface any login failure as a message
        print(f"Login failed: {e}", file=sys.stderr)
        return 1
    print(f"Signed in as {api.get_full_name() or api.display_name}. Tokens saved to {token_file}")
    return 0


def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    session = GarminSession(settings)
    try:
        name = session.call_sync(lambda api: api.get_full_name() or api.display_name)
    except GarminError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"OK: signed in as {name} (tokens in {settings.token_dir})")
    print(f"Write tools: {'enabled' if settings.allow_writes else 'disabled'}")
    return 0


def cmd_logout(settings: Settings, args: argparse.Namespace) -> int:
    Garmin().logout(tokenstore=str(settings.token_dir))
    print(f"Removed saved tokens from {settings.token_dir}")
    return 0


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    from .server import build_server

    mcp = build_server(GarminSession(settings), settings)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="garmin-mcp", description="MCP server for Garmin Connect")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("login", help="sign in interactively (supports MFA) and save tokens")
    p.add_argument("--force", action="store_true", help="discard saved tokens and sign in again")
    sub.add_parser("status", help="check that saved tokens work")
    sub.add_parser("logout", help="delete saved tokens")
    p = sub.add_parser("serve", help="run the MCP server (default)")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["serve", *(argv or [])])

    # stdout is the MCP channel under stdio, so logs always go to stderr.
    logging.basicConfig(
        level=os.getenv("GARMIN_MCP_LOG_LEVEL", "WARNING").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    handlers = {"login": cmd_login, "status": cmd_status, "logout": cmd_logout, "serve": cmd_serve}
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
