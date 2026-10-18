"""Startup transparency: print every outbound host the app is configured to call."""

from __future__ import annotations

import sys

from .config import Settings, is_loopback_host


def egress_report(settings: Settings) -> list[str]:
    lines = ["Outbound hosts this process is configured to contact:"]
    for purpose, host in settings.outbound_hosts():
        tag = "local" if is_loopback_host(host) else "REMOTE"
        lines.append(f"  [{tag:6}] {purpose:<34} {host}")
    lines.append(
        "  (one-time model downloads happen outside this process: `ollama pull` -> ollama.com, "
        "mflux -> huggingface.co; they carry no chat content)"
    )
    if settings.cloud_llm_enabled:
        lines += [
            "",
            "=" * 72,
            "  WARNING: CLOUD_LLM_ENABLED=true",
            "  Text of triggering messages AND the rolling chat context will be sent to",
            f"  api.anthropic.com (model {settings.anthropic_model}). Images stay local.",
            "  Set CLOUD_LLM_ENABLED=false to keep everything on this Mac.",
            "=" * 72,
        ]
    else:
        lines.append("  Cloud text mode is OFF. No message content leaves this machine.")
    return lines


def print_egress_report(settings: Settings) -> None:
    for line in egress_report(settings):
        print(line, file=sys.stderr, flush=True)
