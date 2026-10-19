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
    elif settings.cloud_facts_enabled:
        lines += [
            "",
            "=" * 72,
            "  WARNING: CLOUD_FACTS_ENABLED=true",
            "  When a mention needs outside/current facts (or someone uses /ask), the",
            "  triggering message AND the rolling chat context are sent to api.anthropic.com",
            f"  (model {settings.anthropic_model}, web search on). Other replies stay local.",
            "  Set CLOUD_FACTS_ENABLED=false to keep everything on this Mac.",
            "=" * 72,
        ]
    else:
        lines.append("  Cloud text mode is OFF. No message content leaves this machine.")
    if settings.task_enabled:
        lines += [
            "",
            "=" * 72,
            "  WARNING: TASK_ENABLED=true",
            f"  /task (from {len(settings.task_chat_guids)} chat(s) only) sends the request and up to",
            f"  {settings.task_context_messages} recent messages to api.anthropic.com, and the agent fetches web pages",
            "  from this Mac. It has no shell and can only write inside its task folder.",
            "=" * 72,
        ]
    if settings.image_enabled and settings.image_backend == "openai":
        lines += [
            "",
            "=" * 72,
            "  WARNING: IMAGE_BACKEND=openai",
            f"  Every /image prompt is sent to api.openai.com (model {settings.openai_image_model}).",
            "  Set IMAGE_BACKEND=mflux to generate images on this Mac instead.",
            "=" * 72,
        ]
    return lines


def print_egress_report(settings: Settings) -> None:
    for line in egress_report(settings):
        print(line, file=sys.stderr, flush=True)
