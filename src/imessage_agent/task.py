"""/task: hand a multi-step job to Claude (Agent SDK) with web tools and a sandbox folder.

Opt-in and owner-only. What the agent can touch is fenced three ways:
  * `tools` limits the toolset to web search, web fetch and file tools (no shell,
    no subagents, no MCP servers, none of the owner's Claude Code settings);
  * a PreToolUse hook denies any file access outside the task's sandbox folder;
  * it runs with its own CLAUDE_CONFIG_DIR and the API key, never the owner's login.
The request and recent chat context are sent to Anthropic; WebFetch downloads pages
from this Mac. Session transcripts are deleted after each task.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from .logsafe import exc_name, log_event

FILE_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep")
WEB_TOOLS = ("WebSearch", "WebFetch")
TOOLS = [*WEB_TOOLS, *FILE_TOOLS]
KEEP_DAYS = 14

OPERATOR_PROMPT = (
    "You are a careful research assistant working for a group of friends who asked you a task "
    "from their group chat. Use web search and web fetch to do it properly: check a few sources, "
    "prefer specific, current facts (names, prices, times, addresses), and don't guess.\n"
    "You can save notes or a document as files in the current directory, and nowhere else. You "
    "cannot run commands, send messages, log in, book, buy, or contact anyone; if the task needs "
    "that, do the research and say exactly what the person should do.\n"
    "Your final answer is texted to the group as-is: plain text, no markdown headings, tables or "
    "bold, short paragraphs or simple dashes, casual friendly tone, under 900 characters. Include "
    "at most 3 links, and only ones you actually opened. If you wrote a longer document, say its "
    "filename in one line; it will be attached."
)


@dataclass
class TaskResult:
    ok: bool
    text: str = ""
    files: list[Path] = field(default_factory=list)
    cost_usd: float = 0.0
    turns: int = 0
    duration_ms: int = 0
    reason: str = ""  # short machine reason when not ok


def _inside(path: str, root: Path) -> bool:
    try:
        p = Path(os.path.expanduser(path))
        if not p.is_absolute():
            return False if ".." in p.parts else True  # relative to cwd, which is inside root
        return p.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def guard_decision(tool_name: str, tool_input: dict[str, Any], root: Path) -> str | None:
    """None to allow, or a reason string to deny."""
    if tool_name in WEB_TOOLS:
        return None
    if tool_name in FILE_TOOLS:
        for key in ("file_path", "path", "notebook_path"):
            v = tool_input.get(key)
            if isinstance(v, str) and v and not _inside(v, root):
                return f"{tool_name} outside the task folder is not allowed"
        pat = tool_input.get("pattern")
        if tool_name == "Glob" and isinstance(pat, str) and (pat.startswith(("/", "~")) or ".." in pat):
            return "Glob outside the task folder is not allowed"
        return None
    return f"{tool_name} is not available for tasks"


class TaskRunner:
    def __init__(
        self,
        api_key: str,
        model: str,
        root: str,
        max_turns: int = 25,
        max_budget_usd: float = 1.0,
        timeout: float = 600.0,
        query_fn: Callable[..., AsyncIterator[Any]] | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.root = Path(root).expanduser()
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd
        self.timeout = timeout
        self._query = query_fn  # injected in tests

    def _prepare(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        cutoff = time.time() - KEEP_DAYS * 86400
        for old in self.root.glob("task-*"):
            try:
                if old.is_dir() and old.stat().st_mtime < cutoff:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
        work = self.root / f"task-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        work.mkdir()
        os.chmod(work, 0o700)
        return work

    def _options(self, work: Path):
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        root = self.root

        async def guard(input_data, tool_use_id, context):
            reason = guard_decision(input_data.get("tool_name", ""), input_data.get("tool_input") or {}, root)
            if reason is None:
                return {}
            log_event("task.tool_denied", reason=input_data.get("tool_name", "?")[:40])
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}

        config_dir = root / ".claude-config"
        config_dir.mkdir(exist_ok=True)
        os.chmod(config_dir, 0o700)
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=OPERATOR_PROMPT,
            tools=list(TOOLS),
            allowed_tools=list(TOOLS),
            permission_mode="dontAsk",
            setting_sources=[],
            mcp_servers={},
            strict_mcp_config=True,
            cwd=str(work),
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[guard])]},
            env={
                "ANTHROPIC_API_KEY": self.api_key,
                "CLAUDE_CONFIG_DIR": str(config_dir),
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
        )

    async def run(self, request: str, requester: str, context_lines: list[str]) -> TaskResult:
        started = time.monotonic()
        work = self._prepare()
        prompt = (
            (("Recent group chat, for context:\n" + "\n".join(context_lines) + "\n\n") if context_lines else "")
            + f"Task from {requester}: {request}"
        )
        result: Any = None
        try:
            if self._query is None:
                from claude_agent_sdk import query as sdk_query

                query_fn, options = sdk_query, self._options(work)
            else:
                query_fn, options = self._query, {"cwd": str(work), "root": str(self.root)}

            async def consume():
                nonlocal result
                async for msg in query_fn(prompt=prompt, options=options):
                    if type(msg).__name__ == "ResultMessage":
                        result = msg

            await asyncio.wait_for(consume(), timeout=self.timeout)
        except asyncio.TimeoutError:
            return TaskResult(False, reason="timeout", duration_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:  # noqa: BLE001
            log_event("task.error", error=exc_name(exc))
            return TaskResult(False, reason=exc_name(exc), duration_ms=int((time.monotonic() - started) * 1000))
        finally:
            shutil.rmtree(self.root / ".claude-config" / "projects", ignore_errors=True)  # session transcripts
        dur = int((time.monotonic() - started) * 1000)
        if result is None:
            return TaskResult(False, reason="no_result", duration_ms=dur)
        cost = float(getattr(result, "total_cost_usd", 0) or 0)
        turns = int(getattr(result, "num_turns", 0) or 0)
        subtype = str(getattr(result, "subtype", ""))
        text = (getattr(result, "result", None) or "").strip()
        files = sorted(p for p in work.iterdir() if p.is_file() and not p.name.startswith(".") and p.stat().st_size < 2_000_000)
        if subtype != "success" or getattr(result, "is_error", False) or not text:
            return TaskResult(False, text=text, files=files, cost_usd=cost, turns=turns, duration_ms=dur, reason=subtype or "error")
        return TaskResult(True, text=text, files=files, cost_usd=cost, turns=turns, duration_ms=dur)
