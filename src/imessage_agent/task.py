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
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlparse

from .logsafe import exc_name, log_event
from .taskbrowser import ReadOnlyBrowser, build_server

FILE_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep")
WEB_TOOLS = ("WebSearch", "WebFetch")
TOOLS = [*WEB_TOOLS, *FILE_TOOLS]
MCP_PREFIX = "mcp__task__"
MCP_TOOLS = [f"{MCP_PREFIX}browser_open", f"{MCP_PREFIX}browser_follow", f"{MCP_PREFIX}ask_chat"]
KEEP_DAYS = 14

OPERATOR_PROMPT = (
    "You are a careful research assistant working for a group of friends who asked you a task "
    "from their group chat. Use web search and web fetch to do it properly: check a few sources, "
    "prefer specific, current facts (names, prices, times, addresses), and don't guess.\n"
    "You can save notes or a document as files in the current directory, and nowhere else. You "
    "cannot run commands, send messages, log in, book, buy, or contact anyone; if the task needs "
    "that, do the research and say exactly what the person should do.\n"
    "If you have browser tools, use them for pages WebFetch can't render (JavaScript-heavy sites). "
    "Resy pages render in the browser and can show open times for a date and party size "
    "(resy.com/cities/<city>/search?query=...&date=YYYY-MM-DD&seats=N); OpenTable and Yelp block it. "
    "Booking and review sites (OpenTable, Resy, Yelp and the like) often block automated access: "
    "try a site at most once, don't keep retrying, and get the facts from sources that load "
    "(search results, the place's own website, news and guides). For anything bookable, give a "
    "direct link, prefilled with date, time and party size when the site's URL supports it, so "
    "they can check availability and book themselves.\n"
    "If you have the ask_chat tool and a detail you truly need is missing (date, time, party size, "
    "budget, which option), ask one short question. Never ask for passwords, logins, codes or "
    "payment details; those are always the person's job.\n"
    "Your final answer is texted to the group as-is: plain text, no markdown headings, tables or "
    "bold, short paragraphs or simple dashes, under 900 characters. Include at most 3 links, and "
    "only ones you actually opened. If you wrote a longer document, say its filename in one line; "
    "it will be attached."
)

VOICE_GUIDE = (
    "\n\nThe final answer is posted by {bot_name}, the group's friend in the chat, as his own message. "
    "Write it AS {bot_name}, first person, in his voice: his attitude, humor, slang and the way he talks "
    "to these people. Here is how {bot_name} is described:\n<<<\n{persona}\n>>>\n"
    "Answer what they actually asked in plain terms (a yes/no gets a yes or no), then let him be "
    "himself: a joke, a jab at whoever asked, or a bit of attitude is expected, not optional. "
    "Use that for tone and delivery only. The facts must stay exactly right: every name, address, time, "
    "price, phone number and link precise and complete, nothing invented. Don't mention being an "
    "assistant, a model or 'research tools'; if something couldn't be checked, say so in his words. "
    "Your questions via ask_chat can be in his voice too."
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


def _local_url(url: str) -> bool:
    import ipaddress

    host = (urlparse(url if "://" in url else "https://" + url).hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
        return True
    try:
        a = ipaddress.ip_address(host)
        return a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_unspecified
    except ValueError:
        return False


def guard_decision(tool_name: str, tool_input: dict[str, Any], root: Path) -> str | None:
    """None to allow, or a reason string to deny."""
    if tool_name == "WebFetch":
        if _local_url(str(tool_input.get("url", ""))):
            return "local and private network addresses are off limits"
        return None
    if tool_name in WEB_TOOLS:
        return None
    if tool_name in MCP_TOOLS:
        return None  # the browser and ask tools enforce their own limits
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
        browser: bool = False,
        max_questions: int = 2,
        ask_timeout: float = 240.0,
        bot_name: str = "",
        persona: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self.root = Path(root).expanduser()
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd
        self.timeout = timeout
        self._query = query_fn  # injected in tests
        self.use_browser = browser
        self.max_questions = max_questions
        self.ask_timeout = ask_timeout
        self.system_prompt = OPERATOR_PROMPT + (
            VOICE_GUIDE.replace("{bot_name}", bot_name).replace("{persona}", persona.strip()[:4000])
            if bot_name and persona.strip() else ""
        )

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

    def _options(self, work: Path, server=None):
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
            system_prompt=self.system_prompt,
            tools=list(TOOLS),
            allowed_tools=list(TOOLS) + (list(MCP_TOOLS) if server is not None else []),
            permission_mode="dontAsk",
            setting_sources=[],
            mcp_servers={"task": server} if server is not None else {},
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

    async def run(
        self, request: str, requester: str, context_lines: list[str],
        ask: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> TaskResult:
        started = time.monotonic()
        work = self._prepare()
        browser = ReadOnlyBrowser() if self.use_browser else None
        timed_ask = None
        if ask is not None:
            async def timed_ask(q: str) -> str | None:
                try:
                    return await asyncio.wait_for(ask(q), timeout=self.ask_timeout)
                except asyncio.TimeoutError:
                    return None
        # waiting on the chat doesn't count against the working time
        budget = self.timeout + (self.ask_timeout * self.max_questions if ask is not None else 0)
        prompt = (
            (("Recent group chat, for context:\n" + "\n".join(context_lines) + "\n\n") if context_lines else "")
            + f"Task from {requester}: {request}"
        )
        result: Any = None
        try:
            if self._query is None:
                from claude_agent_sdk import query as sdk_query

                server = build_server(browser, timed_ask, self.max_questions) if (browser or timed_ask) else None
                query_fn, options = sdk_query, self._options(work, server)
            else:
                query_fn, options = self._query, {"cwd": str(work), "root": str(self.root), "ask": timed_ask,
                                                  "system_prompt": self.system_prompt}

            async def consume():
                nonlocal result
                async for msg in query_fn(prompt=prompt, options=options):
                    if type(msg).__name__ == "ResultMessage":
                        result = msg

            await asyncio.wait_for(consume(), timeout=budget)
        except asyncio.TimeoutError:
            return TaskResult(False, reason="timeout", duration_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:  # noqa: BLE001
            log_event("task.error", error=exc_name(exc))
            return TaskResult(False, reason=exc_name(exc), duration_ms=int((time.monotonic() - started) * 1000))
        finally:
            shutil.rmtree(self.root / ".claude-config" / "projects", ignore_errors=True)  # session transcripts
            if browser is not None:
                log_event("task.browser", count=browser.pages_opened)
                await browser.close()
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
