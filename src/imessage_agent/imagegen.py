"""Local image generation with mflux (FLUX on Apple Silicon).

A single asyncio worker drains a queue so only one generation runs at a time.
The prompt is handed to mflux via a 0600 temp file (--prompt-file) so it never shows
up in `ps` output; the file and the generated PNG are deleted as soon as the image
has been sent (or the job failed).
"""

from __future__ import annotations

import asyncio
import os
import random
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from .logsafe import log_event, exc_name


class ImageGenError(RuntimeError):
    pass


@dataclass
class ImageJob:
    chat_guid: str
    prompt: str
    job_id: str
    created: float = field(default_factory=time.monotonic)


class MfluxGenerator:
    def __init__(
        self,
        command: str = "mflux-generate",
        model: str = "schnell",
        steps: int = 4,
        quantize: int | None = 8,
        width: int = 768,
        height: int = 768,
        extra_args: str = "",
        timeout: float = 600.0,
    ):
        self.command = command
        self.model = model
        self.steps = steps
        self.quantize = quantize
        self.width = width
        self.height = height
        self.extra_args = shlex.split(extra_args) if extra_args else []
        self.timeout = timeout

    def available(self) -> bool:
        if os.sep in self.command:
            return Path(self.command).expanduser().is_file()
        return shutil.which(self.command) is not None

    def _argv(self, prompt_file: Path | None, prompt: str | None, out: Path, seed: int) -> list[str]:
        argv = [
            self.command,
            "--model", self.model,
            "--steps", str(self.steps),
            "--seed", str(seed),
            "--width", str(self.width),
            "--height", str(self.height),
            "--output", str(out),
        ]
        if self.quantize:
            argv += ["--quantize", str(self.quantize)]
        if prompt_file is not None:
            argv += ["--prompt-file", str(prompt_file)]
        else:
            argv += ["--prompt", prompt or ""]
        argv += self.extra_args
        return argv

    async def _run(self, argv: list[str], workdir: Path) -> tuple[int, str]:
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "0")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir),
            env=env,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ImageGenError("timeout")
        return proc.returncode or 0, (err or b"").decode(errors="replace")

    async def generate(self, prompt: str, workdir: Path) -> Path:
        """Generate one image. Returns the PNG path inside `workdir`."""
        seed = random.randint(1, 2**31 - 1)
        out = workdir / "image.png"
        prompt_file = workdir / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        os.chmod(prompt_file, 0o600)

        code, err = await self._run(self._argv(prompt_file, None, out, seed), workdir)
        if code != 0 and "prompt-file" in err:
            # Older mflux without --prompt-file: fall back to passing the prompt directly.
            code, err = await self._run(self._argv(None, prompt, out, seed), workdir)
        try:
            prompt_file.unlink()
        except OSError:
            pass
        if code != 0:
            raise ImageGenError(f"mflux exited with code {code}")
        if not out.exists():
            # mflux may add a suffix; pick the first png written.
            pngs = sorted(workdir.glob("*.png"))
            if not pngs:
                raise ImageGenError("no image produced")
            out = pngs[0]
        return out


class ImageQueue:
    """One worker, FIFO. `sender` is called with (job, png_path) and must consume the file."""

    def __init__(
        self,
        generator: MfluxGenerator,
        sender: Callable[[ImageJob, Path], Awaitable[None]],
        on_failure: Callable[[ImageJob, str], Awaitable[None]],
        before_generate: Callable[[], Awaitable[None]] | None = None,
        gate: asyncio.Semaphore | None = None,
    ):
        self.generator = generator
        self._sender = sender
        self._on_failure = on_failure
        self._before = before_generate
        # Shared with the text backend: while held, no LLM reply runs, so the LLM
        # unloaded by `before_generate` cannot be reloaded mid-generation (RAM).
        self._gate = gate
        self._queue: asyncio.Queue[ImageJob] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker(), name="image-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def qsize(self) -> int:
        return self._queue.qsize()

    async def submit(self, job: ImageJob) -> int:
        await self._queue.put(job)
        log_event("image.queued", chat_guid=job.chat_guid, job_id=job.job_id, queue_len=self._queue.qsize())
        return self._queue.qsize()

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            started = time.monotonic()
            tmp = Path(tempfile.mkdtemp(prefix="imsg-img-"))
            os.chmod(tmp, 0o700)
            try:
                if self._gate is not None:
                    await self._gate.acquire()
                try:
                    if self._before:
                        await self._before()
                    png = await self.generator.generate(job.prompt, tmp)
                finally:
                    if self._gate is not None:
                        self._gate.release()
                log_event(
                    "image.generated",
                    chat_guid=job.chat_guid,
                    job_id=job.job_id,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                await self._sender(job, png)
            except Exception as exc:  # noqa: BLE001
                log_event("image.failed", chat_guid=job.chat_guid, job_id=job.job_id, error=exc_name(exc))
                try:
                    await self._on_failure(job, exc_name(exc))
                except Exception as exc2:  # noqa: BLE001
                    log_event("image.failure_notify_failed", chat_guid=job.chat_guid, error=exc_name(exc2))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                self._queue.task_done()
