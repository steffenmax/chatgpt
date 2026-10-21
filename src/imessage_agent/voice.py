"""Voice notes: the bot's reply spoken with a local text-to-speech model via mlx-audio.

Models: Kokoro (fast, plain) or Orpheus (slower, expressive: <laugh>, <sigh> ... tags in the
text become the sound). Orpheus can't take phonemes, so names get a plain respelling there
(VOICE_RESPELL="dziuba=Jooba").

Text is cleaned for speech (no links, emoji or [notes]), names get their pronunciation
(PRONUNCIATIONS="dziuba=ʤˈubə" -> "[dziuba](/ʤˈubə/)", Kokoro's inline phoneme syntax),
then it's synthesized to WAV and converted to AAC .m4a with macOS afconvert. BlueBubbles'
AppleScript sending can't make a native voice-memo bubble (that needs the Private API), so
it arrives as a playable audio attachment. Everything runs on this Mac.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

from .logsafe import log_event

MAX_CHARS = 400
# Orpheus's sound tags; anything else in <angle brackets> is dropped.
SOUND_TAGS = ("laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp")
_TAG = re.compile(r"<\s*/?\s*([a-z]+)\s*>", re.IGNORECASE)


def strip_tags(text: str) -> str:
    return " ".join(_TAG.sub("", text or "").split())

_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NOTE = re.compile(r"\[[^\]]*\]")

VOICE_ASK = re.compile(
    r"\b(voice (note|memo|message|msg)|voice ?note|in a voice|send (me |us )?(a )?(voice|audio)|out ?loud|aloud|"
    r"let (me|us) hear (you|it|that)|say it with your (mouth|chest)|talk to (me|us))\b",
    re.IGNORECASE,
)
SAY_LAST = re.compile(r"\b(say|read) (it|that|this)\b|\bthat\b.*\b(out ?loud|aloud|voice)\b", re.IGNORECASE)


def wants_voice(text: str) -> bool:
    return bool(VOICE_ASK.search(text or ""))


def repeat_last(text: str) -> bool:
    """'say that out loud' / 'read it' means speak the bot's last message, not write a new one."""
    return bool(SAY_LAST.search(text or ""))


def parse_pronunciations(raw: str) -> dict[str, str]:
    out = {}
    for part in (raw or "").split(","):
        if "=" in part:
            word, ph = part.split("=", 1)
            if word.strip() and ph.strip():
                out[word.strip().lower()] = ph.strip().strip("/")
    return out


def speakable(text: str, pronunciations: dict[str, str] | None = None, respell: dict[str, str] | None = None,
              tags: bool = False) -> str:
    """Text ready to speak. Kokoro: phoneme overrides, no tags. Orpheus (tags=True): respellings, sound tags kept."""
    t = _URL.sub("", text or "")
    t = _TAG.sub(lambda m: f"<{m.group(1).lower()}>" if tags and m.group(1).lower() in SOUND_TAGS else " ", t)
    t = _NOTE.sub("", t)
    t = _EMOJI.sub("", t)
    t = " ".join(t.split())
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS].rsplit(" ", 1)[0]
    for word, ph in (pronunciations or {}).items():
        t = re.sub(rf"\b({re.escape(word)})\b", lambda m: f"[{m.group(1)}](/{ph}/)", t, flags=re.IGNORECASE)
    for word, plain in (respell or {}).items():
        t = re.sub(rf"\b{re.escape(word)}\b", plain, t, flags=re.IGNORECASE)
    return t


class VoiceSynth:
    def __init__(self, command: str = "mlx_audio.tts.generate", model: str = "mlx-community/Kokoro-82M-bf16",
                 voice: str = "am_onyx", speed: float = 1.0, pronunciations: dict[str, str] | None = None,
                 timeout: float = 120.0, respell: dict[str, str] | None = None):
        self.command = command
        self.model = model
        self.voice = voice
        self.speed = speed
        self.pronunciations = pronunciations or {}
        self.respell = respell or {}
        self.timeout = timeout

    @property
    def expressive(self) -> bool:
        """Orpheus: understands <laugh>-style tags (and not phonemes)."""
        return "orpheus" in self.model.lower()

    def available(self) -> bool:
        cmd = os.path.expanduser(self.command)
        return (Path(cmd).is_file() if os.sep in cmd else shutil.which(cmd) is not None) and shutil.which("afconvert") is not None

    async def _run(self, argv: list[str], cwd: Path) -> int:
        proc = await asyncio.create_subprocess_exec(*argv, cwd=str(cwd), stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL)
        try:
            return await asyncio.wait_for(proc.wait(), timeout=self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise

    async def synthesize(self, text: str, workdir: Path) -> Path | None:
        """An .m4a of `text` in `workdir`, or None if there's nothing to say or it failed."""
        if self.expressive:
            say = speakable(text, respell=self.respell, tags=True)
        else:
            say = speakable(text, self.pronunciations)
        if not re.search(r"[a-z0-9]", strip_tags(say), re.IGNORECASE):
            return None
        argv = [os.path.expanduser(self.command), "--model", self.model, "--text", say, "--voice", self.voice,
                "--speed", str(self.speed), "--file_prefix", "note", "--audio_format", "wav", "--join_audio"]
        if not self.expressive:
            argv += ["--lang_code", "b" if self.voice.startswith("b") else "a"]
        code = await self._run(argv, workdir)
        wavs = sorted(workdir.glob("note*.wav"))
        if code != 0 or not wavs:
            log_event("voice.failed", status=code)
            return None
        out = workdir / "voice-note.m4a"
        if await self._run(["afconvert", "-f", "m4af", "-d", "aac", str(wavs[0]), str(out)], workdir) != 0 or not out.exists():
            log_event("voice.convert_failed")
            return None
        return out


def new_workdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="voice-"))
    os.chmod(d, 0o700)
    return d
