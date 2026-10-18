#!/usr/bin/env bash
# Pre-flight for the Mac that will run the bot: hardware, macOS, and which pieces
# are installed. Read-only; prints a model recommendation based on RAM.
set -uo pipefail

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [!!]   %s\n' "$*"; }
info() { printf '  [..]   %s\n' "$*"; }

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is meant to run on the Mac that hosts the bot (found $(uname -s))."
  exit 1
fi

echo "== Hardware / OS"
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
ARCH="$(uname -m)"
RAM_BYTES="$(sysctl -n hw.memsize)"
RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
MACOS="$(sw_vers -productVersion)"
info "chip   : $CHIP ($ARCH)"
info "memory : ${RAM_GB} GB"
info "macOS  : $MACOS"
if [[ "$ARCH" != "arm64" ]]; then
  warn "Intel Mac: Ollama works, but mflux (MLX) requires Apple Silicon. Set IMAGE_ENABLED=false."
fi

echo
echo "== Model recommendation for ${RAM_GB} GB (LLM + FLUX schnell must share RAM)"
if   (( RAM_GB <= 8 ));  then echo "  OLLAMA_MODEL=llama3.2:3b     MFLUX_QUANTIZE=4  UNLOAD_LLM_BEFORE_IMAGE=true  (tight; consider IMAGE_ENABLED=false)"
elif (( RAM_GB <= 16 )); then echo "  OLLAMA_MODEL=llama3.1:8b     MFLUX_QUANTIZE=8  UNLOAD_LLM_BEFORE_IMAGE=true"
elif (( RAM_GB <= 24 )); then echo "  OLLAMA_MODEL=qwen2.5:14b     MFLUX_QUANTIZE=8"
elif (( RAM_GB <= 32 )); then echo "  OLLAMA_MODEL=gemma3:27b      MFLUX_QUANTIZE=8  (or qwen2.5:32b)"
else                          echo "  OLLAMA_MODEL=qwen2.5:32b     MFLUX_QUANTIZE=8  (or llama3.3:70b with 64 GB+)"
fi
echo "  Run 'ollama list' to see what you already have; any chat model works."

echo
echo "== Tooling"
for bin in uv python3 ollama mflux-generate; do
  if command -v "$bin" >/dev/null 2>&1; then ok "$bin -> $(command -v "$bin")"; else warn "$bin not found"; fi
done
if command -v python3 >/dev/null; then info "python3 $(python3 --version 2>&1)"; fi

echo
echo "== BlueBubbles Server"
BB_APP="/Applications/BlueBubbles.app"
if [[ -d "$BB_APP" ]]; then
  ok "installed at $BB_APP ($(defaults read "$BB_APP/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null || echo 'version ?'))"
else
  warn "not found in /Applications. Download: https://bluebubbles.app/downloads/server"
fi
if pgrep -xq BlueBubbles; then ok "process running"; else warn "process not running"; fi
BB_PORT="${BLUEBUBBLES_PORT:-1234}"
if curl -fsS -m 3 "http://127.0.0.1:$BB_PORT/api/v1/ping" -o /dev/null -w '' 2>/dev/null \
   || [[ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$BB_PORT/api/v1/ping")" == "401" ]]; then
  ok "REST API answering on 127.0.0.1:$BB_PORT (401 without password is expected)"
else
  warn "nothing answering on 127.0.0.1:$BB_PORT (check BlueBubbles > Settings > API & Webhooks > port)"
fi

echo
echo "== Ollama"
if curl -fsS -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ok "server answering on 127.0.0.1:11434"
  info "models: $(curl -fsS -m 3 http://127.0.0.1:11434/api/tags | python3 -c 'import sys,json;print(", ".join(m["name"] for m in json.load(sys.stdin).get("models",[])) or "(none pulled yet)")' 2>/dev/null)"
else
  warn "not answering on 127.0.0.1:11434 (install: brew install ollama, or https://ollama.com/download; then open the app)"
fi

echo
echo "== Messages.app"
if pgrep -xq Messages; then ok "Messages is running"; else warn "Messages is not running (BlueBubbles needs it open and signed in)"; fi
if [[ -r "$HOME/Library/Messages/chat.db" ]]; then
  ok "Terminal has Full Disk Access (can read chat.db)"
else
  warn "Full Disk Access missing for this terminal/BlueBubbles: System Settings > Privacy & Security > Full Disk Access"
fi

echo
echo "Done. Next: cp .env.example .env, fill it in, then: uv run imessage-agent check"
