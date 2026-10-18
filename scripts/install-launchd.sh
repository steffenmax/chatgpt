#!/usr/bin/env bash
# Render the launchd plist for this checkout and load it as a per-user LaunchAgent.
# Usage: scripts/install-launchd.sh            (install / reload)
#        scripts/install-launchd.sh uninstall
set -euo pipefail

LABEL="com.local.imessage-agent"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$PROJECT_DIR/launchd/$LABEL.plist.template"
RENDERED="$PROJECT_DIR/launchd/$LABEL.rendered.plist"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
UV_BIN="$(command -v uv || true)"
UID_NUM="$(id -u)"

if [[ "${1:-}" == "uninstall" ]]; then
  launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
  rm -f "$TARGET"
  echo "removed $TARGET"
  exit 0
fi

[[ -n "$UV_BIN" ]] || { echo "uv not found on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[[ -f "$PROJECT_DIR/.env" ]] || { echo "no .env in $PROJECT_DIR (copy .env.example first)"; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/imessage-agent"
chmod 700 "$HOME/Library/Logs/imessage-agent"

sed -e "s|__UV__|$UV_BIN|g" \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    "$TEMPLATE" > "$RENDERED"
plutil -lint "$RENDERED" >/dev/null
cp "$RENDERED" "$TARGET"

launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$TARGET"
launchctl kickstart -k "gui/$UID_NUM/$LABEL"

echo "installed and started $LABEL"
echo "  status : launchctl print gui/$UID_NUM/$LABEL | head -20"
echo "  logs   : tail -f ~/Library/Logs/imessage-agent/agent.log"
echo "  stop   : launchctl bootout gui/$UID_NUM/$LABEL"
