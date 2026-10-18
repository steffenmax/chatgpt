# Setup handoff for a local Claude Code session (run on the bot Mac)

You are running on the Mac that will host **imessage-agent**, a private iMessage group-chat
bot. The owner has finished Phase 1: a dedicated Apple ID is signed into Messages.app on this
Mac, Messages in iCloud is off, and a plain reply from Messages.app reaches the owner's phone.
Your job is Phase 2 (install BlueBubbles, Ollama, mflux and this project) and Phase 3
(configure, test, install the launchd agent). Read `README.md` first; it is the reference. This
file is the execution checklist.

## Ground rules

- Privacy is the point of this project. Never `cat`, `grep`, or print message text, the
  Messages database, or BlueBubbles' database. Never paste chat content into this session.
- Never commit `.env`. It is git-ignored; keep it that way.
- `HOST` stays `127.0.0.1`. `CLOUD_LLM_ENABLED` stays `false`. Do not install the `cloud` extra.
- Do not disable System Integrity Protection or set up the BlueBubbles Private API.
  `BLUEBUBBLES_SEND_METHOD=apple-script` is the intended mode.
- Do not change the BlueBubbles proxy to Ngrok or Cloudflare. LAN-only.
- Some steps are GUI-only (permission dialogs, the BlueBubbles wizard, signing into apps).
  Tell the owner exactly what to click, wait for them to say it is done, then verify from the
  shell. Do not try to script System Settings or drive the GUI.
- Large downloads (Ollama model, FLUX weights ~34 GB) are expected. Say so before starting
  each one and run it in the background so the session stays responsive.
- Verify each step with the command given before moving to the next. If a verification
  fails, stop and diagnose; do not skip ahead.

## Phase 2: install

### 2.1 Inventory

```bash
cd ~/imessage-agent 2>/dev/null || { git clone https://github.com/steffenmax/chatgpt.git ~/imessage-agent && cd ~/imessage-agent; }
git checkout imessage-agent
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
scripts/doctor.sh
```

Record chip, RAM, macOS version, and the recommended `OLLAMA_MODEL` and `MFLUX_QUANTIZE` from
the doctor output. Use them later. If the Mac is Intel, plan for `IMAGE_ENABLED=false`.

Check that Messages.app is running and signed into the bot Apple ID (ask the owner to confirm
the "Apple Account" line in Messages → Settings → iMessage shows the bot's email, not a
personal one). If it shows a personal account, stop: Phase 1 is not done.

### 2.2 BlueBubbles Server

If `/Applications/BlueBubbles.app` is missing, download it:

```bash
open https://bluebubbles.app/downloads/server
```

Ask the owner to install it to Applications and open it, then walk them through the wizard:

1. Grant **Full Disk Access** and **Accessibility** when prompted (System Settings → Privacy & Security).
2. **Skip** the Google / Firebase step entirely.
3. Set a strong **server password** and have them paste it to you only once, directly into
   `.env` later. Prefer: they type it into `.env` themselves; you never need to see it.
4. **Proxy Service**: the LAN-only option ("Dynamic DNS" / "LAN URL" depending on version).
5. Leave the port at 1234 unless it conflicts.

Verify (401 without password proves the API is up; 200 with password proves the password):

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:1234/api/v1/ping            # expect 401
curl -s "http://127.0.0.1:1234/api/v1/server/info?password=$BB_PW" | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"];print({k:d.get(k) for k in ("server_version","os_version","detected_imessage","macos_time_sync")})'
```

`detected_imessage` should be true and `macos_time_sync` should not be false. If BlueBubbles
shows no chats in its UI, Full Disk Access was not granted to it; have the owner toggle it and
restart BlueBubbles.

### 2.3 Ollama

```bash
command -v ollama || brew install ollama
brew services start ollama 2>/dev/null || open -a Ollama
curl -s http://127.0.0.1:11434/api/tags >/dev/null && echo ollama up
ollama pull <OLLAMA_MODEL from doctor>          # background; multi-GB
ollama run <OLLAMA_MODEL> "reply with the single word ok"
```

Confirm Ollama is bound to loopback only:

```bash
lsof -nP -iTCP:11434 -sTCP:LISTEN     # should show 127.0.0.1 or localhost, not *:11434
```

If it shows `*:11434`, `OLLAMA_HOST` is set somewhere; unset it.

### 2.4 mflux (skip on Intel)

```bash
uv tool install --upgrade mflux
mflux-generate --help | head -5
```

Tell the owner the first run downloads ~34 GB from Hugging Face, then run in the background:

```bash
mflux-generate --model schnell --steps 4 --quantize <MFLUX_QUANTIZE> --width 512 --height 512 \
  --prompt "a corgi on a surfboard" --output /tmp/mflux-test.png
```

Verify `/tmp/mflux-test.png` exists and `open` it for the owner. Then save a quantized copy so
later runs are faster and offline:

```bash
mkdir -p ~/models
mflux-save --model schnell --quantize <MFLUX_QUANTIZE> --path ~/models/flux-schnell-q<MFLUX_QUANTIZE>
rm -f /tmp/mflux-test.png
```

If generation is killed for memory, retry with `--quantize 4` and/or `--low-ram`, and carry
those into `.env` (`MFLUX_QUANTIZE=4`, `MFLUX_EXTRA_ARGS=--low-ram`).

### 2.5 Project install

```bash
cd ~/imessage-agent
uv sync
uv run pytest -q          # all tests must pass on this Mac too
```

## Phase 3: configure, test, go live

### 3.1 .env

```bash
cp -n .env.example .env
openssl rand -hex 24       # WEBHOOK_SECRET
```

Edit `.env` (use `sed -i ''` or have the owner edit; do not echo the file afterwards since it
holds secrets):

- `BOT_NAME` = whatever the owner wants the bot called
- `BLUEBUBBLES_PASSWORD` = the BlueBubbles server password (owner types it in)
- `WEBHOOK_SECRET` = the generated hex
- `OLLAMA_MODEL`, `MFLUX_QUANTIZE` = from the doctor output
- `MFLUX_MODEL` = `/Users/<owner>/models/flux-schnell-q<N>` if 2.4 saved one
- `UNLOAD_LLM_BEFORE_IMAGE=true` if RAM is 16 GB or less
- `IMAGE_ENABLED=false` on Intel or if mflux failed
- `ALLOWED_CHAT_GUIDS` stays empty until 3.3
- Optional `BOT_PERSONA`; keep the "you can only send text, never claim you did things" sentence.

```bash
uv run imessage-agent check
```

Every line should be reachable/present. Fix before continuing.

### 3.2 Add the bot to the group (owner, on their phone)

Owner: open the group chat → tap the group name → Add Contact → enter the bot Apple ID's
email. The group must be all-iMessage (blue). Then send one message in the group.

### 3.3 Find the GUID, allowlist it

```bash
uv run imessage-agent chats
```

Pick the group's GUID (`iMessage;+;chat...`). Put it in `ALLOWED_CHAT_GUIDS` in `.env`
(quote the value because it contains `;`). Optionally fill `SENDER_NAMES` with the owner's
help so the bot knows people by name; the values are phone numbers/emails mapped to first
names and stay local.

### 3.4 Foreground test

Terminal 1:

```bash
uv run imessage-agent serve
```

Terminal 2 (or background the first):

```bash
uv run imessage-agent register-webhook
uv run imessage-agent send --chat "<GUID>" "hi, bot test"
uv run imessage-agent health
```

The owner should see "hi, bot test" in the group from the bot's address. Then the owner sends,
from their phone, in the group: `/help`, then `<BotName> what's up`, then `/image a corgi surfing`.
Watch the serve output for `event=webhook.received`, `event=trigger`, `event=llm.reply`,
`event=message.sent`, `event=image.queued`, `event=image.sent`. If `/image` says it is off, the
mflux command is not on PATH for the server process; check `MFLUX_COMMAND` or give it the
absolute path from `command -v mflux-generate`.

If nothing arrives at all: `uv run imessage-agent register-webhook` again and confirm it says
"already registered"; confirm the GUID matches exactly; confirm BlueBubbles shows the group.

### 3.5 launchd

Stop the foreground server, then:

```bash
scripts/install-launchd.sh
sleep 3
uv run imessage-agent health
tail -n 20 ~/Library/Logs/imessage-agent/agent.log
```

Confirm the startup lines list only `127.0.0.1` hosts and "Cloud text mode is OFF". Have the
owner send `/help` in the group once more.

### 3.6 Reboot test

Ask the owner to reboot the Mac, log in, wait a minute, and send `/help` in the group. Then:

```bash
pgrep -fl "imessage-agent serve"
uv run imessage-agent health
```

Also confirm sleep prevention: System Settings → Energy (or Displays → Advanced on laptops):
prevent automatic sleeping when the display is off. Recommend auto-login for this user so
Messages.app comes up after a reboot without anyone at the keyboard.

## Done when

- `uv run imessage-agent check` reports OK
- `/help`, a mention, and `/image` all work from the group
- launchd agent is loaded and survives a reboot
- `.env` is untracked (`git status` clean apart from nothing) and `HOST=127.0.0.1`

Report back to the owner with: model chosen, quantization, whether the saved mflux model is
in use, the chat GUID(s) allowlisted, and the exact `launchctl` commands to stop, start and
read logs. Do not include the BlueBubbles password or webhook secret in the summary.
