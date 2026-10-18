# imessage-agent

A private, self-hosted AI participant for iMessage group chats. Everything runs on one Mac:

- **BlueBubbles Server** bridges iMessage (local REST API + webhooks)
- **Ollama** generates text replies (local LLM)
- **mflux** generates images with FLUX on Apple Silicon (local, always)
- This project: a small FastAPI service bound to `127.0.0.1` that glues them together

By default **no message content leaves the machine**. See [Privacy model](#privacy-model).

```
 iPhone / friends ──iMessage──▶ Apple ──▶ Messages.app (bot's Apple ID, on this Mac)
                                               │ reads chat.db / AppleScript
                                               ▼
                                     BlueBubbles Server :1234
                                      │ webhook POST          ▲ REST send
                                      ▼                       │
                              imessage-agent 127.0.0.1:8765 ──┘
                                 │  text              │ /image
                                 ▼                    ▼
                          Ollama :11434          mflux-generate
                          (local LLM)           (FLUX, local, temp file deleted after send)
```

## What it does

| Trigger | Behaviour |
|---|---|
| Message contains the bot's name (`BOT_NAME`, whole word, `@` optional) | Replies, short and casual, using the last N messages of that chat as context (with sender names) |
| `/image <prompt>` | Replies "on it", queues a local FLUX job (one at a time), sends the PNG back, deletes the file |
| `/ask <question>` | Only with `CLOUD_FACTS_ENABLED=true`: answers via the Anthropic API with web search |
| A question (`...?`) nobody answers for `QUIET_QUESTION_SECONDS` | Treated as addressed to the bot (off by default) |
| `/forget` | Wipes that chat's context immediately |
| `/help` | Lists commands |
| Anything else | Ignored (but remembered in the rolling context so replies make sense) |

- Only responds in chats listed in `ALLOWED_CHAT_GUIDS`. Other chats that try are logged **by GUID only** so you can add them.
- Context lives in RAM only: last 30 messages per chat (configurable), wiped after 6 idle hours (configurable) and on every restart.
- Ignores its own messages, reactions/tapbacks, and group system events, and de-duplicates webhook deliveries.

## Requirements

- Apple Silicon Mac (Intel works for text; mflux needs MLX/Apple Silicon)
- macOS 13+ recommended, a second Apple ID for the bot, and the Mac left on
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- RAM: 16 GB is comfortable for an 8B LLM plus FLUX schnell at 8-bit. See the table in the Ollama step.

Run `scripts/doctor.sh` on the Mac first. It prints chip, RAM, macOS version, whether BlueBubbles/Ollama/mflux are installed and reachable, and a model recommendation for that RAM.

## Setup, step by step

### 1. A dedicated Apple ID for the bot

Use a separate Apple ID so the bot is its own participant and never has access to your personal iMessage history.

1. Create one at <https://account.apple.com> (needs an email and a phone number for two-factor verification; the phone can be your own).
2. On the Mac that will run the bot, open **Messages → Settings → iMessage** and sign in with the bot Apple ID. Under "You can be reached for messages at", make sure the bot's email is checked. Set "Start new conversations from" to that email.
3. **Turn off "Enable Messages in iCloud"** for this account (same settings pane). Otherwise the bot's copy of every group message syncs to Apple's servers under the bot account.
4. Leave Messages.app running. BlueBubbles needs it open and signed in.

You can keep your own Apple ID for iCloud/App Store on the Mac; only Messages needs to be the bot account.

### 2. BlueBubbles Server

1. Download the server from <https://bluebubbles.app/downloads/server> and move it to `/Applications`.
2. On first launch grant **Full Disk Access** (to read the Messages database) and **Accessibility** (to send via AppleScript) when prompted. Both are in System Settings → Privacy & Security.
3. In the setup wizard:
   - **Skip the Google / Firebase step.** It only exists for the BlueBubbles phone apps; this bot does not use it.
   - Set a strong **server password**. This is `BLUEBUBBLES_PASSWORD`.
   - For **Proxy Service**, choose the LAN-only option (shown as "Dynamic DNS" / "LAN URL" depending on version), not Ngrok or Cloudflare. That way BlueBubbles never opens a public tunnel to your iMessage account.
4. Note the port under **Settings → API & Webhooks** (default `1234`). The bot talks to `http://127.0.0.1:1234`.
5. Optional: the **Private API** (reactions, typing indicators, replies, faster sending) requires disabling System Integrity Protection. The bot works fine without it using `BLUEBUBBLES_SEND_METHOD=apple-script`. Only switch to `private-api` if you have gone through BlueBubbles' Private API guide and the toggle shows connected.

Verify from Terminal (expect `{"status":200,"message":"pong"}`):

```bash
curl "http://127.0.0.1:1234/api/v1/ping?password=YOUR_PASSWORD"
```

### 3. Ollama and a model

```bash
brew install ollama        # or download from https://ollama.com/download
ollama serve &             # the Ollama.app menu-bar item does the same
```

Pick a model by RAM. These all fit alongside FLUX schnell at 8-bit on the listed size (on 16 GB, also set `UNLOAD_LLM_BEFORE_IMAGE=true`):

| Mac RAM | `OLLAMA_MODEL` | Notes |
|---|---|---|
| 8 GB | `llama3.2:3b` | tight with images; consider `IMAGE_ENABLED=false` or `MFLUX_QUANTIZE=4` |
| 16 GB | `llama3.1:8b` (default) | `qwen2.5:7b` is a good alternative |
| 24 GB | `qwen2.5:14b` | |
| 32 GB+ | `gemma3:27b` or `qwen2.5:32b` | |

```bash
ollama pull llama3.1:8b
```

Ollama binds to `127.0.0.1:11434` by default. Do not set `OLLAMA_HOST=0.0.0.0`.

The model download is the only time Ollama talks to the internet (ollama.com registry). Pulling a model carries no chat content.

### 4. mflux (local image generation)

```bash
uv tool install --upgrade mflux
mflux-generate --help | head
```

The first image generation downloads FLUX.1-schnell weights (~34 GB) from Hugging Face. That is a one-time model download with no chat content in it. To avoid re-quantizing on every run and to work fully offline afterwards, save a quantized copy once and point the bot at it:

```bash
mkdir -p ~/models
mflux-save --model schnell --quantize 8 --path ~/models/flux-schnell-8bit
# then in .env:  MFLUX_MODEL=/Users/you/models/flux-schnell-8bit
```

Test it by hand before wiring it to the bot:

```bash
mflux-generate --model schnell --steps 4 --quantize 8 --width 512 --height 512 \
  --prompt "a corgi on a surfboard" --output /tmp/test.png && open /tmp/test.png
```

### 5. Install this project

```bash
git clone <this repo> ~/imessage-agent && cd ~/imessage-agent
uv sync                      # add --extra cloud only if you plan to enable cloud text mode
cp .env.example .env
openssl rand -hex 24         # paste as WEBHOOK_SECRET
```

Edit `.env`: `BOT_NAME`, `BLUEBUBBLES_PASSWORD`, `WEBHOOK_SECRET`, `OLLAMA_MODEL`. Leave `ALLOWED_CHAT_GUIDS` for the next step.

```bash
uv run imessage-agent check
```

This prints every outbound host the bot is configured to call and tests BlueBubbles, Ollama (and whether the model is pulled) and mflux.

### 6. Add the bot to a group chat and find the chat GUID

1. From your own phone, open the group chat → tap the group name at the top → **Add Contact** → enter the bot Apple ID's email. The group must be an iMessage group (blue bubbles); SMS/RCS groups will not work.
2. Send any message in the group so BlueBubbles sees the chat.
3. List chats and copy the GUID:

```bash
uv run imessage-agent chats
```

Group chat GUIDs look like `iMessage;+;chat123456789012345678` (one-to-one chats look like `iMessage;-;+15551234567`). Put the group GUIDs, comma-separated, into `ALLOWED_CHAT_GUIDS`.

Alternatively, start the bot with an empty allowlist, say the bot's name in the group, and read the GUID from the log line `event=chat.not_allowlisted chat_guid=...`.

### 7. Register the webhook and test

```bash
uv run imessage-agent serve &          # foreground run for the first test
uv run imessage-agent register-webhook # tells BlueBubbles to POST new-message events to 127.0.0.1:8765/webhook?secret=...
uv run imessage-agent send --chat "iMessage;+;chat123..." "hi from the bot"
uv run imessage-agent health
```

Then in the group: `/help`, `<BotName> what's up`, `/image a corgi surfing`.

BlueBubbles can only put the secret in the URL (its webhooks send no custom headers), which is why the registration URL contains `?secret=...`. That URL is stored inside BlueBubbles' local database and only ever used against `127.0.0.1`.

### 8. Run on boot with launchd

```bash
scripts/install-launchd.sh
tail -f ~/Library/Logs/imessage-agent/agent.log
```

The LaunchAgent starts at login, restarts on crash (`KeepAlive`), and logs to `~/Library/Logs/imessage-agent/`. `scripts/install-launchd.sh uninstall` removes it. The Mac must be logged into the bot's user session for Messages.app to run; disable sleep in System Settings → Energy (or Displays → Advanced on laptops) and consider auto-login for that user.

## CLI

| Command | Purpose |
|---|---|
| `imessage-agent serve` | Run the webhook server in the foreground |
| `imessage-agent check` | Print configured outbound hosts; test BlueBubbles, Ollama, model presence, mflux |
| `imessage-agent chats` | List chats with GUIDs, names, participant counts |
| `imessage-agent register-webhook [--replace]` | Create the BlueBubbles webhook for `new-message` |
| `imessage-agent send --chat <guid> "text"` | Send a test message |
| `imessage-agent health` | Query `GET /health` on the running server |

`GET http://127.0.0.1:8765/health` returns status, BlueBubbles/Ollama reachability, backend name, cloud flag, image queue length, and how many chats currently hold context.

## Privacy model

**Where data exists, and for how long:**

| Data | Where | Lifetime | Leaves the Mac? |
|---|---|---|---|
| Message text of allowlisted chats | RAM of the bot process (`ContextStore`), max N per chat | Until 6 h idle, `/forget`, or process restart | No (unless cloud text mode is on) |
| Message text of all chats the bot account is in | Messages.app `chat.db` and BlueBubbles' own SQLite under `~/Library/Application Support/bluebubbles-server` | Apple/BlueBubbles retention; not controlled by this bot | Only via Apple's iMessage delivery (end-to-end encrypted); Messages in iCloud should be off for the bot account |
| Image prompts | RAM; a `0600` temp file in a `0700` temp dir for the duration of one mflux run | Deleted when the job finishes | No, ever |
| Generated images | Temp dir until sent, then sent through BlueBubbles to iMessage | Deleted right after sending | Only to the chat that asked |
| Bot replies | Sent through BlueBubbles; kept in RAM context like any other message | As above | Only to the chat |
| Logs | `~/Library/Logs/imessage-agent/agent.log` (stderr) | Until you delete them | No |
| Secrets (`BLUEBUBBLES_PASSWORD`, `WEBHOOK_SECRET`, API key) | `.env` and process environment; the webhook URL including the secret is stored in BlueBubbles' local DB | Until changed | No |

**What is logged:** event type, chat GUID, message GUID, command kind, durations, queue length, HTTP status, exception class names, timestamps. The logging helper only accepts an allow-list of field names and raises if code tries to log anything else; a filter additionally scrubs all three secrets from every record (including URLs inside HTTP exceptions). uvicorn's access log is disabled because it would print the webhook query string. Tests assert that message text, prompts, bot replies and secrets never appear in captured logs.

**Network:** the process opens connections only to the hosts printed at startup: BlueBubbles (`127.0.0.1:1234`), Ollama (`127.0.0.1:11434`) and, only if `CLOUD_LLM_ENABLED=true`, `api.anthropic.com:443`. The server refuses to bind to anything but loopback. There is no telemetry, update check, crash reporter or analytics. Model downloads (`ollama pull`, mflux's first run) are separate one-time downloads of public weights.

**Cloud fact lookups (`CLOUD_FACTS_ENABLED=true`, default off):** a lighter middle ground. Replies stay local unless the local model judges that the last message needs outside or current information (scores, news, schedules, prices), or someone uses `/ask`. Only those mentions send the triggering message plus that chat's rolling context to the Anthropic API, which answers with web search (`CLOUD_FACTS_MAX_SEARCHES` per reply). The log records every escalation as `event=llm.escalate` with the reason (`router` or `command`). Needs `ANTHROPIC_API_KEY` and `uv sync --extra cloud`.

**Cloud text mode (`CLOUD_LLM_ENABLED=true`, default off):** the triggering message plus that chat's rolling context (sender names and text) are sent to the Anthropic API to produce the text reply. Nothing else changes: images are always generated locally, the allowlist/TTL/logging rules still apply, and a boxed warning is printed at startup. Install the optional dependency with `uv sync --extra cloud`.

**Threat model notes:**
- Anyone in an allowlisted group can trigger the bot and see its replies; the bot's context for that chat is only ever sent back into that same chat.
- The bot reads every message in groups it is a member of (it must, to keep context). It never persists them.
- Another local process on the Mac could reach `127.0.0.1:8765` but cannot inject messages without the webhook secret, and `/health` exposes only counters.
- The BlueBubbles server password grants full read/send access to the bot account's iMessage. Keep BlueBubbles LAN-only and firewalled.

## Development

```bash
uv sync --group dev
uv run pytest
```

Tests cover mention detection, slash-command parsing, the allowlist (including GUID-only logging), context window and TTL wipe, `/forget`, image queueing/cleanup/failure handling, webhook authentication, health, and the log-redaction guarantee.

Layout:

```
src/imessage_agent/
  config.py      settings (.env), outbound-host list, startup validation
  logsafe.py     allow-listed structured logging + secret redaction
  context.py     in-memory rolling context with TTL
  triggers.py    mention / command detection
  bluebubbles.py BlueBubbles REST client (send text, attachment, chats, webhooks, contacts)
  llm.py         Ollama backend, optional Anthropic backend, prompt assembly
  imagegen.py    mflux subprocess + single-worker queue
  handler.py     the agent: webhook payload -> action
  main.py        FastAPI app (/webhook, /health), lifespan, wiring
  cli.py         imessage-agent command
```

## Troubleshooting

- **`check` says BlueBubbles not reachable**: BlueBubbles Server not running, wrong port, or wrong password. `curl "http://127.0.0.1:1234/api/v1/ping?password=..."`.
- **Bot never answers**: confirm the webhook exists (`register-webhook` prints "already registered"), the chat GUID is in `ALLOWED_CHAT_GUIDS` exactly, and the message contains the bot name as a whole word. Watch the log for `event=webhook.received` and `event=trigger`.
- **Replies arrive but `/image` fails**: run the manual `mflux-generate` test from step 4; check RAM (`UNLOAD_LLM_BEFORE_IMAGE=true`, `MFLUX_QUANTIZE=4`, `MFLUX_EXTRA_ARGS=--low-ram`). launchd has a minimal `PATH`; the plist adds `~/.local/bin` and `/opt/homebrew/bin`.
- **Sent messages fail with apple-script**: BlueBubbles needs Accessibility permission and Messages.app in the foreground session. Group sends via AppleScript require the group to already exist in Messages.app on the Mac.
- **Duplicate replies**: only one webhook should point at the bot; use `register-webhook --replace`.
