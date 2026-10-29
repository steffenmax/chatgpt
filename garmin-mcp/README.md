# garmin-mcp

An [MCP](https://modelcontextprotocol.io) server that lets Claude (Desktop, Claude Code, or any
MCP client) read your **Garmin Connect** data: sleep, HRV, Body Battery, stress, heart rate,
training readiness/status, VO2 max, race predictions, activities, body composition, gear and
your training calendar.

It runs locally on your computer and talks to Garmin Connect as you, using the community
[`garminconnect`](https://github.com/cyberjar09/python-garminconnect) library. Garmin has no
public API for personal accounts (the official Health API is for approved businesses), so this
uses the same endpoints as the Garmin Connect website. It may break when Garmin changes things;
updating `garminconnect` usually fixes it.

**Read-only by default.** Two small write tools (log weight, log water) exist but are hidden
unless you set `GARMIN_MCP_ALLOW_WRITES=1`.

## Tools

| Tool | What it returns |
|---|---|
| `get_profile` | Name, unit system, timezone, primary device |
| `get_daily_summary` | Steps, calories, floors, intensity minutes, HR, stress, Body Battery, SpO2 for a day |
| `get_sleep` | Sleep score, stages, overnight HRV/HR, Body Battery change |
| `get_heart_rate` | Resting / min / max HR, 7-day resting average |
| `get_stress` | Average / max stress, time in each stress level |
| `get_body_battery` | Body Battery per day over a range |
| `get_hrv` | Overnight HRV, weekly average, baseline, status |
| `get_training_readiness` | Readiness score and contributing factors |
| `get_training_status` | Productive / maintaining / overreaching, load balance, VO2 max |
| `get_fitness_metrics` | VO2 max, fitness age, race predictions, endurance score |
| `get_personal_records` | PRs |
| `list_activities` | Recent activities (compact: km, minutes, pace, HR, power, training effect) |
| `get_activity` | One activity in full, optionally with laps and HR zones |
| `get_trends` | One value per day for steps, resting HR, sleep, stress, Body Battery, HRV, intensity minutes, calories or weight |
| `get_body_composition` | Weight, BMI, body fat, muscle/bone mass |
| `get_calendar` | Upcoming scheduled workouts and race events |
| `get_devices`, `get_gear` | Your watches/sensors, shoes/bikes |
| `log_weight`, `log_hydration` | *(only with `GARMIN_MCP_ALLOW_WRITES=1`)* |

Per-minute time series (HR samples, stress samples, sleep movement) are left out unless you ask
for them with `include_samples`, so answers stay small and fast.

## Setup

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone -b garmin-mcp https://github.com/steffenmax/chatgpt.git
cd chatgpt/garmin-mcp
uv sync
```

### 1. Sign in once

```bash
uv run garmin-mcp login
```

It asks for your Garmin email, password and (if enabled) the MFA code, then saves OAuth tokens
to `~/.garminconnect/garmin_tokens.json` (mode 600). Your password is **not** stored. Tokens
refresh themselves; if Garmin ever rejects them, the tools will say so and you run `login`
again (`login --force` to replace working tokens).

```bash
uv run garmin-mcp status   # check the saved tokens work
uv run garmin-mcp logout   # delete them
```

### 2. Add it to Claude

**Claude Code**

```bash
claude mcp add garmin -- uv run --directory /ABSOLUTE/PATH/TO/chatgpt/garmin-mcp garmin-mcp
```

**Claude Desktop**: Settings → Developer → Edit Config, then add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "garmin": {
      "command": "uv",
      "args": ["run", "--directory", "/ABSOLUTE/PATH/TO/chatgpt/garmin-mcp", "garmin-mcp"]
    }
  }
}
```

Use the full path to `uv` (`which uv`) if Claude Desktop can't find it. Restart Claude Desktop
and ask something like *"How did I sleep this week compared with last week?"* or *"Am I ready
for a hard run today?"*

### Other MCP clients / HTTP

`garmin-mcp serve --transport http` serves Streamable HTTP on `http://127.0.0.1:8787/mcp`.
It has **no authentication**, so keep it on localhost. Don't expose it to the internet as-is:
anyone who can reach it can read your health data.

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GARMINTOKENS` | `~/.garminconnect` | Where tokens are stored |
| `GARMIN_EMAIL`, `GARMIN_PASSWORD` | – | Used by `login` instead of prompting; the server also falls back to them if tokens are missing (only works without MFA) |
| `GARMIN_IS_CN` | `false` | Set for garmin.cn accounts |
| `GARMIN_MCP_ALLOW_WRITES` | `false` | Expose `log_weight` and `log_hydration` |
| `GARMIN_MCP_LOG_LEVEL` | `WARNING` | Logs go to stderr |

Example with writes enabled in Claude Desktop: add `"env": {"GARMIN_MCP_ALLOW_WRITES": "1"}`
next to `"args"`.

## Notes

- Dates accept `YYYY-MM-DD`, `today` or `yesterday`, and are interpreted in your computer's
  local date. Ranges are capped at 366 days.
- Garmin values are metric (metres, seconds, m/s, grams for weight) unless a field name says
  otherwise.
- Too many logins in a short time gets you rate-limited by Garmin for a while; the token cache
  exists so that normally only `login` ever sends your password.

## Development

```bash
uv run pytest
```

Tests use a fake Garmin client and the MCP SDK's in-memory client, so they need no account.
