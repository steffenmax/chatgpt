"""The MCP server: tool definitions over a `GarminSession`."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .config import Settings
from .garmin import GarminError, GarminSession

READ = ToolAnnotations(read_only_hint=True, open_world_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

MAX_RANGE_DAYS = 366
# Per-minute/per-epoch time series (heart rate, stress, sleep movement...) are huge and
# rarely what the model needs; top-level lists longer than this are dropped by default.
SERIES_THRESHOLD = 48

INSTRUCTIONS = """\
Read access to the user's Garmin Connect account: daily health (steps, sleep, heart rate,
stress, Body Battery, HRV), training metrics (readiness, status, VO2 max, race predictions),
activities, body composition, devices, gear and calendar.

Dates are YYYY-MM-DD, or "today" / "yesterday". Garmin units are metric: distances in
metres, durations in seconds, speeds in m/s, unless a field name says otherwise
(the compact activity list adds km, minutes and pace fields).
Start with list_activities or get_daily_summary; use get_trends for multi-day questions
instead of calling a daily tool many times."""

TrendMetric = Literal[
    "steps",
    "resting_hr",
    "sleep",
    "stress",
    "body_battery",
    "hrv",
    "intensity_minutes",
    "calories",
    "weight",
]


def parse_date(value: str | None, field: str = "date") -> date:
    if value is None or value.strip().lower() in {"", "today"}:
        return date.today()
    v = value.strip().lower()
    if v == "yesterday":
        return date.today() - timedelta(days=1)
    try:
        return date.fromisoformat(v)
    except ValueError:
        raise ToolError(f"{field} must be YYYY-MM-DD, 'today' or 'yesterday', got {value!r}") from None


def parse_range(start: str | None, end: str | None, default_days: int = 7) -> tuple[str, str]:
    end_d = parse_date(end, "end_date")
    start_d = parse_date(start, "start_date") if start else end_d - timedelta(days=default_days - 1)
    if start_d > end_d:
        raise ToolError("start_date must be on or before end_date")
    if (end_d - start_d).days + 1 > MAX_RANGE_DAYS:
        raise ToolError(f"Date ranges are limited to {MAX_RANGE_DAYS} days")
    return start_d.isoformat(), end_d.isoformat()


def strip_series(data: Any, include_samples: bool) -> Any:
    """Drop long top-level lists (raw time series) unless the caller asked for them."""
    if include_samples or not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in data.items():
        if isinstance(v, list) and len(v) > SERIES_THRESHOLD:
            dropped.append(k)
        else:
            out[k] = v
    if dropped:
        out["_omitted_series"] = sorted(dropped)
    return out


def as_object(data: Any) -> dict[str, Any]:
    """Structured tool output must be an object; Garmin sometimes returns a list or null."""
    return data if isinstance(data, dict) else {"result": data}


def _round(x: Any, nd: int = 2) -> Any:
    return round(x, nd) if isinstance(x, (int, float)) else None


def _pace(speed_mps: Any) -> str | None:
    if not isinstance(speed_mps, (int, float)) or speed_mps <= 0:
        return None
    secs = round(1000 / speed_mps)
    return f"{secs // 60}:{secs % 60:02d}"


COMPACT_FIELDS = (
    "activityId",
    "activityName",
    "startTimeLocal",
    "distance",
    "duration",
    "movingDuration",
    "elevationGain",
    "averageHR",
    "maxHR",
    "calories",
    "averageSpeed",
    "avgPower",
    "normPower",
    "averageRunningCadenceInStepsPerMinute",
    "aerobicTrainingEffect",
    "anaerobicTrainingEffect",
    "activityTrainingLoad",
    "vO2MaxValue",
    "locationName",
)


def compact_activity(a: dict[str, Any]) -> dict[str, Any]:
    out = {k: a[k] for k in COMPACT_FIELDS if a.get(k) is not None}
    atype = a.get("activityType")
    if isinstance(atype, dict):
        out["type"] = atype.get("typeKey")
    if isinstance(a.get("distance"), (int, float)):
        out["distance_km"] = _round(a["distance"] / 1000)
    if isinstance(a.get("duration"), (int, float)):
        out["duration_min"] = _round(a["duration"] / 60, 1)
    if pace := _pace(a.get("averageSpeed")):
        out["pace_min_per_km"] = pace
    return out


def build_server(session: GarminSession, settings: Settings) -> MCPServer:
    mcp = MCPServer(name="garmin", title="Garmin Connect", instructions=INSTRUCTIONS)

    async def run(fn: Callable[[Any], Any]) -> Any:
        try:
            return await session.call(fn)
        except GarminError as e:
            raise ToolError(str(e)) from e

    # ---- profile & devices -------------------------------------------------

    @mcp.tool(annotations=READ)
    async def get_profile() -> dict[str, Any]:
        """The signed-in user's name, unit system, timezone and primary device."""

        def fn(api: Any) -> dict[str, Any]:
            return {
                "full_name": api.get_full_name(),
                "display_name": getattr(api, "display_name", None),
                "unit_system": api.get_unit_system(),
                "timezone": api.get_user_timezone(),
                "primary_training_device": api.get_primary_training_device(),
            }

        return await run(fn)

    @mcp.tool(annotations=READ)
    async def get_devices() -> dict[str, Any]:
        """Garmin devices registered to the account, with battery/firmware info where available."""
        return {"devices": await run(lambda api: api.get_devices())}

    @mcp.tool(annotations=READ)
    async def get_gear() -> dict[str, Any]:
        """Shoes, bikes and other gear tracked in Garmin Connect, with usage where available."""
        return {"gear": await run(lambda api: api.get_gear_list(limit=100))}

    # ---- daily health ------------------------------------------------------

    @mcp.tool(annotations=READ)
    async def get_daily_summary(date: str = "today") -> dict[str, Any]:
        """Daily totals for one day: steps, distance, calories, floors, intensity minutes,
        resting/min/max heart rate, stress averages, Body Battery high/low, SpO2, respiration."""
        d = parse_date(date).isoformat()
        return as_object(await run(lambda api: api.get_user_summary(d)))

    @mcp.tool(annotations=READ)
    async def get_sleep(date: str = "today", include_samples: bool = False) -> dict[str, Any]:
        """Sleep for the night ending on `date`: score, duration, deep/light/REM/awake time,
        overnight HRV, resting HR and Body Battery change. `include_samples` adds the
        per-minute movement/stage/HR series (large)."""
        d = parse_date(date).isoformat()
        return as_object(strip_series(await run(lambda api: api.get_daily_sleep(d)), include_samples))

    @mcp.tool(annotations=READ)
    async def get_heart_rate(date: str = "today", include_samples: bool = False) -> dict[str, Any]:
        """Resting, min and max heart rate for a day, plus the 7-day resting average.
        `include_samples` adds the ~2-minute heart-rate series (large)."""
        d = parse_date(date).isoformat()
        return as_object(strip_series(await run(lambda api: api.get_heart_rates(d)), include_samples))

    @mcp.tool(annotations=READ)
    async def get_stress(date: str = "today", include_samples: bool = False) -> dict[str, Any]:
        """Average/max stress and time in each stress level for a day.
        `include_samples` adds the 3-minute stress and Body Battery series (large)."""
        d = parse_date(date).isoformat()
        return as_object(strip_series(await run(lambda api: api.get_stress_data(d)), include_samples))

    @mcp.tool(annotations=READ)
    async def get_body_battery(start_date: str | None = None, end_date: str = "today") -> dict[str, Any]:
        """Body Battery per day (charged, drained, levels and events). Defaults to the last 7 days."""
        s, e = parse_range(start_date, end_date)
        days = await run(lambda api: api.get_body_battery(s, e))
        return {"days": [strip_series(d, False) for d in days or []]}

    @mcp.tool(annotations=READ)
    async def get_hrv(date: str = "today", include_samples: bool = False) -> dict[str, Any]:
        """Overnight heart-rate variability: last-night average, 7-day average, baseline
        range and status (balanced / unbalanced / low)."""
        d = parse_date(date).isoformat()
        data = await run(lambda api: api.get_hrv_data(d))
        if data is None:
            raise ToolError(f"No HRV data for {d} (HRV needs a compatible watch worn overnight).")
        return as_object(strip_series(data, include_samples))

    # ---- training ----------------------------------------------------------

    @mcp.tool(annotations=READ)
    async def get_training_readiness(date: str = "today") -> dict[str, Any]:
        """Training Readiness score and its contributing factors (sleep, recovery time,
        HRV, acute load, stress history)."""
        d = parse_date(date).isoformat()
        return {"entries": await run(lambda api: api.get_training_readiness(d))}

    @mcp.tool(annotations=READ)
    async def get_training_status(date: str = "today") -> dict[str, Any]:
        """Training status (productive, maintaining, overreaching...), acute/chronic
        load and load balance, and the latest VO2 max."""
        d = parse_date(date).isoformat()
        return as_object(await run(lambda api: api.get_training_status(d)))

    @mcp.tool(annotations=READ)
    async def get_fitness_metrics(date: str = "today") -> dict[str, Any]:
        """VO2 max (running/cycling) and fitness age for a day, plus current race predictions
        (5K, 10K, half, marathon) and endurance score."""
        d = parse_date(date).isoformat()

        def fn(api: Any) -> dict[str, Any]:
            out: dict[str, Any] = {"max_metrics": api.get_max_metrics(d)}
            # Not every account/device has these; one missing shouldn't sink the rest.
            for key, call in (
                ("fitness_age", lambda: api.get_fitnessage_data(d)),
                ("race_predictions", lambda: api.get_race_predictions()),
                ("endurance_score", lambda: api.get_endurance_score(d)),
            ):
                try:
                    out[key] = call()
                except Exception as e:  # noqa: BLE001
                    out[key] = {"unavailable": str(e)}
            return out

        return await run(fn)

    @mcp.tool(annotations=READ)
    async def get_personal_records() -> dict[str, Any]:
        """Personal records (fastest 1K/5K/10K/half/marathon, longest ride, most steps...)."""
        return {"records": await run(lambda api: api.get_personal_record())}

    # ---- activities --------------------------------------------------------

    @mcp.tool(annotations=READ)
    async def list_activities(
        start_date: str | None = None,
        end_date: str = "today",
        activity_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Recent activities, newest first, in a compact form (id, name, type, start time,
        distance_km, duration_min, pace, HR, power, training effect, load).
        With no dates, returns the latest `limit` activities. With dates, returns that range.
        `activity_type` is a Garmin type key such as running, cycling, swimming,
        strength_training, hiking, walking, trail_running."""
        if not 1 <= limit <= 200:
            raise ToolError("limit must be between 1 and 200")
        if start_date is None:
            acts = await run(lambda api: api.get_activities(0, limit, activity_type))
        else:
            s, e = parse_range(start_date, end_date)
            acts = await run(lambda api: api.get_activities_by_date(s, e, activity_type))
            acts = sorted(acts or [], key=lambda a: a.get("startTimeLocal") or "", reverse=True)[:limit]
        return {"activities": [compact_activity(a) for a in acts or []]}

    @mcp.tool(annotations=READ)
    async def get_activity(activity_id: int, include_splits: bool = False) -> dict[str, Any]:
        """Full summary of one activity (from list_activities' activityId): distance, time,
        pace/speed, HR, power, cadence, elevation, training effect, weather.
        `include_splits` adds per-lap splits and time in heart-rate zones."""

        def fn(api: Any) -> dict[str, Any]:
            out: dict[str, Any] = {"activity": api.get_activity(str(activity_id))}
            if include_splits:
                out["splits"] = api.get_activity_splits(str(activity_id))
                out["hr_zones"] = api.get_activity_hr_in_timezones(str(activity_id))
            return out

        return await run(fn)

    # ---- trends & body -----------------------------------------------------

    @mcp.tool(annotations=READ)
    async def get_trends(
        metric: TrendMetric, start_date: str | None = None, end_date: str = "today"
    ) -> dict[str, Any]:
        """One value per day for a metric over a date range (default: last 28 days, max 366).
        Use this for "how has my X changed" questions instead of calling daily tools repeatedly."""
        s, e = parse_range(start_date, end_date, default_days=28)
        calls: dict[str, Callable[[Any], Any]] = {
            "steps": lambda api: api.get_daily_steps(s, e),
            "resting_hr": lambda api: api.get_rhr_daily(s, e),
            "sleep": lambda api: api.get_sleep_daily(s, e),
            "stress": lambda api: api.get_daily_stress_stats(s, e),
            "body_battery": lambda api: api.get_daily_body_battery_stats(s, e),
            "hrv": lambda api: api.get_hrv_data_range(s, e),
            "intensity_minutes": lambda api: api.get_daily_intensity_minutes_stats(s, e),
            "calories": lambda api: api.get_calories_daily(s, e),
            "weight": lambda api: api.get_weigh_ins(s, e),
        }
        return {"metric": metric, "start_date": s, "end_date": e, "data": await run(calls[metric])}

    @mcp.tool(annotations=READ)
    async def get_body_composition(start_date: str | None = None, end_date: str = "today") -> dict[str, Any]:
        """Weight, BMI, body fat, muscle and bone mass from weigh-ins (Index scale or manual).
        Defaults to the last 30 days. Weights are in grams."""
        s, e = parse_range(start_date, end_date, default_days=30)
        return as_object(await run(lambda api: api.get_body_composition(s, e)))

    # ---- calendar ----------------------------------------------------------

    @mcp.tool(annotations=READ)
    async def get_calendar(days_forward: int = 14) -> dict[str, Any]:
        """Upcoming scheduled workouts (incl. training-plan workouts) and race/goal events."""
        if not 1 <= days_forward <= 365:
            raise ToolError("days_forward must be between 1 and 365")
        today = date.today()
        horizon = today + timedelta(days=days_forward)

        def fn(api: Any) -> dict[str, Any]:
            months = sorted({(today.year, today.month), (horizon.year, horizon.month)})
            items: list[dict[str, Any]] = []
            for y, m in months:
                cal = api.get_scheduled_workouts(y, m) or {}
                items.extend(cal.get("calendarItems") or [])
            upcoming = [
                i for i in items
                if today.isoformat() <= (i.get("date") or "") <= horizon.isoformat()
            ]
            upcoming.sort(key=lambda i: i.get("date") or "")
            return {
                "scheduled": upcoming,
                "events": api.get_upcoming_events(days_forward=days_forward),
            }

        return await run(fn)

    # ---- writes (opt-in) ---------------------------------------------------

    if settings.allow_writes:

        @mcp.tool(annotations=WRITE)
        async def log_weight(weight: float, unit: Literal["kg", "lbs"] = "kg") -> dict[str, Any]:
            """Add a weigh-in to Garmin Connect, timestamped now."""
            if not 20 <= weight <= 700:
                raise ToolError("weight looks implausible; expected 20-700")
            return {"result": await run(lambda api: api.add_weigh_in(weight, unitKey=unit))}

        @mcp.tool(annotations=WRITE)
        async def log_hydration(milliliters: float) -> dict[str, Any]:
            """Add water intake for today in ml (negative subtracts)."""
            if not -5000 <= milliliters <= 5000 or milliliters == 0:
                raise ToolError("milliliters must be non-zero and within +/-5000")
            return {"result": await run(lambda api: api.add_hydration_data(milliliters))}

    return mcp
