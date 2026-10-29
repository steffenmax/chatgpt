"""Pull one run from Garmin Connect into a JSON file for render.mjs.

    uv run python share-card/fetch_run.py                 # latest run today
    uv run python share-card/fetch_run.py --date 2026-10-29
    uv run python share-card/fetch_run.py --id 123456789
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from garmin_mcp.config import Settings
from garmin_mcp.garmin import GarminSession


def pick_activity(api, day: str, activity_id: int | None):
    if activity_id:
        return activity_id
    acts = api.get_activities_by_date(day, day) or []
    runs = [a for a in acts if "run" in ((a.get("activityType") or {}).get("typeKey") or "")]
    if not runs:
        sys.exit(f"No runs found on {day}. Activities that day: {[a.get('activityName') for a in acts]}")
    runs.sort(key=lambda a: a.get("startTimeLocal") or "")
    return runs[-1]["activityId"]


def fetch(api, activity_id: int) -> dict:
    act = api.get_activity(str(activity_id))
    s = act.get("summaryDTO") or {}
    details = api.get_activity_details(str(activity_id), maxchart=10, maxpoly=1500) or {}
    poly = ((details.get("geoPolylineDTO") or {}).get("polyline")) or []
    splits = api.get_activity_splits(str(activity_id)) or {}
    laps = [
        {
            "distance_m": lap.get("distance"),
            "duration_s": lap.get("duration"),
            "avg_speed": lap.get("averageSpeed"),
            "avg_hr": lap.get("averageHR"),
        }
        for lap in splits.get("lapDTOs") or []
    ]
    return {
        "activity_id": activity_id,
        "name": act.get("activityName"),
        "type": (act.get("activityTypeDTO") or {}).get("typeKey"),
        "location": act.get("locationName"),
        "start_local": s.get("startTimeLocal"),
        "unit_system": api.get_unit_system(),
        "distance_m": s.get("distance"),
        "duration_s": s.get("duration"),
        "moving_s": s.get("movingDuration"),
        "avg_speed": s.get("averageSpeed"),
        "avg_hr": s.get("averageHR"),
        "max_hr": s.get("maxHR"),
        "elev_gain_m": s.get("elevationGain"),
        "cadence": s.get("averageRunCadence"),
        "calories": s.get("calories"),
        "aerobic_te": s.get("trainingEffect"),
        "route": [[p["lat"], p["lon"]] for p in poly if p.get("lat") is not None],
        "laps": laps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--id", type=int)
    ap.add_argument("--out", default="share-card/run.json")
    args = ap.parse_args()
    session = GarminSession(Settings.from_env())
    data = session.call_sync(lambda api: fetch(api, pick_activity(api, args.date, args.id)))
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.out}: {data['name']} ({data['distance_m']} m, {len(data['route'])} GPS points)")


if __name__ == "__main__":
    main()
