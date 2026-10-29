from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from garminconnect import GarminConnectAuthenticationError
from mcp import Client

from garmin_mcp.config import Settings
from garmin_mcp.garmin import GarminSession
from garmin_mcp.server import build_server, compact_activity, parse_range, strip_series

pytestmark = pytest.mark.anyio

RUN = {
    "activityId": 123,
    "activityName": "Morning Run",
    "activityType": {"typeKey": "running"},
    "startTimeLocal": "2026-10-28 07:00:00",
    "distance": 10000.0,
    "duration": 3000.0,
    "averageSpeed": 10000 / 3000,
    "averageHR": 150,
    "someHugeField": {"x": 1},
}


class FakeGarmin:
    display_name = "runner42"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, tuple]] = []

    def login(self, tokenstore: str | None = None) -> tuple[None, None]:
        return None, None

    def get_full_name(self) -> str:
        return "Test Runner"

    def get_activities(self, start, limit, activitytype=None):
        self.calls.append(("get_activities", (start, limit, activitytype)))
        return [RUN]

    def get_activities_by_date(self, s, e, activitytype=None):
        self.calls.append(("get_activities_by_date", (s, e, activitytype)))
        older = {**RUN, "activityId": 1, "startTimeLocal": "2026-10-01 07:00:00"}
        return [older, RUN]

    def get_heart_rates(self, d):
        return {"restingHeartRate": 48, "heartRateValues": [[i, 60] for i in range(500)]}

    def get_user_summary(self, d):
        return None

    def get_training_status(self, d):
        raise GarminConnectAuthenticationError("401 Unauthorized")

    def add_weigh_in(self, weight, unitKey="kg"):
        self.calls.append(("add_weigh_in", (weight, unitKey)))
        return {"ok": True}


def make(tmp_path: Path, allow_writes: bool = False) -> tuple[Any, FakeGarmin]:
    settings = Settings(token_dir=tmp_path, email=None, password=None, is_cn=False, allow_writes=allow_writes)
    fake = FakeGarmin()
    return build_server(GarminSession(settings, factory=lambda **kw: fake), settings), fake


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def payload(result: Any) -> Any:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_read_only_by_default(tmp_path):
    server, _ = make(tmp_path)
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert "list_activities" in tools and "get_trends" in tools
    assert "log_weight" not in tools
    assert all(t.annotations.read_only_hint for t in tools.values())


async def test_writes_opt_in(tmp_path):
    server, fake = make(tmp_path, allow_writes=True)
    async with Client(server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"log_weight", "log_hydration"} <= names
        res = await c.call_tool("log_weight", {"weight": 72.5})
    assert not res.is_error
    assert ("add_weigh_in", (72.5, "kg")) in fake.calls


async def test_list_activities_compact(tmp_path):
    server, fake = make(tmp_path)
    async with Client(server) as c:
        res = await c.call_tool("list_activities", {"limit": 5, "activity_type": "running"})
    act = payload(res)["activities"][0]
    assert act["type"] == "running"
    assert act["distance_km"] == 10.0 and act["duration_min"] == 50.0
    assert act["pace_min_per_km"] == "5:00"
    assert "someHugeField" not in act
    assert fake.calls[0] == ("get_activities", (0, 5, "running"))


async def test_list_activities_by_date_newest_first(tmp_path):
    server, _ = make(tmp_path)
    async with Client(server) as c:
        res = await c.call_tool("list_activities", {"start_date": "2026-10-01", "end_date": "2026-10-28", "limit": 1})
    assert [a["activityId"] for a in payload(res)["activities"]] == [123]


async def test_series_stripped_unless_requested(tmp_path):
    server, _ = make(tmp_path)
    async with Client(server) as c:
        slim = payload(await c.call_tool("get_heart_rate", {"date": "2026-10-28"}))
        full = payload(await c.call_tool("get_heart_rate", {"date": "2026-10-28", "include_samples": True}))
    assert slim["restingHeartRate"] == 48 and "heartRateValues" not in slim
    assert slim["_omitted_series"] == ["heartRateValues"]
    assert len(full["heartRateValues"]) == 500


async def test_null_result_wrapped(tmp_path):
    server, _ = make(tmp_path)
    async with Client(server) as c:
        res = await c.call_tool("get_daily_summary", {})
    assert not res.is_error and payload(res) == {"result": None}


async def test_auth_error_tells_user_to_login(tmp_path):
    server, _ = make(tmp_path)
    async with Client(server) as c:
        res = await c.call_tool("get_training_status", {"date": "yesterday"})
    assert res.is_error
    assert "garmin-mcp login" in res.content[0].text


async def test_bad_date_is_tool_error(tmp_path):
    server, _ = make(tmp_path)
    async with Client(server) as c:
        res = await c.call_tool("get_sleep", {"date": "last tuesday"})
    assert res.is_error and "YYYY-MM-DD" in res.content[0].text


def test_parse_range_defaults_and_limits():
    end = date.today()
    assert parse_range(None, "today") == ((end - timedelta(days=6)).isoformat(), end.isoformat())
    with pytest.raises(Exception, match="on or before"):
        parse_range("2026-10-10", "2026-10-01")
    with pytest.raises(Exception, match="limited"):
        parse_range("2020-01-01", "2026-01-01")


def test_strip_series_keeps_short_lists():
    data = {"a": [1, 2], "b": list(range(100))}
    assert strip_series(data, False) == {"a": [1, 2], "_omitted_series": ["b"]}


def test_compact_activity_handles_missing_fields():
    assert compact_activity({"activityId": 1}) == {"activityId": 1}
