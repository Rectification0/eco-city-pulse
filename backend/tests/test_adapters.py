"""Source adapters (tasks 2.1-2.3, 2.11).

No test here touches the network. Adapters are split into ``fetch`` (I/O) and
``parse`` (pure), and everything worth asserting lives in ``parse`` — which is
the reason for the split.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from core.config import Settings
from core.exceptions import SchemaValidationError, UploadRejectedError
from services.adapters import registered_specs
from services.adapters.aqi_scale import aqi_to_concentration
from services.adapters.aqicn import AqicnAdapter
from services.adapters.demo import DemoAdapter
from services.adapters.openweather import TEMPERATURE_BOUNDS, OpenWeatherAdapter
from services.adapters.synthetic_traffic import SyntheticTrafficAdapter
from services.adapters.tomtom import (
    OBSERVED_AT_KEY,
    REQUESTED_POINT_KEY,
    TomTomAdapter,
    congestion_index,
)
from services.adapters.upload import UploadAdapter, parse_upload
from services.geo_service import Station

IST = timezone(timedelta(hours=5, minutes=30))
STATION = Station(district_id="new-delhi", lat=28.61, lon=77.21)


# --- Registry ---------------------------------------------------------------


def test_every_registered_source_has_a_unique_name() -> None:
    """Names are the key ``data_sources`` is reconciled on."""
    names = [spec.name for spec in registered_specs()]

    assert len(set(names)) == len(names)


def test_the_three_specified_providers_are_registered() -> None:
    """specs §5.1 names AQICN, OpenWeather and TomTom."""
    names = {spec.name for spec in registered_specs()}

    assert {"AQICN", "OpenWeather", "TomTom Traffic"} <= names


@pytest.mark.parametrize(
    "adapter", [AqicnAdapter(), OpenWeatherAdapter(), TomTomAdapter()]
)
def test_live_adapters_are_unconfigured_without_a_key(adapter: object) -> None:
    """DR-1: no key means the source is simply not switched on."""
    assert adapter.is_configured(Settings(_env_file=None)) is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "adapter", [SyntheticTrafficAdapter(), DemoAdapter(), UploadAdapter([])]
)
def test_offline_adapters_need_no_credential(adapter: object) -> None:
    assert adapter.is_configured(Settings(_env_file=None)) is True  # type: ignore[attr-defined]


def test_a_key_makes_a_live_adapter_configured() -> None:
    settings = Settings(_env_file=None, aqicn_api_key="a-token")

    assert AqicnAdapter().is_configured(settings) is True


# --- AQI conversion ---------------------------------------------------------


@pytest.mark.parametrize(
    ("aqi", "expected"),
    [(0, 0.0), (50, 12.0), (51, 12.1), (100, 35.4), (151, 55.5), (200, 150.4)],
)
def test_aqi_breakpoints_invert_exactly(aqi: float, expected: float) -> None:
    """The EPA scale is piecewise linear, so its endpoints must round-trip."""
    assert aqi_to_concentration(aqi, pollutant="pm25") == pytest.approx(expected)


def test_aqi_interpolates_within_a_segment() -> None:
    # Midway through 151-200 maps midway through 55.5-150.4.
    assert aqi_to_concentration(175.5, pollutant="pm25") == pytest.approx(102.95, abs=0.1)


@pytest.mark.parametrize("aqi", [-1, 501, 10_000])
def test_aqi_outside_the_scale_returns_none(aqi: float) -> None:
    """Beyond 500 the scale is undefined; extrapolating would fabricate a
    measurement rather than report that none is available."""
    assert aqi_to_concentration(aqi, pollutant="pm25") is None


def test_unknown_pollutant_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="breakpoints"):
        aqi_to_concentration(50, pollutant="so2")


# --- AQICN ------------------------------------------------------------------


def _aqicn_payload(**overrides: object) -> dict:
    payload = {
        "status": "ok",
        "data": {
            "city": {"geo": [28.61, 77.21], "name": "Delhi"},
            "iaqi": {
                "pm25": {"v": 155},
                "pm10": {"v": 80},
                "t": {"v": 22.5},
                "h": {"v": 40},
            },
            "time": {"iso": "2026-01-01T17:30:00+05:30"},
        },
    }
    payload.update(overrides)  # type: ignore[arg-type]
    return payload


def test_aqicn_converts_the_index_to_a_concentration() -> None:
    """``iaqi.pm25`` is an AQI index. Storing it as µg/m³ would put two
    different quantities in one column."""
    reading = AqicnAdapter().parse(_aqicn_payload())[0]

    assert reading.pm25 == pytest.approx(63.25, abs=0.01)
    assert reading.pm25 != 155


def test_aqicn_keeps_temperature_and_humidity_in_their_real_units() -> None:
    reading = AqicnAdapter().parse(_aqicn_payload())[0]

    assert reading.temp == 22.5
    assert reading.humidity == 40


def test_aqicn_timestamp_is_converted_to_utc() -> None:
    reading = AqicnAdapter().parse(_aqicn_payload())[0]

    assert reading.timestamp == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def test_aqicn_reassembles_a_split_local_timestamp() -> None:
    """Older payloads give ``time.s`` without an offset and ``time.tz`` beside
    it; treating ``s`` as UTC would shift the reading by hours."""
    payload = _aqicn_payload()
    payload["data"]["time"] = {"s": "2026-01-01 17:30:00", "tz": "+05:30"}  # type: ignore[index]

    assert AqicnAdapter().parse(payload)[0].timestamp == datetime(
        2026, 1, 1, 12, tzinfo=timezone.utc
    )


def test_aqicn_non_ok_status_is_rejected() -> None:
    """On failure AQICN returns a *string* in ``data``; the schema must not
    assume an object is always there."""
    with pytest.raises(SchemaValidationError, match="non-ok"):
        AqicnAdapter().parse({"status": "error", "data": "Unknown station"})


def test_aqicn_payload_without_a_timestamp_is_rejected() -> None:
    payload = _aqicn_payload()
    payload["data"]["time"] = {}  # type: ignore[index]

    with pytest.raises(SchemaValidationError, match="timestamp"):
        AqicnAdapter().parse(payload)


def test_aqicn_missing_pollutants_become_none_not_zero() -> None:
    """A station that reports no PM10 has missing data, not clean air."""
    payload = _aqicn_payload()
    payload["data"]["iaqi"] = {"pm25": {"v": 100}}  # type: ignore[index]

    reading = AqicnAdapter().parse(payload)[0]
    assert reading.pm10 is None
    assert reading.humidity is None


def test_aqicn_garbage_payload_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AqicnAdapter().parse({"unexpected": True})


# --- OpenWeather ------------------------------------------------------------


def _openweather_payload(**main: object) -> dict:
    return {
        "coord": {"lat": 28.61, "lon": 77.21},
        "main": {"temp": 22.5, "humidity": 40, **main},
        "dt": 1767268800,
    }


def test_openweather_parses_a_current_conditions_payload() -> None:
    reading = OpenWeatherAdapter().parse(_openweather_payload())[0]

    assert reading.temp == 22.5
    assert reading.humidity == 40
    assert reading.timestamp == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def test_openweather_carries_no_pollutant_fields() -> None:
    """Weather rows leave the air-quality columns empty rather than zero."""
    reading = OpenWeatherAdapter().parse(_openweather_payload())[0]

    assert reading.pm25 is None
    assert reading.pm10 is None


def test_openweather_rejects_a_kelvin_response() -> None:
    """Forgetting ``units=metric`` gives Kelvin, and a silent 273-degree offset
    would poison every correlation downstream."""
    with pytest.raises(SchemaValidationError, match="Celsius"):
        OpenWeatherAdapter().parse(_openweather_payload(temp=295.65))


@pytest.mark.parametrize("temp", [TEMPERATURE_BOUNDS[0] - 1, TEMPERATURE_BOUNDS[1] + 1])
def test_openweather_rejects_implausible_temperatures(temp: float) -> None:
    with pytest.raises(SchemaValidationError):
        OpenWeatherAdapter().parse(_openweather_payload(temp=temp))


def test_openweather_rejects_impossible_humidity() -> None:
    with pytest.raises(ValidationError):
        OpenWeatherAdapter().parse(_openweather_payload(humidity=140))


# --- TomTom -----------------------------------------------------------------


def _tomtom_payload(**flow: object) -> dict:
    return {
        "flowSegmentData": {"currentSpeed": 30, "freeFlowSpeed": 60, **flow},
        OBSERVED_AT_KEY: "2026-01-01T12:00:00Z",
        REQUESTED_POINT_KEY: [28.61, 77.21],
    }


def test_tomtom_converts_speeds_to_a_congestion_index() -> None:
    reading = TomTomAdapter().parse(_tomtom_payload())[0]

    assert reading.traffic_score == 50.0
    assert reading.timestamp == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("current", "free_flow", "expected"),
    [(60, 60, 0.0), (0, 60, 100.0), (30, 60, 50.0), (70, 60, 0.0)],
)
def test_congestion_index_spans_zero_to_one_hundred(
    current: float, free_flow: float, expected: float
) -> None:
    """A road can exceed its free-flow speed; clamping keeps the score inside
    the range the column and the UI both assume."""
    assert congestion_index(current_speed=current, free_flow_speed=free_flow) == expected


def test_a_closed_road_is_maximum_congestion() -> None:
    assert congestion_index(current_speed=None, free_flow_speed=None, road_closure=True) == 100.0


def test_congestion_index_is_none_when_speeds_are_missing() -> None:
    assert congestion_index(current_speed=None, free_flow_speed=60) is None


def test_a_zero_free_flow_speed_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="free-flow"):
        congestion_index(current_speed=10, free_flow_speed=0)


def test_tomtom_payload_without_the_stamped_time_is_rejected() -> None:
    """``parse`` stays pure: it never reaches for the clock itself."""
    payload = _tomtom_payload()
    del payload[OBSERVED_AT_KEY]

    with pytest.raises(ValidationError):
        TomTomAdapter().parse(payload)


# --- Synthetic traffic fallback (task 2.2) ----------------------------------


def test_synthetic_traffic_produces_a_reading_per_station() -> None:
    adapter = SyntheticTrafficAdapter()
    payloads = list(adapter.fetch(Settings(_env_file=None), stations=[STATION, STATION]))

    assert len(payloads) == 2
    assert all(0 <= p["traffic_score"] <= 100 for p in payloads)


def test_synthetic_traffic_readings_are_on_the_hourly_grid() -> None:
    adapter = SyntheticTrafficAdapter()
    payload = next(iter(adapter.fetch(Settings(_env_file=None), stations=[STATION])))

    reading = adapter.parse(payload)[0]
    assert (reading.timestamp.minute, reading.timestamp.second) == (0, 0)


def test_synthetic_traffic_is_deterministic_for_an_hour() -> None:
    """Idempotent re-ingestion depends on this."""
    adapter = SyntheticTrafficAdapter()
    settings = Settings(_env_file=None)

    first = [p["traffic_score"] for p in adapter.fetch(settings, stations=[STATION])]
    second = [p["traffic_score"] for p in adapter.fetch(settings, stations=[STATION])]

    assert first == second


def test_synthetic_traffic_is_labelled_as_synthetic() -> None:
    """It must never be mistakable for a measurement."""
    assert "Synthetic" in SyntheticTrafficAdapter.spec.name
    payload = next(
        iter(SyntheticTrafficAdapter().fetch(Settings(_env_file=None), stations=[STATION]))
    )
    assert payload["synthetic"] is True


def test_synthetic_traffic_rejects_an_out_of_range_score() -> None:
    """SEC-1 applies to locally generated records too."""
    with pytest.raises(ValidationError):
        SyntheticTrafficAdapter().parse(
            {
                "district_id": "x",
                "lat": 28.61,
                "lon": 77.21,
                "observed_at": "2026-01-01T12:00:00Z",
                "traffic_score": 150,
            }
        )


# --- Upload parsing (task 2.3) ----------------------------------------------


CSV_SAMPLE = b"""timestamp,latitude,longitude,PM2.5,pm10,temperature,rh,traffic
2026-01-01T12:00:00Z,28.61,77.21,55.2,110.4,18.0,60,45
"""


def test_csv_upload_maps_column_aliases() -> None:
    """``PM2.5``/``latitude``/``rh`` all have one canonical meaning; rejecting
    a file over header spelling pushes people into hand-editing data."""
    rows = parse_upload(CSV_SAMPLE, filename="sample.csv")
    reading = UploadAdapter(rows).parse(rows[0])[0]

    assert reading.pm25 == 55.2
    assert reading.pm10 == 110.4
    assert reading.temp == 18.0
    assert reading.humidity == 60
    assert reading.traffic_score == 45
    assert (reading.lat, reading.lon) == (28.61, 77.21)


def test_csv_upload_detects_semicolon_delimiters() -> None:
    content = b"timestamp;lat;lon;pm25\n2026-01-01T12:00:00Z;28.61;77.21;55.2\n"

    rows = parse_upload(content)
    assert UploadAdapter(rows).parse(rows[0])[0].pm25 == 55.2


def test_csv_upload_strips_the_excel_byte_order_mark() -> None:
    content = "﻿timestamp,lat,lon,pm25\n2026-01-01T12:00:00Z,28.61,77.21,9\n".encode()

    rows = parse_upload(content)
    assert UploadAdapter(rows).parse(rows[0])[0].pm25 == 9


@pytest.mark.parametrize("token", ["", "NA", "n/a", "null", "-", "NaN"])
def test_blank_tokens_become_missing_values_not_errors(token: str) -> None:
    """An empty cell is missingness, which Phase 3 exists to handle."""
    content = f"timestamp,lat,lon,pm25,pm10\n2026-01-01T12:00:00Z,28.61,77.21,9,{token}\n"

    rows = parse_upload(content.encode())
    reading = UploadAdapter(rows).parse(rows[0])[0]

    assert reading.pm25 == 9
    assert reading.pm10 is None


def test_json_upload_accepts_a_bare_list() -> None:
    content = b'[{"timestamp": "2026-01-01T12:00:00Z", "lat": 28.61, "lon": 77.21, "pm25": 9}]'

    rows = parse_upload(content)
    assert UploadAdapter(rows).parse(rows[0])[0].pm25 == 9


def test_json_upload_accepts_a_wrapped_list() -> None:
    content = b'{"records": [{"timestamp": "2026-01-01T12:00:00Z", "lat": 1, "lon": 2, "pm25": 9}]}'

    assert len(parse_upload(content)) == 1


def test_format_is_decided_by_content_not_extension() -> None:
    """A browser will happily label a JSON file .csv."""
    content = b'[{"timestamp": "2026-01-01T12:00:00Z", "lat": 1, "lon": 2, "pm25": 9}]'

    rows = parse_upload(content, filename="actually.csv")
    assert rows[0]["pm25"] == 9


def test_naive_upload_timestamps_use_the_stated_offset() -> None:
    """DR-2: the assumption is the caller's, made explicit."""
    content = b"timestamp,lat,lon,pm25\n2026-01-01 17:30:00,28.61,77.21,9\n"
    rows = parse_upload(content)

    reading = UploadAdapter(rows, assume_timezone=IST).parse(rows[0])[0]
    assert reading.timestamp == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"", "empty"),
        (b"   \n ", "empty"),
        (b"\xff\xfe\x00bad", "UTF-8"),
        (b"timestamp,lat,lon\n", "no rows"),
        (b"[", "valid JSON"),
        (b"[1, 2, 3]", "no record objects"),
    ],
)
def test_unusable_uploads_are_rejected_with_a_reason(content: bytes, match: str) -> None:
    with pytest.raises(UploadRejectedError, match=match):
        parse_upload(content)


def test_a_single_json_object_is_a_one_record_upload() -> None:
    """An export of one reading is a legitimate file, not a malformed list."""
    content = b'{"timestamp": "2026-01-01T12:00:00Z", "lat": 1, "lon": 2, "pm25": 9}'

    assert len(parse_upload(content)) == 1


def test_a_row_missing_required_columns_is_a_validation_error() -> None:
    """Per-row failure: the caller quarantines this row and keeps the rest."""
    with pytest.raises(ValidationError):
        UploadAdapter([]).parse({"pm25": 9})


# --- Demo adapter (task 2.9) ------------------------------------------------


def test_demo_adapter_yields_payloads_for_the_requested_window() -> None:
    adapter = DemoAdapter(days=2, end=datetime(2026, 3, 1, tzinfo=timezone.utc))
    payloads = list(adapter.fetch(Settings(_env_file=None), stations=[STATION]))

    assert len(payloads) == 48


def test_demo_payloads_round_trip_through_the_schema() -> None:
    """Demo data must survive the same validation as a live feed (AC-2)."""
    adapter = DemoAdapter(days=1, end=datetime(2026, 3, 1, tzinfo=timezone.utc))
    payloads = list(adapter.fetch(Settings(_env_file=None), stations=[STATION]))

    readings = [reading for p in payloads for reading in adapter.parse(p)]
    assert len(readings) == len(payloads)
    assert all(r.timestamp.tzinfo is timezone.utc for r in readings)


def test_demo_adapter_is_labelled_synthetic() -> None:
    assert "synthetic" in DemoAdapter.spec.name.lower()
