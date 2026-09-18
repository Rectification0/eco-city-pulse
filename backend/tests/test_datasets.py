"""The pandas boundary (``services/datasets.py``).

Small but load-bearing: every phase from here on builds on the shape this
module produces, and two of its guarantees -- UTC timestamps and per-station
sorting -- are assumptions the quality engine, the feature builder and the
model trainer all make silently.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from services import datasets
from services.datasets import MEASUREMENT_COLUMNS, STATION_COLUMN

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _raw(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": 1,
        "source_id": 1,
        "timestamp": START,
        "lat": 28.61,
        "lon": 77.21,
        "pm25": 50.0,
        "pm10": 110.0,
        "temp": 22.0,
        "humidity": 60.0,
        "traffic_score": 40.0,
        "is_anomaly": False,
    }
    row.update(overrides)
    return row


def test_the_station_key_is_stable_for_one_location() -> None:
    assert datasets.station_key(28.61, 77.21) == datasets.station_key(28.610000, 77.210000)


def test_different_locations_get_different_keys() -> None:
    assert datasets.station_key(28.61, 77.21) != datasets.station_key(28.62, 77.21)


def test_measurement_columns_are_numeric_even_when_all_null() -> None:
    """SQL NULL arrives as None; an object column would make every later
    ``isna()`` and ``mean()`` quietly do the wrong thing."""
    frame = datasets.prepare_frame(
        pd.DataFrame([_raw(pm25=None, pm10=None, temp=None, humidity=None)])
    )

    for column in MEASUREMENT_COLUMNS:
        assert frame[column].dtype == "float64"


def test_timestamps_are_timezone_aware_utc() -> None:
    frame = datasets.prepare_frame(pd.DataFrame([_raw()]))

    assert str(frame["timestamp"].dt.tz) == "UTC"


def test_rows_are_sorted_by_station_then_time() -> None:
    """Every downstream step assumes a per-station series in chronological
    order; sorting once here means none of them has to remember."""
    rows = [
        _raw(id=1, timestamp=START + timedelta(hours=2), lat=28.70),
        _raw(id=2, timestamp=START, lat=28.70),
        _raw(id=3, timestamp=START + timedelta(hours=1), lat=28.60),
    ]

    frame = datasets.prepare_frame(pd.DataFrame(rows))

    assert frame[STATION_COLUMN].is_monotonic_increasing
    for _, group in frame.groupby(STATION_COLUMN):
        assert group["timestamp"].is_monotonic_increasing


def test_a_station_column_is_added() -> None:
    frame = datasets.prepare_frame(pd.DataFrame([_raw()]))

    assert frame.loc[0, STATION_COLUMN] == datasets.station_key(28.61, 77.21)


def test_a_null_anomaly_flag_becomes_false() -> None:
    frame = datasets.prepare_frame(pd.DataFrame([_raw(is_anomaly=None)]))

    assert frame.loc[0, "is_anomaly"] is False or frame.loc[0, "is_anomaly"] == False  # noqa: E712


def test_preparing_an_empty_frame_yields_the_right_columns() -> None:
    columns = [*datasets.IDENTITY_COLUMNS, *MEASUREMENT_COLUMNS, "is_anomaly"]

    frame = datasets.prepare_frame(pd.DataFrame(columns=columns))

    assert frame.empty
    assert STATION_COLUMN in frame.columns


def test_describe_window_reports_the_slice() -> None:
    rows = [
        _raw(id=1, timestamp=START),
        _raw(id=2, timestamp=START + timedelta(hours=5), lat=28.70),
    ]
    frame = datasets.prepare_frame(pd.DataFrame(rows))

    window = datasets.describe_window(frame)

    assert window.rows == 2
    assert window.stations == 2
    assert window.start == START
    assert window.end == START + timedelta(hours=5)


def test_describe_window_of_an_empty_frame_is_zeroed() -> None:
    frame = datasets.prepare_frame(
        pd.DataFrame(columns=[*datasets.IDENTITY_COLUMNS, *MEASUREMENT_COLUMNS, "is_anomaly"])
    )

    window = datasets.describe_window(frame)

    assert window.rows == 0
    assert window.start is None


def test_require_rows_raises_a_domain_error() -> None:
    """A domain error, not whatever numpy raises three frames later."""
    frame = datasets.prepare_frame(pd.DataFrame([_raw()]))

    with pytest.raises(InsufficientDataError) as caught:
        datasets.require_rows(frame, minimum=10, what="STL decomposition")

    assert caught.value.status_code == 422
    assert caught.value.details["available"] == 1


def test_require_rows_passes_when_there_is_enough() -> None:
    frame = datasets.prepare_frame(pd.DataFrame([_raw(), _raw(id=2)]))

    datasets.require_rows(frame, minimum=2, what="anything")


def test_the_window_serialises_to_primitives() -> None:
    import json

    frame = datasets.prepare_frame(pd.DataFrame([_raw()]))

    assert json.loads(json.dumps(datasets.describe_window(frame).as_dict()))["rows"] == 1
