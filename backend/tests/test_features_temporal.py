"""Temporal features (task 5.1, specs §6.1).

The assertions that matter are about the *clock*, not the arithmetic. Deriving
these features in UTC is the easy mistake -- everything still runs, every value
is in range, and the rush-hour peak quietly lands five and a half hours from
where it happened.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from services.datasets import prepare_frame
from services.features import temporal
from services.features.spec import SEASONS, TEMPORAL_FEATURES
from tests.test_features_windows import build_frame

# 2026-01-01 is a Thursday.
THURSDAY = datetime(2026, 1, 1, tzinfo=timezone.utc)


def frame_at(*timestamps: datetime) -> pd.DataFrame:
    records = [
        {
            "id": index + 1,
            "source_id": 1,
            "timestamp": timestamp,
            "lat": 28.6,
            "lon": 77.2,
            "pm25": 50.0,
            "pm10": 90.0,
            "temp": 20.0,
            "humidity": 55.0,
            "traffic_score": 40.0,
            "is_anomaly": False,
        }
        for index, timestamp in enumerate(timestamps)
    ]
    return prepare_frame(pd.DataFrame(records))


# --- Local time -------------------------------------------------------------


def test_the_hour_is_the_local_hour_not_the_utc_one() -> None:
    """Midnight UTC is 05:30 in the modelled city, and 5 is the honest answer."""
    result = temporal.add_temporal(frame_at(THURSDAY))

    assert result["hour_of_day"].iloc[0] == 5


def test_a_late_utc_evening_belongs_to_the_next_local_day() -> None:
    """Friday 20:00 UTC is Saturday 01:30 IST -- a weekend hour, and a model
    told otherwise would learn a weekend rhythm on a weekday."""
    friday_night = THURSDAY + timedelta(days=1, hours=20)

    result = temporal.add_temporal(frame_at(friday_night))

    assert result["day_of_week"].iloc[0] == 5
    assert result["is_weekend"].iloc[0] == 1


def test_the_offset_is_carried_by_the_caller_not_hardcoded() -> None:
    result = temporal.add_temporal(frame_at(THURSDAY), offset_minutes=0)

    assert result["hour_of_day"].iloc[0] == 0


def test_a_missing_timestamp_is_refused_rather_than_guessed() -> None:
    frame = frame_at(THURSDAY)
    frame.loc[0, "timestamp"] = pd.NaT

    with pytest.raises(ValueError, match="timestamp"):
        temporal.add_temporal(frame)


# --- Calendar features ------------------------------------------------------


def test_every_temporal_feature_is_produced() -> None:
    result = temporal.add_temporal(build_frame(rows=48))

    for name in TEMPORAL_FEATURES:
        assert name in result.columns
        assert result[name].notna().all()


def test_weekday_and_weekend_are_distinguished() -> None:
    week = frame_at(*(THURSDAY + timedelta(days=day) for day in range(7)))

    result = temporal.add_temporal(week)

    # Thu, Fri, Sat, Sun, Mon, Tue, Wed -> two weekend days.
    assert result["is_weekend"].sum() == 2
    assert list(result["day_of_week"]) == [3, 4, 5, 6, 0, 1, 2]


@pytest.mark.parametrize(
    ("month", "expected"),
    [
        (1, "winter"),
        (2, "winter"),
        (3, "summer"),
        (5, "summer"),
        (6, "monsoon"),
        (9, "monsoon"),
        (10, "post_monsoon"),
        (11, "post_monsoon"),
        (12, "winter"),
    ],
)
def test_seasons_follow_the_regional_calendar(month: int, expected: str) -> None:
    """Post-monsoon (Oct-Nov) is the stubble-burning window that dominates
    North Indian PM2.5; a Sep-Nov "autumn" would split it in two."""
    # Midday local, so the offset cannot push the row into an adjacent month.
    timestamp = datetime(2026, month, 15, 6, tzinfo=timezone.utc)

    result = temporal.add_temporal(frame_at(timestamp))

    assert temporal.season_label(result["season"].iloc[0]) == expected


def test_every_month_maps_to_a_known_season() -> None:
    months = [datetime(2026, month, 15, 6, tzinfo=timezone.utc) for month in range(1, 13)]

    result = temporal.add_temporal(frame_at(*months))

    assert set(result["season"]) == set(range(len(SEASONS)))


def test_temporal_features_use_only_the_row_they_describe() -> None:
    """No window, no neighbour: rewriting every measurement changes nothing."""
    frame = build_frame(rows=72)
    mutated = frame.copy()
    mutated["pm25"] = 9999.0

    pd.testing.assert_frame_equal(
        temporal.add_temporal(frame)[list(TEMPORAL_FEATURES)],
        temporal.add_temporal(mutated)[list(TEMPORAL_FEATURES)],
    )


def test_an_empty_frame_still_gets_the_columns() -> None:
    result = temporal.add_temporal(build_frame(rows=0))

    for name in TEMPORAL_FEATURES:
        assert name in result.columns


# --- Cyclical encoding (offered to Phase 7, not in the default set) ---------


def test_cyclical_encoding_puts_hour_23_next_to_hour_0() -> None:
    hours = pd.Series([0, 23])

    sin, cos = temporal.cyclical(hours, 24)
    distance = float(((sin[0] - sin[1]) ** 2 + (cos[0] - cos[1]) ** 2) ** 0.5)

    assert distance < 0.3  # adjacent, where the raw integers are 23 apart
