"""Harmonization (tasks 2.4-2.6, 2.11; DR-2 … DR-4).

Three of the four requirements this phase exists to satisfy are single
functions, so this is where most of Phase 2's correctness is pinned down.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.exceptions import HarmonizationError
from services.harmonizer import (
    MEASUREMENT_FIELDS,
    SourceReading,
    floor_to_hour,
    harmonize,
    harmonize_coordinates,
    iter_normalized,
    resample_hourly,
    to_decimal_degrees,
    to_utc,
)

IST = timezone(timedelta(hours=5, minutes=30))
UTC_NOON = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


# --- DR-2: timestamps -------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 17, 30, tzinfo=IST),
        "2026-01-01T12:00:00+00:00",
        "2026-01-01T17:30:00+05:30",
        "2026-01-01T12:00:00Z",
        "2026-01-01T12:00:00z",
        1767268800,
        1767268800.0,
        "1767268800",
    ],
)
def test_every_supported_representation_lands_on_the_same_instant(value: object) -> None:
    """DR-2: one clock, whatever the source chose to speak."""
    assert to_utc(value) == UTC_NOON  # type: ignore[arg-type]


def test_result_is_always_utc_not_merely_aware() -> None:
    converted = to_utc("2026-01-01T17:30:00+05:30")

    assert converted.utcoffset() == timedelta(0)
    assert converted.tzinfo is timezone.utc


def test_naive_timestamps_use_the_stated_assumption() -> None:
    """The caller states the zone; nothing here guesses (DR-2)."""
    assert to_utc(datetime(2026, 1, 1, 17, 30), assume_timezone=IST) == UTC_NOON


def test_naive_timestamps_default_to_utc() -> None:
    assert to_utc("2026-01-01T12:00:00") == UTC_NOON


@pytest.mark.parametrize(
    "value", ["not-a-time", "", "   ", "2026-13-45T99:00:00", float("nan"), float("inf")]
)
def test_unparseable_timestamps_are_rejected(value: object) -> None:
    with pytest.raises(HarmonizationError):
        to_utc(value)  # type: ignore[arg-type]


def test_a_boolean_is_not_a_timestamp() -> None:
    """bool is a subclass of int, so True would otherwise read as epoch 1."""
    with pytest.raises(HarmonizationError, match="boolean"):
        to_utc(True)


def test_floor_to_hour_puts_readings_on_the_grid() -> None:
    assert floor_to_hour(datetime(2026, 1, 1, 12, 59, 59, 999, tzinfo=timezone.utc)) == UTC_NOON


# --- DR-3: coordinates ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "axis", "expected"),
    [
        (28.61, "lat", 28.61),
        ("28.61", "lat", 28.61),
        ("28.61N", "lat", 28.61),
        ("28.61 N", "lat", 28.61),
        ("28.61S", "lat", -28.61),
        ("-28.61", "lat", -28.61),
        ("28°36'36\"N", "lat", 28.61),
        ("28 36 36 N", "lat", 28.61),
        (77.21, "lon", 77.21),
        ("77.21E", "lon", 77.21),
        ("77.21W", "lon", -77.21),
        ("0", "lat", 0.0),
    ],
)
def test_coordinate_formats_convert_to_decimal_degrees(
    value: object, axis: str, expected: float
) -> None:
    """DR-3. DMS, hemisphere suffixes and plain floats all mean one thing."""
    assert to_decimal_degrees(value, axis=axis) == pytest.approx(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "axis"),
    [(91.0, "lat"), (-91.0, "lat"), (181.0, "lon"), (-181.0, "lon")],
)
def test_out_of_range_coordinates_are_rejected(value: float, axis: str) -> None:
    """A range check catches radians and transpositions that leave the range.

    It cannot catch radians that happen to stay inside it -- 0.4993 rad is a
    perfectly valid latitude -- and nothing at this layer could. The defence
    against that is the district polygons the seeded stations come from.
    """
    with pytest.raises(HarmonizationError, match="outside"):
        to_decimal_degrees(value, axis=axis)


def test_a_latitude_marked_east_is_rejected() -> None:
    """The commonest coordinate bug there is: the pair arrives transposed."""
    with pytest.raises(HarmonizationError, match="swapped"):
        to_decimal_degrees("77.21E", axis="lat")


def test_a_negative_value_that_is_also_marked_south_is_rejected() -> None:
    """-28.61S is ambiguous: it means +28.61 read one way, -28.61 the other."""
    with pytest.raises(HarmonizationError, match="negative"):
        to_decimal_degrees("-28.61S", axis="lat")


@pytest.mark.parametrize("value", ["nonsense", "28 61 00", "28 30 99", ""])
def test_unparseable_coordinates_are_rejected(value: str) -> None:
    with pytest.raises(HarmonizationError):
        to_decimal_degrees(value, axis="lat")


def test_coordinates_are_rounded_to_a_stable_precision() -> None:
    """Live feeds jitter the last digits; without rounding, one sensor would
    resample into several 'locations' and fragment the hourly grid."""
    lat, lon = harmonize_coordinates(28.6123456789, 77.2187654321)

    assert lat == 28.61235
    assert lon == 77.21877


def test_an_invalid_axis_is_a_programming_error_not_a_data_error() -> None:
    with pytest.raises(ValueError, match="axis"):
        to_decimal_degrees(1.0, axis="altitude")


# --- DR-4: hourly resample --------------------------------------------------


def _reading(minute: int, **values: float | None) -> SourceReading:
    return SourceReading(
        timestamp=datetime(2026, 1, 1, 12, minute, tzinfo=timezone.utc),
        lat=28.61,
        lon=77.21,
        **values,
    )


def test_readings_within_an_hour_are_averaged() -> None:
    """DR-4: mean aggregation within the hour."""
    resampled = resample_hourly([_reading(5, pm25=10.0), _reading(35, pm25=20.0)])

    assert len(resampled) == 1
    assert resampled[0].timestamp == UTC_NOON
    assert resampled[0].pm25 == 15.0


def test_each_field_averages_over_the_values_it_actually_has() -> None:
    """A missing temperature in one sample must not discard that hour's PM2.5,
    nor drag the temperature average toward zero."""
    resampled = resample_hourly(
        [_reading(5, pm25=10.0, temp=None), _reading(35, pm25=20.0, temp=30.0)]
    )

    assert resampled[0].pm25 == 15.0
    assert resampled[0].temp == 30.0


def test_a_field_with_no_values_at_all_stays_none() -> None:
    resampled = resample_hourly([_reading(5, pm25=10.0)])

    assert resampled[0].humidity is None


def test_different_hours_stay_separate() -> None:
    resampled = resample_hourly(
        [
            _reading(5, pm25=10.0),
            SourceReading(
                timestamp=datetime(2026, 1, 1, 13, 5, tzinfo=timezone.utc),
                lat=28.61,
                lon=77.21,
                pm25=40.0,
            ),
        ]
    )

    assert [r.pm25 for r in resampled] == [10.0, 40.0]


def test_different_locations_are_never_averaged_together() -> None:
    """Averaging across stations would smear the city's spatial gradient --
    the thing the map exists to show -- into one meaningless number."""
    resampled = resample_hourly(
        [
            _reading(5, pm25=10.0),
            SourceReading(timestamp=UTC_NOON, lat=28.80, lon=77.21, pm25=90.0),
        ]
    )

    assert len(resampled) == 2
    assert sorted(r.pm25 for r in resampled) == [10.0, 90.0]  # type: ignore[misc]


def test_output_is_sorted_deterministically() -> None:
    unsorted = [
        SourceReading(timestamp=UTC_NOON, lat=28.80, lon=77.21, pm25=1.0),
        SourceReading(timestamp=UTC_NOON, lat=28.50, lon=77.21, pm25=2.0),
        SourceReading(
            timestamp=datetime(2026, 1, 1, 11, tzinfo=timezone.utc),
            lat=28.90,
            lon=77.21,
            pm25=3.0,
        ),
    ]

    resampled = resample_hourly(unsorted)

    assert [(r.timestamp, r.lat) for r in resampled] == sorted(
        (r.timestamp, r.lat) for r in resampled
    )


def test_resampling_is_idempotent() -> None:
    """Already-hourly data must survive a second pass unchanged, which is what
    lets demo data go through the same pipeline as a live feed."""
    once = resample_hourly([_reading(5, pm25=10.0), _reading(35, pm25=20.0)])

    assert resample_hourly(once) == once


def test_harmonize_drops_readings_with_no_measurements() -> None:
    """A row of nothing but a coordinate would show up later as a phantom
    observation that is 100% missing."""
    empty = SourceReading(timestamp=UTC_NOON, lat=28.61, lon=77.21)

    assert harmonize([empty]) == []
    assert harmonize([empty], drop_empty=False) != []


def test_measurement_fields_match_the_reading_shape() -> None:
    """Guards the one place the resampler and the write path could drift."""
    reading = SourceReading(timestamp=UTC_NOON, lat=0.0, lon=0.0)

    assert set(reading.measurements()) == set(MEASUREMENT_FIELDS)


def test_iter_normalized_applies_both_conversions() -> None:
    raw = SourceReading(
        timestamp=datetime(2026, 1, 1, 17, 30),  # naive
        lat="28.61N",  # type: ignore[arg-type]
        lon="77.21E",  # type: ignore[arg-type]
        pm25=10.0,
    )

    normalized = list(iter_normalized([raw], assume_timezone=IST))[0]

    assert normalized.timestamp == UTC_NOON
    assert (normalized.lat, normalized.lon) == (28.61, 77.21)


def test_floor_to_hour_converts_before_truncating() -> None:
    """Flooring first would truncate in the source's own zone: 17:30+05:30
    would become 17:00 local = 11:30 UTC, which is not on the grid at all and
    would shard one hour into two buckets."""
    assert floor_to_hour(datetime(2026, 1, 1, 17, 30, tzinfo=IST)) == UTC_NOON


def test_floor_to_hour_rejects_naive_input() -> None:
    """``astimezone`` on a naive value silently uses the *server's* zone, which
    makes the stored offset depend on where the container runs."""
    with pytest.raises(HarmonizationError, match="naive"):
        floor_to_hour(datetime(2026, 1, 1, 12))


def test_resampling_groups_by_the_utc_hour_across_zones() -> None:
    """Two readings 30 minutes apart in different notations are one hour."""
    resampled = resample_hourly(
        [
            SourceReading(
                timestamp=datetime(2026, 1, 1, 17, 40, tzinfo=IST), lat=1.0, lon=2.0, pm25=10.0
            ),
            SourceReading(
                timestamp=datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc),
                lat=1.0,
                lon=2.0,
                pm25=20.0,
            ),
        ]
    )

    assert len(resampled) == 1
    assert resampled[0].timestamp == UTC_NOON
    assert resampled[0].pm25 == 15.0
