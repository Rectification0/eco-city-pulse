"""Demo dataset generation (task 1.9, DR-1 … DR-4, AC-2).

The exit criterion for Phase 1 is that the demo seed loads with no network
connection, so that is what ``test_generation_needs_no_network`` enforces
literally, by removing the socket module's ability to open one.
"""

from __future__ import annotations

import socket
import statistics
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest

from services import demo_data
from services.demo_data import Station, generate_observations

END = datetime(2026, 3, 1, tzinfo=timezone.utc)

STATIONS = (
    Station(district_id="north-delhi", lat=28.805, lon=77.17),
    Station(district_id="south-delhi", lat=28.5, lon=77.175),
)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make any outbound connection attempt fail loudly (AC-2)."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Demo mode must not touch the network (DR-1).")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield


def _records(days: int = 7, **kwargs: object) -> list[demo_data.DemoObservation]:
    return list(
        generate_observations(days=days, end=END, stations=STATIONS, **kwargs)  # type: ignore[arg-type]
    )


def test_generation_needs_no_network(no_network: None) -> None:
    """AC-2: the whole demo dataset is produced offline."""
    records = _records(days=3)

    assert len(records) == 3 * 24 * len(STATIONS)


def test_row_count_is_hourly_per_station() -> None:
    assert len(_records(days=7)) == 7 * 24 * len(STATIONS)


def test_timestamps_are_utc_and_on_the_hour() -> None:
    """DR-2 and DR-4 together: aware, UTC, and exactly on the hourly grid."""
    for record in _records(days=2):
        assert record.timestamp.tzinfo is not None
        assert record.timestamp.utcoffset() == timedelta(0)
        assert (record.timestamp.minute, record.timestamp.second) == (0, 0)


def test_each_station_series_is_contiguous_and_ascending() -> None:
    records = _records(days=5)
    series = [r for r in records if r.district_id == "north-delhi"]

    gaps = {
        b.timestamp - a.timestamp for a, b in zip(series, series[1:], strict=False)
    }
    assert gaps == {timedelta(hours=1)}


def test_coordinates_are_decimal_degrees() -> None:
    """DR-3. Radians or a lat/lon transposition would fail this."""
    for record in _records(days=2):
        assert -90 <= record.lat <= 90
        assert -180 <= record.lon <= 180


def test_values_for_an_hour_do_not_depend_on_the_window() -> None:
    """A short run and a long run must agree wherever they overlap.

    This is what makes re-seeding safe: the same hour always yields the same
    row, so ON CONFLICT DO NOTHING skips a duplicate rather than hiding a
    divergent value.
    """
    short = {(r.district_id, r.timestamp): r for r in _records(days=3)}
    long = {(r.district_id, r.timestamp): r for r in _records(days=30)}

    overlap = short.keys() & long.keys()
    assert len(overlap) == 3 * 24 * len(STATIONS)
    assert all(short[key] == long[key] for key in overlap)


def test_a_different_seed_produces_a_different_dataset() -> None:
    """Otherwise the seed argument would be decorative."""
    default = [r.pm25 for r in _records(days=3)]
    alternate = [r.pm25 for r in _records(days=3, seed=7)]

    assert default != alternate


def test_generation_before_the_epoch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at or after"):
        list(
            demo_data.generate_station_series(
                STATIONS[0],
                start=demo_data.DEMO_EPOCH - timedelta(hours=1),
                end=demo_data.DEMO_EPOCH + timedelta(hours=1),
            )
        )


def test_zero_days_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(generate_observations(days=0, end=END, stations=STATIONS))


# --- Structure the later phases depend on -----------------------------------


def test_dataset_contains_missing_values() -> None:
    """Phase 3 has nothing to impute if the demo data is complete (FEAT-03)."""
    records = _records(days=30)

    for field in ("pm25", "pm10", "temp", "humidity", "traffic_score"):
        missing = sum(1 for r in records if getattr(r, field) is None)
        assert missing > 0, f"{field} has no missing values to analyse"


def test_missingness_is_not_uniformly_random() -> None:
    """specs §5.3: the MCAR/MAR/MNAR judgement must be a real one.

    PM10 drops out far more often when PM2.5 is high -- missingness driven by an
    observed variable, i.e. MAR, which is what justifies MICE over a mean fill.
    """
    records = _records(days=120)
    high = [r for r in records if r.pm25 is not None and r.pm25 > 150]
    low = [r for r in records if r.pm25 is not None and r.pm25 <= 150]

    assert high, "the dataset should contain heavily polluted hours"
    high_rate = sum(1 for r in high if r.pm10 is None) / len(high)
    low_rate = sum(1 for r in low if r.pm10 is None) / len(low)

    assert high_rate > low_rate * 2


def test_dataset_contains_contiguous_gaps() -> None:
    """Forward/backward fill (3.3) targets runs, not scattered single holes."""
    series = [r for r in _records(days=60) if r.district_id == "north-delhi"]

    longest = current = 0
    for record in series:
        current = current + 1 if record.pm25 is None else 0
        longest = max(longest, current)

    assert longest >= 4


def test_pm25_is_autocorrelated() -> None:
    """Lag features (5.2) and the naive baseline (7.5) both rely on this."""
    series = [
        r.pm25
        for r in _records(days=30)
        if r.district_id == "north-delhi" and r.pm25 is not None
    ]
    pairs = [(a, b) for a, b in zip(series, series[1:], strict=False)]

    mean = statistics.fmean(series)
    covariance = statistics.fmean([(a - mean) * (b - mean) for a, b in pairs])
    variance = statistics.pvariance(series, mu=mean)

    assert covariance / variance > 0.5


def test_pm25_tracks_traffic_and_opposes_temperature() -> None:
    """Gives the correlation matrix (4.2) and PCA (6.2) something to report."""
    records = [
        r
        for r in _records(days=45)
        if None not in (r.pm25, r.traffic_score, r.temp)
    ]

    def correlation(xs: list[float], ys: list[float]) -> float:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        cov = statistics.fmean([(x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)])
        return cov / (statistics.pstdev(xs, mu=mx) * statistics.pstdev(ys, mu=my))

    pm25 = [r.pm25 for r in records]  # type: ignore[misc]
    assert correlation(pm25, [r.traffic_score for r in records]) > 0  # type: ignore[misc]
    assert correlation(pm25, [r.temp for r in records]) < 0  # type: ignore[misc]


def test_seasonal_swing_is_present() -> None:
    """STL decomposition (4.5) needs a seasonal component to separate."""
    winter = [
        r.pm25
        for r in generate_observations(
            days=20, end=datetime(2026, 1, 20, tzinfo=timezone.utc), stations=STATIONS
        )
        if r.pm25 is not None
    ]
    monsoon = [
        r.pm25
        for r in generate_observations(
            days=20, end=datetime(2026, 8, 20, tzinfo=timezone.utc), stations=STATIONS
        )
        if r.pm25 is not None
    ]

    assert statistics.fmean(winter) > 2 * statistics.fmean(monsoon)


def test_a_spatial_gradient_exists_across_stations() -> None:
    """Task 10.4 renders a gradient; a flat city would render nothing."""
    records = _records(days=30)
    by_station = {
        station.district_id: statistics.fmean(
            [
                r.pm25
                for r in records
                if r.district_id == station.district_id and r.pm25 is not None
            ]
        )
        for station in STATIONS
    }

    assert max(by_station.values()) - min(by_station.values()) > 5


def test_values_stay_physically_plausible() -> None:
    records = _records(days=60)

    for record in records:
        if record.pm25 is not None:
            assert 0 < record.pm25 < 1000
        if record.humidity is not None:
            assert 0 <= record.humidity <= 100
        if record.traffic_score is not None:
            assert 0 <= record.traffic_score <= 100
        if record.pm10 is not None and record.pm25 is not None:
            assert record.pm10 > record.pm25  # coarse fraction includes the fine


def test_stations_come_from_the_district_boundaries() -> None:
    """Task 1.8 feeds 1.9: seeded points land inside the polygons on the map."""
    stations = demo_data.stations_from_districts()

    assert len(stations) == 11
    assert all(28.40 <= s.lat <= 28.89 for s in stations)
    assert all(76.84 <= s.lon <= 77.35 for s in stations)
