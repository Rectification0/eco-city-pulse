"""STL decomposition (task 4.5, Module 4).

The fixtures are built from a known trend plus a known daily cycle, so the test
can check that STL recovered *those* rather than merely returning three arrays
of the right length.

Every test skips cleanly when statsmodels cannot load. That is not theoretical:
its STL is a compiled extension, and this suite has already met a machine where
an application-control policy blocked the DLL.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from services.datasets import MEASUREMENT_COLUMNS, prepare_frame
from services.eda import decomposition

pytestmark = pytest.mark.skipif(
    not decomposition.is_available(),
    reason="statsmodels could not be loaded in this environment",
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def seasonal_frame(
    *,
    hours: int = 24 * 30,
    slope: float = 0.02,
    amplitude: float = 20.0,
    noise: float = 1.0,
    stations: int = 1,
    seed: int = 5,
) -> pd.DataFrame:
    """A series with a known linear trend and a known 24-hour cycle."""
    rng = np.random.default_rng(seed)
    records = []
    for station in range(stations):
        for hour in range(hours):
            value = (
                60.0
                + slope * hour
                + amplitude * math.sin(2 * math.pi * hour / 24)
                + rng.normal(0, noise)
            )
            records.append(
                {
                    "id": station * hours + hour + 1,
                    "source_id": 1,
                    "timestamp": START + timedelta(hours=hour),
                    "lat": 28.61 + station * 0.1,
                    "lon": 77.21,
                    "pm25": value,
                    "pm10": value * 2,
                    "temp": 22.0,
                    "humidity": 55.0,
                    "traffic_score": 40.0,
                    "is_anomaly": False,
                }
            )
    return prepare_frame(pd.DataFrame(records))


# --- The components ---------------------------------------------------------


def test_the_components_add_back_up_to_the_observed_series() -> None:
    """STL's defining identity. If this fails, nothing else here means much.

    The tolerance is 2e-4 rather than machine epsilon because the components
    are rounded to four decimals on the way out, to keep the JSON payload
    small. Three independently rounded values can differ from their unrounded
    sum by up to 1.5e-4.
    """
    result = decomposition.decompose(seasonal_frame(), max_points=None)

    reconstructed = [
        t + s + r
        for t, s, r in zip(result.trend, result.seasonal, result.residual, strict=True)
    ]
    assert reconstructed == pytest.approx(list(result.observed), abs=2e-4)


def test_the_known_trend_is_recovered() -> None:
    """Injected slope 0.02/hour over 30 days is about +14 end to end."""
    result = decomposition.decompose(seasonal_frame(slope=0.02), max_points=None)

    rise = result.trend[-1] - result.trend[0]
    assert rise == pytest.approx(0.02 * (24 * 30 - 1), rel=0.25)


def test_the_known_daily_cycle_is_recovered() -> None:
    result = decomposition.decompose(seasonal_frame(amplitude=20.0), max_points=None)

    span = max(result.seasonal) - min(result.seasonal)
    assert span == pytest.approx(40.0, rel=0.2)  # peak-to-trough of a +/-20 sine


def test_a_strongly_seasonal_series_scores_high_seasonal_strength() -> None:
    result = decomposition.decompose(
        seasonal_frame(amplitude=25.0, noise=0.5), max_points=None
    )

    assert result.strength.seasonal > 0.9
    assert 0.0 <= result.strength.trend <= 1.0


def test_pure_noise_scores_low_on_both_strengths() -> None:
    result = decomposition.decompose(
        seasonal_frame(slope=0.0, amplitude=0.0, noise=10.0), max_points=None
    )

    assert result.strength.seasonal < 0.5
    assert result.strength.trend < 0.5


# --- Inputs STL cannot take unaided -----------------------------------------


def test_absent_hours_are_interpolated_and_counted() -> None:
    """A decomposition resting on invented points should say how many."""
    frame = seasonal_frame(hours=24 * 10)
    frame = frame.drop(index=frame.index[50:70]).reset_index(drop=True)

    result = decomposition.decompose(frame, max_points=None)

    assert result.interpolated_points == 20
    assert result.points_analysed == 24 * 10


def test_a_heavily_interpolated_series_carries_a_caveat() -> None:
    frame = seasonal_frame(hours=24 * 10)
    # Drop a quarter of the hours, scattered.
    frame = frame.drop(index=frame.index[::4]).reset_index(drop=True)

    result = decomposition.decompose(frame, max_points=None)

    assert result.interpolated_points > 0
    assert "interpolated" in result.caveat


def test_a_clean_series_carries_no_caveat() -> None:
    assert decomposition.decompose(seasonal_frame()).caveat == ""


def test_only_one_station_is_decomposed() -> None:
    """Stacking stations into one series produces a meaningless result."""
    result = decomposition.decompose(seasonal_frame(stations=3), max_points=None)

    assert result.points_analysed == 24 * 30  # one station's worth, not three


def test_the_busiest_station_is_chosen_by_default() -> None:
    frame = seasonal_frame(stations=2)
    sparse = frame["station"].unique()[1]
    # Blank most of the second station's readings.
    frame.loc[frame["station"] == sparse, "pm25"] = np.nan
    frame.loc[frame.index[-5:], "pm25"] = 50.0

    assert decomposition.busiest_station(frame, "pm25") != sparse


def test_a_named_station_is_honoured() -> None:
    frame = seasonal_frame(stations=2)
    chosen = sorted(frame["station"].unique())[1]

    assert decomposition.decompose(frame, station=chosen).station == chosen


# --- Refusals ---------------------------------------------------------------


def test_too_short_a_window_is_refused_rather_than_fitted() -> None:
    """Two days cannot support a trend; a confident-looking line from them
    would be worse than an error."""
    with pytest.raises(InsufficientDataError, match="full cycles"):
        decomposition.decompose(seasonal_frame(hours=40))


def test_a_constant_series_is_refused() -> None:
    frame = seasonal_frame(hours=24 * 10)
    frame["pm25"] = 42.0

    with pytest.raises(InsufficientDataError, match="constant"):
        decomposition.decompose(frame)


def test_an_unknown_column_is_refused() -> None:
    with pytest.raises(InsufficientDataError, match="Unknown column"):
        decomposition.decompose(seasonal_frame(), column="ozone")


def test_an_unknown_station_is_refused() -> None:
    with pytest.raises(InsufficientDataError, match="no observations"):
        decomposition.decompose(seasonal_frame(), station="0.00000,0.00000")


def test_an_empty_frame_is_refused() -> None:
    empty = seasonal_frame(hours=24 * 4).iloc[0:0]

    with pytest.raises(InsufficientDataError):
        decomposition.decompose(empty)


# --- Output shape -----------------------------------------------------------


def test_only_the_trailing_window_is_returned_by_default() -> None:
    """The full series is megabytes of JSON for a chart that cannot render it."""
    result = decomposition.decompose(seasonal_frame(hours=24 * 40), max_points=100)

    assert result.points_returned == 100
    assert result.points_analysed == 24 * 40
    assert len(result.timestamps) == 100


def test_every_component_has_the_same_length_as_the_timestamps() -> None:
    result = decomposition.decompose(seasonal_frame(), max_points=200)

    assert (
        len(result.timestamps)
        == len(result.observed)
        == len(result.trend)
        == len(result.seasonal)
        == len(result.residual)
    )


def test_a_weekly_period_can_be_requested() -> None:
    result = decomposition.decompose(
        seasonal_frame(hours=24 * 30), period=168, max_points=None
    )

    assert result.period == 168


def test_the_result_is_json_serialisable() -> None:
    import json

    payload = decomposition.decompose(seasonal_frame(), max_points=50).as_dict()

    assert json.loads(json.dumps(payload))["period"] == 24


def test_pm25_is_the_default_series() -> None:
    """Module 4 names PM2.5 specifically."""
    assert decomposition.decompose(seasonal_frame()).column == "pm25"


@pytest.mark.parametrize("column", MEASUREMENT_COLUMNS)
def test_any_measurement_column_can_be_decomposed(column: str) -> None:
    frame = seasonal_frame(hours=24 * 8)
    rng = np.random.default_rng(23)
    frame[column] = 50 + rng.normal(0, 5, len(frame)) + 10 * np.sin(
        2 * np.pi * np.arange(len(frame)) / 24
    )

    assert decomposition.decompose(frame, column=column).column == column
