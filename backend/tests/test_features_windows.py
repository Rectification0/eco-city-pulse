"""Lag and rolling features (tasks 5.2, 5.3, 5.7; AC-8).

Two things are being proved here, and only one of them is arithmetic.

The first is that a lag is a shift in **time**, not in rows. The fixtures put a
gap in the series on purpose, because that is the only case where the two
differ -- and it is the case where a row-wise ``shift(1)`` produces a plausible
wrong number that nothing downstream would question.

The second is AC-8: no feature reads forward. That is asserted the direct way --
change the future, check the past did not move.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from services.datasets import prepare_frame
from services.features.spec import DEFAULT_SPEC, LagSpec, RollingSpec
from services.features.windows import add_window_features, hourly_grid

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


COLUMNS = (
    "id",
    "source_id",
    "timestamp",
    "lat",
    "lon",
    "pm25",
    "pm10",
    "temp",
    "humidity",
    "traffic_score",
    "is_anomaly",
)


def build_frame(rows: int = 120, *, stations: int = 1, ramp: bool = True) -> pd.DataFrame:
    """An hourly frame whose values are known by construction.

    With ``ramp``, ``pm25`` is simply the hour index, so the correct value of
    ``pm25_lag_24h`` at hour *h* is ``h - 24`` and an assertion reads like the
    definition of the feature rather than like a regression baseline.
    """
    if rows == 0:
        return prepare_frame(pd.DataFrame(columns=list(COLUMNS)))

    records = []
    for station in range(stations):
        for hour in range(rows):
            value = float(hour) if ramp else float((hour % 12) * 3)
            records.append(
                {
                    "id": station * rows + hour + 1,
                    "source_id": 1,
                    "timestamp": START + timedelta(hours=hour),
                    "lat": 28.6 + station * 0.1,
                    "lon": 77.2,
                    "pm25": value,
                    "pm10": value * 2,
                    "temp": 20.0 + hour % 7,
                    "humidity": 50.0,
                    "traffic_score": float(hour % 10),
                    "is_anomaly": False,
                }
            )
    return prepare_frame(pd.DataFrame(records))


LAGS = (LagSpec("pm25", 1), LagSpec("pm25", 24), LagSpec("temp", 3))
ROLLINGS = (RollingSpec("pm25", 24, "mean"), RollingSpec("traffic_score", 6, "std"))


# --- Lag correctness (task 5.2) ---------------------------------------------


def test_a_lag_is_the_value_that_many_hours_earlier() -> None:
    frame = build_frame(rows=60)

    result = add_window_features(frame, lags=LAGS)

    assert result["pm25_lag_1h"].iloc[30] == pytest.approx(29.0)
    assert result["pm25_lag_24h"].iloc[30] == pytest.approx(6.0)
    assert result["temp_lag_3h"].iloc[30] == pytest.approx(frame["temp"].iloc[27])


def test_the_first_rows_of_a_series_have_no_lag_to_report() -> None:
    """NaN, not a filled-in guess: before the series starts there is no value."""
    frame = build_frame(rows=60)

    result = add_window_features(frame, lags=LAGS)

    assert result["pm25_lag_1h"].iloc[0] != result["pm25_lag_1h"].iloc[0]  # NaN
    assert result["pm25_lag_24h"].iloc[:24].isna().all()
    assert result["pm25_lag_24h"].iloc[24:].notna().all()


def test_a_lag_never_crosses_a_gap_in_the_series() -> None:
    """The case a row-wise shift gets wrong.

    Hours 50-55 are absent, so the row at hour 56 has no reading one hour
    earlier. ``shift(1)`` on rows would hand it hour 49 -- seven hours stale,
    labelled ``lag_1h``.
    """
    frame = build_frame(rows=80)
    gap = frame["timestamp"].between(START + timedelta(hours=50), START + timedelta(hours=55))
    frame = frame[~gap].reset_index(drop=True)

    result = add_window_features(frame, lags=LAGS)
    at_56 = result[result["timestamp"] == START + timedelta(hours=56)].iloc[0]
    previous_row = result[result["timestamp"] == START + timedelta(hours=49)].iloc[0]

    assert np.isnan(at_56["pm25_lag_1h"])
    assert previous_row["pm25"] == pytest.approx(49.0)  # the value shift() would have used


def test_a_lag_that_reaches_over_a_gap_to_a_real_hour_still_works() -> None:
    """The grid restores alignment; it does not just erase everything near a gap."""
    frame = build_frame(rows=100)
    gap = frame["timestamp"].between(START + timedelta(hours=50), START + timedelta(hours=55))
    frame = frame[~gap].reset_index(drop=True)

    result = add_window_features(frame, lags=LAGS)
    at_72 = result[result["timestamp"] == START + timedelta(hours=72)].iloc[0]

    assert at_72["pm25_lag_24h"] == pytest.approx(48.0)


def test_a_lag_never_crosses_stations() -> None:
    frame = build_frame(rows=40, stations=2)

    result = add_window_features(frame, lags=LAGS)
    first_rows = result.groupby("station", sort=False).head(1)

    # Each station's series starts afresh; neither inherits the other's tail.
    assert first_rows["pm25_lag_1h"].isna().all()


# --- Rolling statistics (task 5.3) ------------------------------------------


def test_a_rolling_mean_covers_the_trailing_window_including_now() -> None:
    frame = build_frame(rows=60)

    result = add_window_features(frame, rollings=ROLLINGS)

    expected = frame["pm25"].iloc[7:31].mean()  # hours 7..30 inclusive = 24 hours
    assert result["pm25_rolling_mean_24h"].iloc[30] == pytest.approx(expected)


def test_a_rolling_window_can_be_shifted_off_the_current_hour() -> None:
    frame = build_frame(rows=60)
    strictly_prior = (RollingSpec("pm25", 24, "mean", include_current=False),)

    result = add_window_features(frame, rollings=strictly_prior)

    expected = frame["pm25"].iloc[6:30].mean()  # one hour further back
    assert result["pm25_rolling_mean_24h"].iloc[30] == pytest.approx(expected)


def test_a_window_below_minimum_coverage_reports_nothing() -> None:
    """A 24-hour "mean" from two readings is not a daily mean."""
    frame = build_frame(rows=60)

    result = add_window_features(frame, rollings=ROLLINGS)

    assert result["pm25_rolling_mean_24h"].iloc[:11].isna().all()
    assert result["pm25_rolling_mean_24h"].iloc[11:].notna().all()


def test_a_rolling_std_needs_two_observations() -> None:
    frame = build_frame(rows=20)

    result = add_window_features(frame, rollings=(RollingSpec("traffic_score", 6, "std"),))

    assert np.isnan(result["traffic_score_rolling_std_6h"].iloc[0])
    assert result["traffic_score_rolling_std_6h"].iloc[5] == pytest.approx(
        frame["traffic_score"].iloc[:6].std()
    )


def test_a_rolling_window_counts_hours_not_rows_across_a_gap() -> None:
    frame = build_frame(rows=80)
    gap = frame["timestamp"].between(START + timedelta(hours=40), START + timedelta(hours=51))
    frame = frame[~gap].reset_index(drop=True)

    result = add_window_features(frame, rollings=(RollingSpec("pm25", 24, "mean"),))
    at_60 = result[result["timestamp"] == START + timedelta(hours=60)].iloc[0]

    # Hours 37..60 span the gap: only the 12 present hours contribute, which
    # clears the 50% coverage floor. A row-wise window would have reached back
    # to hour 25 and silently widened itself to 36 hours.
    present = [hour for hour in range(37, 61) if not 40 <= hour <= 51]
    assert at_60["pm25_rolling_mean_24h"] == pytest.approx(float(np.mean(present)))


# --- Exact vs incremental windows -------------------------------------------


def test_a_window_is_reduced_from_its_own_contents_only() -> None:
    """The reason Phase 5 can claim identity rather than near-identity.

    pandas' incremental accumulator carries rounding error that depends on how
    many rows preceded the window, so the same hour can differ in its last bits
    between a full-history build and a serving window. Reducing each window
    from its own contents removes the dependence entirely.
    """
    frame = build_frame(rows=400, ramp=False)
    rolling = (RollingSpec("pm25", 24, "mean"),)

    full = add_window_features(frame, rollings=rolling)
    tail = frame.iloc[300:].reset_index(drop=True)
    windowed = add_window_features(tail, rollings=rolling)

    at = START + timedelta(hours=380)
    assert (
        full.loc[full["timestamp"] == at, "pm25_rolling_mean_24h"].iloc[0]
        == windowed.loc[windowed["timestamp"] == at, "pm25_rolling_mean_24h"].iloc[0]
    )


def test_the_fast_mode_agrees_with_the_exact_one() -> None:
    """Turning exactness off trades bits for speed, not correctness."""
    frame = build_frame(rows=200, ramp=False)

    exact = add_window_features(frame, rollings=ROLLINGS, exact_windows=True)
    fast = add_window_features(frame, rollings=ROLLINGS, exact_windows=False)

    for name in ("pm25_rolling_mean_24h", "traffic_score_rolling_std_6h"):
        assert np.allclose(exact[name], fast[name], rtol=1e-9, equal_nan=True)


# --- No forward-looking leakage (AC-8) --------------------------------------


def test_changing_the_future_never_changes_a_past_feature() -> None:
    """AC-8 stated as a property, asserted as one.

    Every feature at *t* is a function of ``[t - k, t]``. So rewriting every
    observation after hour 60 must leave every feature at or before hour 60
    bit-for-bit identical.
    """
    frame = build_frame(rows=120)
    names = list(DEFAULT_SPEC.feature_names[5:])  # the windowed features

    before = add_window_features(frame, lags=LAGS, rollings=ROLLINGS)

    mutated = frame.copy()
    future = mutated.index > 60
    for column in ("pm25", "pm10", "temp", "humidity", "traffic_score"):
        mutated.loc[future, column] = 9999.0
    after = add_window_features(mutated, lags=LAGS, rollings=ROLLINGS)

    pd.testing.assert_frame_equal(
        before.loc[:60, names], after.loc[:60, names], check_exact=True
    )


def test_the_future_does_move_when_it_should() -> None:
    """The guard against a vacuous leakage test: if nothing ever changed, the
    assertion above would pass on a transformer that returned constants."""
    frame = build_frame(rows=120)
    mutated = frame.copy()
    mutated.loc[mutated.index > 60, "pm25"] = 9999.0

    before = add_window_features(frame, lags=LAGS)
    after = add_window_features(mutated, lags=LAGS)

    assert before["pm25_lag_1h"].iloc[70] != after["pm25_lag_1h"].iloc[70]


# --- The grid itself --------------------------------------------------------


def test_the_grid_fills_absent_hours_with_nan() -> None:
    frame = build_frame(rows=30)
    frame = frame[frame.index != 10].reset_index(drop=True)

    grid = hourly_grid(frame, ("pm25",))

    assert len(grid) == 30
    assert grid["pm25"].isna().sum() == 1


def test_two_readings_in_one_hour_are_averaged() -> None:
    """DR-4 should prevent this; if it ever slips, averaging matches what the
    harmonizer would have done rather than letting an arbitrary row win."""
    frame = build_frame(rows=10)
    duplicate = frame.iloc[[5]].copy()
    duplicate["pm25"] = 100.0
    frame = prepare_frame(pd.concat([frame, duplicate], ignore_index=True))

    grid = hourly_grid(frame, ("pm25",))

    assert grid["pm25"].iloc[5] == pytest.approx((5.0 + 100.0) / 2)


def test_an_empty_frame_still_gets_the_feature_columns() -> None:
    frame = build_frame(rows=0)

    result = add_window_features(frame, lags=LAGS, rollings=ROLLINGS)

    for spec in (*LAGS, *ROLLINGS):
        assert spec.name in result.columns
    assert result.empty


def test_an_absurd_time_span_is_refused_rather_than_allocated() -> None:
    frame = build_frame(rows=5)
    frame.loc[0, "timestamp"] = pd.Timestamp("1900-01-01", tz="UTC")

    with pytest.raises(ValueError, match="grid ceiling"):
        add_window_features(frame, lags=LAGS)
