"""Forecast targets (task 7.1, specs §6.3).

The target is the only value in the pipeline allowed to come from the future --
that is what makes it a forecast. So the direction of the shift is tested
explicitly, and so is the case where the two plausible implementations disagree:
a series with a hole in it.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from services.ml import targets
from tests.test_features_windows import START, build_frame


def test_the_target_is_the_value_that_many_hours_later() -> None:
    frame = build_frame(rows=60)  # pm25 is the hour index

    result, summary = targets.add_target(frame, 1)

    assert result["pm25_h1"].iloc[10] == pytest.approx(11.0)
    assert summary.name == "pm25_h1"
    assert summary.horizon == 1


def test_the_shift_points_forward_not_backward() -> None:
    """A sign error here trains a model to predict the past from the present,
    which scores beautifully and forecasts nothing."""
    frame = build_frame(rows=60)

    result, _ = targets.add_target(frame, 6)

    assert result["pm25_h6"].iloc[10] == pytest.approx(16.0)
    assert result["pm25_h6"].iloc[10] > result["pm25"].iloc[10]


def test_the_last_hours_of_a_series_have_no_target() -> None:
    """By definition: the future they describe has not happened."""
    frame = build_frame(rows=60)

    result, summary = targets.add_target(frame, 24)

    assert result["pm25_h24"].iloc[-24:].isna().all()
    assert result["pm25_h24"].iloc[:-24].notna().all()
    assert summary.missing == 24


def test_a_target_never_reaches_across_a_gap() -> None:
    """The case a row-wise ``shift(-1)`` gets wrong: with hours 50-55 absent,
    the row at hour 49 has no reading one hour later."""
    frame = build_frame(rows=80)
    gap = frame["timestamp"].between(
        START + timedelta(hours=50), START + timedelta(hours=55)
    )
    frame = frame[~gap].reset_index(drop=True)

    result, _ = targets.add_target(frame, 1)
    at_49 = result[result["timestamp"] == START + timedelta(hours=49)].iloc[0]

    assert np.isnan(at_49["pm25_h1"])


def test_a_target_that_clears_a_gap_still_resolves() -> None:
    frame = build_frame(rows=100)
    gap = frame["timestamp"].between(
        START + timedelta(hours=50), START + timedelta(hours=55)
    )
    frame = frame[~gap].reset_index(drop=True)

    result, _ = targets.add_target(frame, 24)
    at_40 = result[result["timestamp"] == START + timedelta(hours=40)].iloc[0]

    assert at_40["pm25_h24"] == pytest.approx(64.0)


def test_a_target_never_crosses_stations() -> None:
    frame = build_frame(rows=40, stations=2)

    result, _ = targets.add_target(frame, 1)
    last_rows = result.groupby("station", sort=False).tail(1)

    # Each station's series ends on its own; neither borrows the other's start.
    assert last_rows["pm25_h1"].isna().all()


def test_every_horizon_gets_its_own_column() -> None:
    frame = build_frame(rows=100)

    result, summaries = targets.add_targets(frame, (1, 6, 24))

    assert [summary.name for summary in summaries] == ["pm25_h1", "pm25_h6", "pm25_h24"]
    for name in ("pm25_h1", "pm25_h6", "pm25_h24"):
        assert name in result.columns


def test_coverage_shrinks_as_the_horizon_grows() -> None:
    """A 24-hour target costs a day of rows at the end of every station."""
    frame = build_frame(rows=100)

    _, summaries = targets.add_targets(frame, (1, 24))

    assert summaries[0].present > summaries[1].present
    assert summaries[0].coverage_pct > summaries[1].coverage_pct


def test_a_zero_hour_horizon_is_refused() -> None:
    """A horizon of zero is the current reading under a name that claims to be
    a forecast -- a perfect score and a useless model."""
    frame = build_frame(rows=40)

    with pytest.raises(ValueError, match="at least 1 hour"):
        targets.add_target(frame, 0)


def test_the_target_name_matches_the_registry_key() -> None:
    assert targets.target_name(6) == "pm25_h6"
