"""Outlier detection (tasks 3.4-3.7, AC-5).

AC-5 is the one that matters most here, and it is asserted directly: an
anomalous record is **retained** and flagged. Every other test exists to make
sure the flag means something when it is set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.datasets import MEASUREMENT_COLUMNS
from services.quality import outliers
from tests.test_missingness import build_frame

# --- IQR (task 3.4) ---------------------------------------------------------


def test_iqr_bounds_match_the_boxplot_definition() -> None:
    """``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`` on a series with known quartiles."""
    series = pd.Series(range(101))  # Q1 = 25, Q3 = 75, IQR = 50

    low, high = outliers.iqr_bounds(series)

    assert low == pytest.approx(25 - 1.5 * 50)
    assert high == pytest.approx(75 + 1.5 * 50)


def test_iqr_bounds_of_a_constant_series_never_flag() -> None:
    """No spread means nothing to reason about; flagging every distinct value
    would be noise rather than detection."""
    low, high = outliers.iqr_bounds(pd.Series([5.0] * 50))

    assert low == float("-inf")
    assert high == float("inf")


def test_iqr_bounds_of_an_empty_series_never_flag() -> None:
    low, high = outliers.iqr_bounds(pd.Series(dtype="float64"))

    assert (low, high) == (float("-inf"), float("inf"))


def test_iqr_flags_a_clear_outlier() -> None:
    frame = build_frame(rows=200, seed=51)
    frame.loc[100, "pm25"] = 5000.0

    flags = outliers.flag_iqr(frame)

    assert flags.loc[100, "pm25"]
    assert flags["pm25"].sum() < 10  # not flagging half the series


def test_iqr_bounds_are_fitted_per_station() -> None:
    """A city-wide bound on a real pollution gradient flags the dirtiest
    district wholesale rather than any anomaly."""
    frame = build_frame(rows=200, stations=2, seed=53)
    dirty = frame["station"].unique()[1]
    # One station simply runs much higher. Nothing here is anomalous.
    frame.loc[frame["station"] == dirty, "pm25"] += 400

    flags = outliers.flag_iqr(frame)

    assert flags.loc[frame["station"] == dirty, "pm25"].sum() < 10


# --- Z-score (task 3.5) -----------------------------------------------------


def test_zscore_flags_beyond_three_standard_deviations() -> None:
    frame = build_frame(rows=500, seed=57)
    mean = frame["pm25"].mean()
    std = frame["pm25"].std(ddof=0)
    frame.loc[250, "pm25"] = mean + 6 * std

    flags = outliers.flag_zscore(frame)

    assert flags.loc[250, "pm25"]


def test_zscore_threshold_is_configurable() -> None:
    frame = build_frame(rows=400, seed=59)
    mean = frame["pm25"].mean()
    std = frame["pm25"].std(ddof=0)
    frame.loc[200, "pm25"] = mean + 3.5 * std

    assert not outliers.flag_zscore(frame, threshold=4.0).loc[200, "pm25"]
    assert outliers.flag_zscore(frame, threshold=3.0).loc[200, "pm25"]


def test_zscore_ignores_a_constant_column() -> None:
    frame = build_frame(rows=100)
    frame["pm25"] = 42.0

    assert outliers.flag_zscore(frame)["pm25"].sum() == 0


# --- Isolation Forest (task 3.6) --------------------------------------------


def test_isolation_forest_flags_an_implausible_combination() -> None:
    """The point of the multivariate detector: each value is individually
    ordinary, the combination is not."""
    frame = build_frame(rows=400, seed=61)
    # A tight linear relationship between two columns, restricted to those two
    # so the structure is the whole signal rather than one axis among five.
    frame["pm25"] = np.linspace(40, 60, len(frame))
    frame["traffic_score"] = np.linspace(40, 60, len(frame))
    # Bottom of one range, top of the other — neither value is extreme alone,
    # and no univariate detector could see it.
    frame.loc[200, "pm25"] = 40.0
    frame.loc[200, "traffic_score"] = 60.0

    flags = outliers.flag_isolation_forest(
        frame, columns=("pm25", "traffic_score"), contamination=0.01
    )

    assert flags.loc[200]
    # Confirm the premise: the univariate detectors genuinely miss it.
    univariate = outliers.flag_iqr(frame, columns=("pm25", "traffic_score"))
    assert not univariate.loc[200].any()


def test_isolation_forest_is_deterministic() -> None:
    frame = build_frame(rows=300, seed=63)

    first = outliers.flag_isolation_forest(frame)
    second = outliers.flag_isolation_forest(frame)

    pd.testing.assert_series_equal(first, second)


def test_isolation_forest_respects_the_contamination_rate() -> None:
    """"auto" flagged a quarter of the real dataset; a detector that calls 25%
    of the data anomalous contributes nothing to a vote."""
    frame = build_frame(rows=1000, seed=67)

    flagged = outliers.flag_isolation_forest(frame, contamination=0.02).sum()

    assert 5 <= flagged <= 40


def test_isolation_forest_declines_on_too_little_data() -> None:
    assert not outliers.flag_isolation_forest(build_frame(rows=5)).any()


# --- The vote (task 3.7) ----------------------------------------------------


def _flags(values: list[bool], columns: tuple[str, ...] = ("pm25",)) -> pd.DataFrame:
    return pd.DataFrame({column: values for column in columns})


def test_a_majority_of_detectors_is_required() -> None:
    iqr = _flags([True, True, False, False])
    zscore = _flags([True, False, True, False])
    forest = pd.Series([False, False, False, True])

    flagged, _ = outliers.combine(iqr, zscore, forest, min_votes=2)

    assert flagged.tolist() == [True, False, False, False]


def test_a_single_detector_is_not_enough_by_default() -> None:
    iqr = _flags([True])
    zscore = _flags([False])
    forest = pd.Series([False])

    flagged, _ = outliers.combine(iqr, zscore, forest)

    assert flagged.tolist() == [False]


def test_the_vote_threshold_is_configurable() -> None:
    iqr = _flags([True])
    zscore = _flags([False])
    forest = pd.Series([False])

    flagged, _ = outliers.combine(iqr, zscore, forest, min_votes=1)

    assert flagged.tolist() == [True]


def test_an_imputed_value_is_never_flagged() -> None:
    """The engine must not report its own invention as an anomaly."""
    iqr = _flags([True, True])
    zscore = _flags([True, True])
    forest = pd.Series([True, True])
    observed = pd.DataFrame({"pm25": [True, False]})  # second value was imputed

    flagged, summary = outliers.combine(iqr, zscore, forest, observed_mask=observed)

    assert flagged.tolist() == [True, False]
    assert summary.votes_by_detector["iqr"] == 1


def test_the_summary_counts_each_detector_separately() -> None:
    iqr = _flags([True, True, False])
    zscore = _flags([True, False, False])
    forest = pd.Series([False, False, True])

    _, summary = outliers.combine(iqr, zscore, forest)

    assert summary.votes_by_detector == {"iqr": 2, "zscore": 1, "isolation_forest": 1}
    assert summary.flagged == 1
    assert summary.rows == 3


def test_the_summary_states_what_a_flag_does_not_mean() -> None:
    """A statistical outlier is not a diagnosis; that is why AC-5 keeps the row."""
    _, summary = outliers.combine(_flags([False]), _flags([False]), pd.Series([False]))

    assert "sensor fault" in summary.note
    assert "AC-5" in summary.note


# --- End to end -------------------------------------------------------------


def test_detect_flags_a_spike_and_keeps_every_row() -> None:
    """AC-5, stated directly: an outlier could be a valid pollution spike, and
    deleting it would destroy exactly the signal worth explaining."""
    frame = build_frame(rows=400, seed=71)
    frame.loc[200, ["pm25", "pm10"]] = [900.0, 1800.0]

    flags, summary = outliers.detect(frame)

    assert len(flags) == len(frame)  # nothing removed
    assert flags.loc[200]
    assert summary.flagged >= 1


def test_detection_does_not_modify_the_frame() -> None:
    frame = build_frame(rows=200, seed=73)
    before = frame.copy()

    outliers.detect(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_a_clean_frame_yields_few_flags() -> None:
    """A majority vote on ordinary data should stay quiet."""
    frame = build_frame(rows=1000, seed=79)

    _, summary = outliers.detect(frame)

    assert summary.flagged_pct < 3.0


def test_detect_on_an_empty_frame_returns_an_empty_result() -> None:
    frame = build_frame(rows=1).iloc[0:0]

    flags, summary = outliers.detect(frame)

    assert flags.empty
    assert summary.rows == 0


def test_flagged_by_column_attributes_the_reason() -> None:
    """An operator needs to know which reading looked wrong, not just that the
    row did."""
    frame = build_frame(rows=400, seed=83)
    frame.loc[200, "pm25"] = 5000.0

    _, summary = outliers.detect(frame)

    assert summary.flagged_by_column["pm25"] >= 1


@pytest.mark.parametrize("column", MEASUREMENT_COLUMNS)
def test_every_measurement_column_is_examined(column: str) -> None:
    flags = outliers.flag_iqr(build_frame(rows=100))

    assert column in flags.columns
