"""Imputation (tasks 3.2, 3.3; FEAT-03, AC-4).

AC-4 -- "zero nulls after MICE on the modelled feature set" -- is the headline,
but the ordering matters just as much: short gaps must be filled locally and
long ones must not, because carrying a value across a two-day outage invents a
flat line where there is no information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.datasets import MEASUREMENT_COLUMNS
from services.quality import imputation
from tests.test_missingness import build_frame

# --- Short-gap fill (task 3.3) ----------------------------------------------


def test_a_short_gap_is_filled() -> None:
    frame = build_frame(rows=50)
    frame.loc[10:11, "pm25"] = np.nan

    filled, counts = imputation.fill_short_gaps(frame, max_gap_hours=3)

    assert filled["pm25"].isna().sum() == 0
    assert counts["pm25"] == 2


def test_a_long_gap_is_left_for_mice() -> None:
    """``ffill(limit=n)`` would fill the first n hours of a week-long outage;
    whole runs are filled or nothing is."""
    frame = build_frame(rows=50)
    frame.loc[10:19, "pm25"] = np.nan  # run of 10, limit 3

    filled, counts = imputation.fill_short_gaps(frame, max_gap_hours=3)

    assert counts["pm25"] == 0
    assert filled.loc[10:19, "pm25"].isna().all()


def test_a_gap_exactly_at_the_limit_is_filled() -> None:
    frame = build_frame(rows=50)
    frame.loc[10:12, "pm25"] = np.nan  # run of 3

    _, counts = imputation.fill_short_gaps(frame, max_gap_hours=3)

    assert counts["pm25"] == 3


def test_a_gap_one_hour_over_the_limit_is_not_filled() -> None:
    frame = build_frame(rows=50)
    frame.loc[10:13, "pm25"] = np.nan  # run of 4

    _, counts = imputation.fill_short_gaps(frame, max_gap_hours=3)

    assert counts["pm25"] == 0


def test_short_and_long_gaps_in_one_column_are_handled_separately() -> None:
    frame = build_frame(rows=100)
    frame.loc[10:11, "pm25"] = np.nan  # short
    frame.loc[50:69, "pm25"] = np.nan  # long

    filled, counts = imputation.fill_short_gaps(frame, max_gap_hours=3)

    assert counts["pm25"] == 2
    assert filled.loc[10:11, "pm25"].notna().all()
    assert filled.loc[50:69, "pm25"].isna().all()


def test_a_fill_never_crosses_stations() -> None:
    """Carrying one sensor's reading into another's gap would fabricate a
    measurement at a place it was never taken."""
    frame = build_frame(rows=20, stations=2)
    first, second = frame["station"].unique()
    station_b = frame.index[frame["station"] == second]

    # Blank the whole of station B's pm25 except a single trailing value, and
    # make station A's values unmistakable.
    frame.loc[frame["station"] == first, "pm25"] = 999.0
    frame.loc[station_b[:-1], "pm25"] = np.nan
    frame.loc[station_b[-1], "pm25"] = 5.0

    filled, _ = imputation.fill_short_gaps(frame, max_gap_hours=100)

    assert (filled.loc[station_b, "pm25"] != 999.0).all()


def test_forward_fill_uses_the_previous_value() -> None:
    frame = build_frame(rows=20)
    frame.loc[5, "pm25"] = 42.0
    frame.loc[6, "pm25"] = np.nan

    filled, _ = imputation.fill_short_gaps(frame, max_gap_hours=3, backward=False)

    assert filled.loc[6, "pm25"] == 42.0


def test_backward_fill_can_be_disabled() -> None:
    """It reads future values, which is a leakage hazard across a train/test
    boundary (AC-8); Phase 7 needs the switch."""
    frame = build_frame(rows=20)
    frame.loc[0, "pm25"] = np.nan  # nothing earlier to carry forward

    forward_only, _ = imputation.fill_short_gaps(frame, backward=False)
    both, _ = imputation.fill_short_gaps(frame, backward=True)

    assert np.isnan(forward_only.loc[0, "pm25"])
    assert not np.isnan(both.loc[0, "pm25"])


def test_filling_does_not_change_observed_values() -> None:
    frame = build_frame(rows=60)
    original = frame["pm25"].copy()
    frame.loc[10:11, "pm25"] = np.nan

    filled, _ = imputation.fill_short_gaps(frame)

    untouched = frame["pm25"].notna()
    pd.testing.assert_series_equal(
        filled.loc[untouched, "pm25"], original.loc[untouched], check_names=False
    )


# --- MICE (task 3.2) --------------------------------------------------------


def test_mice_removes_every_remaining_null() -> None:
    """AC-4 on the stage that is meant to guarantee it."""
    frame = build_frame(rows=300, seed=15)
    rng = np.random.default_rng(3)
    for column in MEASUREMENT_COLUMNS:
        frame.loc[rng.choice(frame.index, size=30, replace=False), column] = np.nan

    imputed, counts, _ = imputation.impute_mice(frame)

    assert imputed[list(MEASUREMENT_COLUMNS)].isna().to_numpy().sum() == 0
    assert all(count > 0 for count in counts.values())


def test_mice_is_deterministic() -> None:
    """A run that produced different values each time would make every
    downstream metric unreproducible."""
    frame = build_frame(rows=200, seed=17)
    frame.loc[20:39, "pm25"] = np.nan

    first, _, _ = imputation.impute_mice(frame)
    second, _, _ = imputation.impute_mice(frame)

    pd.testing.assert_series_equal(first["pm25"], second["pm25"])


def test_mice_keeps_imputed_values_inside_the_observed_range() -> None:
    """An unconstrained linear model will happily predict a negative
    concentration, which is not a measurement."""
    frame = build_frame(rows=300, seed=19)
    frame.loc[50:99, "pm25"] = np.nan
    observed_low = frame["pm25"].min()
    observed_high = frame["pm25"].max()

    imputed, _, _ = imputation.impute_mice(frame)

    assert imputed["pm25"].min() >= observed_low - 1e-9
    assert imputed["pm25"].max() <= observed_high + 1e-9


def test_mice_does_not_alter_observed_values() -> None:
    frame = build_frame(rows=200, seed=23)
    original = frame["pm25"].copy()
    frame.loc[10:29, "pm25"] = np.nan

    imputed, _, _ = imputation.impute_mice(frame)

    observed = frame["pm25"].notna()
    pd.testing.assert_series_equal(
        imputed.loc[observed, "pm25"], original.loc[observed], check_names=False
    )


def test_mice_falls_back_when_a_station_has_too_little_data() -> None:
    """Weaker, and honest about it — but still better than leaving a null that
    would drop the row from every model."""
    frame = build_frame(rows=12, seed=29)
    frame.loc[0:5, "pm25"] = np.nan

    imputed, _, _ = imputation.impute_mice(frame)

    assert imputed["pm25"].isna().sum() == 0


def test_a_column_empty_for_one_station_still_ends_up_filled() -> None:
    frame = build_frame(rows=100, stations=2, seed=31)
    first = frame["station"].unique()[0]
    frame.loc[frame["station"] == first, "pm25"] = np.nan

    imputed, _, _ = imputation.impute_mice(frame)

    assert imputed["pm25"].isna().sum() == 0


def test_mice_on_a_complete_frame_changes_nothing() -> None:
    frame = build_frame(rows=100)

    imputed, counts, _ = imputation.impute_mice(frame)

    assert counts == {column: 0 for column in MEASUREMENT_COLUMNS}
    pd.testing.assert_frame_equal(imputed, frame)


# --- The two stages together ------------------------------------------------


def test_short_gaps_go_to_the_fill_and_long_ones_to_mice() -> None:
    """design §7 assigns the two repairs to different kinds of gap; the
    summary has to show they actually divided the work."""
    frame = build_frame(rows=300, seed=37)
    frame.loc[10:11, "pm25"] = np.nan  # short -> fill
    frame.loc[100:129, "pm25"] = np.nan  # long -> MICE

    _, summary = imputation.impute(frame, max_gap_hours=3)

    assert summary.filled_short_gaps["pm25"] == 2
    assert summary.imputed_by_mice["pm25"] == 30


def test_the_feature_set_is_null_free_afterwards() -> None:
    """AC-4, end to end."""
    frame = build_frame(rows=400, seed=41)
    rng = np.random.default_rng(5)
    for column in MEASUREMENT_COLUMNS:
        frame.loc[rng.choice(frame.index, size=50, replace=False), column] = np.nan
    frame.loc[200:249, "temp"] = np.nan  # one long outage too

    cleaned, summary = imputation.impute(frame)

    assert summary.is_complete
    assert cleaned[list(MEASUREMENT_COLUMNS)].isna().to_numpy().sum() == 0
    assert summary.remaining_nulls == {column: 0 for column in MEASUREMENT_COLUMNS}


def test_imputation_never_changes_the_row_count() -> None:
    frame = build_frame(rows=200, seed=43)
    frame.loc[50:79, "pm25"] = np.nan

    cleaned, _ = imputation.impute(frame)

    assert len(cleaned) == len(frame)


def test_the_summary_records_how_it_was_configured() -> None:
    """The backward-fill choice affects leakage, so it travels with the run."""
    frame = build_frame(rows=100)
    frame.loc[10, "pm25"] = np.nan

    _, summary = imputation.impute(frame, max_gap_hours=6, backward=False)

    assert summary.max_gap_hours == 6
    assert summary.backward_fill_used is False


def test_convergence_is_reported_rather_than_warned_about() -> None:
    """A run that did not settle is information the report should carry, not a
    line of console noise a caller has to notice."""
    frame = build_frame(rows=200, seed=47)
    frame.loc[50:79, "pm25"] = np.nan

    _, summary = imputation.impute(frame)

    assert summary.stations_imputed == 1
    assert summary.stations_not_converged <= summary.stations_imputed


def test_a_fully_complete_frame_needs_no_repair() -> None:
    _, summary = imputation.impute(build_frame(rows=100))

    assert summary.is_complete
    assert sum(summary.filled_short_gaps.values()) == 0
    assert sum(summary.imputed_by_mice.values()) == 0


@pytest.mark.parametrize("max_gap", [0, 1, 24])
def test_the_gap_limit_is_respected_at_the_edges(max_gap: int) -> None:
    frame = build_frame(rows=60)
    frame.loc[10:11, "pm25"] = np.nan  # run of 2

    _, counts = imputation.fill_short_gaps(frame, max_gap_hours=max_gap)

    assert counts["pm25"] == (2 if max_gap >= 2 else 0)
