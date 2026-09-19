"""Time-aware splitting (tasks 7.2, 7.10, 7.13; AC-8, design §10.1).

design §10.1 calls leakage "the dominant failure mode in time-series ML". These
tests are the standing proof that the defence works, and they are written around
the three ways it can fail:

* a split that shuffles (obvious),
* a split that cuts a multi-station frame by row position (subtle),
* a split that ignores the target's own reach across the boundary (invisible
  without thinking about what the label *is*).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from core.exceptions import InsufficientDataError
from services.ml import splitting
from tests.test_features_windows import START, build_frame


def test_the_split_is_chronological() -> None:
    frame = build_frame(rows=500)

    split = splitting.chronological_split(frame, test_fraction=0.2)

    assert split.train_end < split.test_start
    assert len(split.test_index) == pytest.approx(100, abs=2)


def test_every_training_row_precedes_every_test_row() -> None:
    """Task 7.13, stated as the audit states it."""
    frame = build_frame(rows=500)

    split = splitting.chronological_split(frame, test_fraction=0.25)
    train_times = frame["timestamp"].iloc[split.train_index]
    test_times = frame["timestamp"].iloc[split.test_index]

    assert train_times.max() < test_times.min()


def test_a_multi_station_frame_is_cut_by_time_not_by_row() -> None:
    """The subtle failure.

    Rows arrive sorted by ``(station, timestamp)``, so a positional cut would
    put one whole station in train and another in test -- both spanning the same
    period, every "future" test hour having a same-hour twin in training.
    """
    frame = build_frame(rows=300, stations=3)

    split = splitting.chronological_split(frame, test_fraction=0.2)
    train = frame.iloc[split.train_index]
    test = frame.iloc[split.test_index]

    # Every station appears on both sides: the cut was made on time.
    assert set(train["station"]) == set(test["station"])
    assert train["timestamp"].max() < test["timestamp"].min()


def test_no_timestamp_appears_on_both_sides() -> None:
    frame = build_frame(rows=300, stations=3)

    split = splitting.chronological_split(frame, test_fraction=0.2)
    audit = splitting.audit(frame, split.train_index, split.test_index)

    assert audit["no_shared_timestamps"]
    assert audit["no_overlapping_rows"]


# --- The embargo ------------------------------------------------------------


def test_the_embargo_purges_the_rows_whose_target_crosses_the_boundary() -> None:
    """The invisible failure.

    A training row at time *t* carries the PM2.5 at *t + horizon*. If that
    lands inside the test period, the row has the answer to a test question
    written on it.
    """
    frame = build_frame(rows=500)

    split = splitting.chronological_split(frame, test_fraction=0.2, embargo_hours=24)

    assert split.gap_hours >= 24
    assert split.embargoed_rows == 24


def test_without_an_embargo_train_runs_right_up_to_the_boundary() -> None:
    """The contrast that shows the embargo is doing something."""
    frame = build_frame(rows=500)

    split = splitting.chronological_split(frame, test_fraction=0.2, embargo_hours=0)

    assert split.gap_hours == 1.0  # one hourly step, nothing purged
    assert split.embargoed_rows == 0


def test_the_audit_reports_the_embargo_it_was_given() -> None:
    frame = build_frame(rows=500)

    split = splitting.chronological_split(frame, test_fraction=0.2, embargo_hours=6)
    audit = splitting.audit(
        frame, split.train_index, split.test_index, embargo_hours=6
    )

    assert audit["train_precedes_test"]
    assert audit["embargo_respected"]
    assert audit["gap_hours"] >= 6


def test_an_explicit_split_timestamp_is_honoured() -> None:
    frame = build_frame(rows=500)
    at = START + timedelta(hours=400)

    split = splitting.split_at_timestamp(frame, split_at=at)

    assert split.test_start == at
    assert split.train_end < at


# --- Cross-validation (task 7.10) -------------------------------------------


def test_cross_validation_expands_rather_than_folding() -> None:
    """K-Fold would train on Thursday to predict Tuesday."""
    frame = build_frame(rows=500)

    folds = list(splitting.expanding_window_splits(frame, n_splits=4))

    assert len(folds) == 4
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes)  # the training window only ever grows


def test_every_fold_trains_strictly_before_it_tests() -> None:
    frame = build_frame(rows=500, stations=2)

    for train_index, test_index in splitting.expanding_window_splits(
        frame, n_splits=4, embargo_hours=6
    ):
        audit = splitting.audit(frame, train_index, test_index, embargo_hours=6)
        assert audit["train_precedes_test"]
        assert audit["embargo_respected"]
        assert audit["no_shared_timestamps"]


def test_folds_never_reuse_a_row_across_the_boundary() -> None:
    frame = build_frame(rows=400)

    for train_index, test_index in splitting.expanding_window_splits(frame, n_splits=3):
        assert not set(train_index) & set(test_index)


def test_a_fold_keeps_every_station_of_an_hour_on_one_side() -> None:
    """Otherwise one station's 14:00 trains a model tested on another
    station's 14:00 -- the same hour, the same weather, scored as the future."""
    frame = build_frame(rows=200, stations=3)

    for train_index, test_index in splitting.expanding_window_splits(frame, n_splits=3):
        train_times = set(frame["timestamp"].iloc[train_index])
        test_times = set(frame["timestamp"].iloc[test_index])
        assert train_times.isdisjoint(test_times)


# --- Refusals ---------------------------------------------------------------


def test_too_little_history_is_refused_rather_than_split() -> None:
    frame = build_frame(rows=40)

    with pytest.raises(InsufficientDataError, match="Not enough history"):
        splitting.chronological_split(frame, test_fraction=0.2)


def test_an_impossible_test_fraction_is_refused() -> None:
    frame = build_frame(rows=500)

    with pytest.raises(ValueError, match="test_fraction"):
        splitting.chronological_split(frame, test_fraction=1.5)


def test_too_few_timestamps_to_fold_is_a_domain_error() -> None:
    frame = build_frame(rows=200)

    with pytest.raises(InsufficientDataError, match="distinct timestamps"):
        list(splitting.expanding_window_splits(frame.iloc[:3], n_splits=5))


def test_the_split_serialises_for_the_report() -> None:
    frame = build_frame(rows=500)

    payload = splitting.chronological_split(
        frame, test_fraction=0.2, embargo_hours=24
    ).as_dict()

    assert payload["embargo_hours"] == 24
    assert payload["gap_hours"] >= 24
    assert np.isfinite(payload["train_rows"])
