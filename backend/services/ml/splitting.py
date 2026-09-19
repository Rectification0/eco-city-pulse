"""Time-aware splitting and cross-validation (tasks 7.2, 7.10; AC-8, design §10.1).

design §10.1 calls leakage "the dominant failure mode in time-series ML". This
module is where the defence lives, and it has three parts.

**1. The split is chronological, over timestamps, not rows.** A random split is
the obvious mistake, but splitting a multi-station frame by row position is the
subtle one: rows are sorted by ``(station, timestamp)``, so a positional cut
would put the whole of one station in train and another in test while both span
the same period -- every "future" test hour having a same-hour twin in training.
The cut is therefore made on a **timestamp**, and every row on or after it is
test regardless of which station it came from.

**2. An embargo sits between them.** This is the part a plain chronological
split misses. A training row at time *t* carries the target ``pm25`` at
*t + horizon*. If *t + horizon* lands inside the test period, that training row
has the answer to a test question written on it. So the last ``horizon`` hours
before the cut are **purged** from training. The cost is a handful of rows; the
alternative is a 24-hour model that has read the first day of its own test set.

**3. Cross-validation expands, never folds.** ``TimeSeriesSplit`` over the
*unique timestamps* of the training portion, with the same embargo as ``gap``.
K-Fold would train on Thursday to predict Tuesday.

Every splitter here returns positional indices into the frame it was given, and
each one satisfies the audit of task 7.13 by construction: max(train timestamp)
+ embargo <= min(test timestamp).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from core.exceptions import InsufficientDataError

DEFAULT_TEST_FRACTION = 0.2
DEFAULT_CV_SPLITS = 5

# Below this, a "test set" is a handful of hours and its MAE is noise.
MIN_TEST_ROWS = 24
MIN_TRAIN_ROWS = 100


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """One chronological split, and the evidence that it is one."""

    train_index: np.ndarray
    test_index: np.ndarray
    split_at: datetime
    embargo_hours: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargoed_rows: int

    @property
    def gap_hours(self) -> float:
        """Hours between the last training row and the first test row.

        At least ``embargo_hours`` whenever the split is sound, which is exactly
        what the leakage audit asserts (task 7.13).
        """
        return (self.test_start - self.train_end).total_seconds() / 3600.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_at": self.split_at.isoformat(),
            "embargo_hours": self.embargo_hours,
            "embargoed_rows": self.embargoed_rows,
            "train_rows": len(self.train_index),
            "test_rows": len(self.test_index),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "gap_hours": round(self.gap_hours, 2),
        }


def split_at_timestamp(
    frame: pd.DataFrame,
    *,
    split_at: datetime,
    embargo_hours: int = 0,
) -> TimeSplit:
    """Everything before the cut trains; everything from it onward tests.

    ``embargo_hours`` removes the rows immediately before the cut whose target
    would fall on the far side of it.
    """
    timestamps = frame["timestamp"]
    embargo_start = split_at - timedelta(hours=embargo_hours)

    train_mask = timestamps < embargo_start
    test_mask = timestamps >= split_at
    embargoed = int(((timestamps >= embargo_start) & (timestamps < split_at)).sum())

    train_index = np.flatnonzero(train_mask.to_numpy())
    test_index = np.flatnonzero(test_mask.to_numpy())

    if len(train_index) < MIN_TRAIN_ROWS or len(test_index) < MIN_TEST_ROWS:
        raise InsufficientDataError(
            "Not enough history either side of the split to train and evaluate: "
            f"{len(train_index)} train rows (need {MIN_TRAIN_ROWS}), "
            f"{len(test_index)} test rows (need {MIN_TEST_ROWS}).",
            details={
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "split_at": split_at.isoformat(),
            },
        )

    return TimeSplit(
        train_index=train_index,
        test_index=test_index,
        split_at=split_at,
        embargo_hours=embargo_hours,
        train_start=timestamps.iloc[train_index].min().to_pydatetime(),
        train_end=timestamps.iloc[train_index].max().to_pydatetime(),
        test_start=timestamps.iloc[test_index].min().to_pydatetime(),
        test_end=timestamps.iloc[test_index].max().to_pydatetime(),
        embargoed_rows=embargoed,
    )


def chronological_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    embargo_hours: int = 0,
) -> TimeSplit:
    """Split so the last ``test_fraction`` of the *period* is held out (task 7.2).

    The fraction is taken over distinct timestamps rather than over rows, so a
    station that reports more often cannot drag the boundary toward its own
    data.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if frame.empty:
        raise InsufficientDataError("Cannot split an empty frame.")

    moments = np.sort(frame["timestamp"].unique())
    cut = int(len(moments) * (1.0 - test_fraction))
    cut = min(max(cut, 1), len(moments) - 1)

    return split_at_timestamp(
        frame, split_at=pd.Timestamp(moments[cut]).to_pydatetime(), embargo_hours=embargo_hours
    )


def expanding_window_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    embargo_hours: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """``TimeSeriesSplit`` over unique timestamps, mapped back to rows (task 7.10).

    Splitting the timestamps and then mapping keeps every row of a given hour on
    the same side of every boundary. Splitting rows directly would let one
    station's 14:00 train a model tested on another station's 14:00 -- the same
    hour, the same weather, scored as if it were the future.

    ``gap`` carries the embargo into every fold, so the validation scores are
    produced under the same discipline as the final test score. Folds whose
    training portion is empty after the gap are skipped rather than yielded as
    degenerate.
    """
    moments = np.sort(frame["timestamp"].unique())

    if len(moments) < n_splits + 1:
        raise InsufficientDataError(
            f"{n_splits} expanding-window folds need at least {n_splits + 1} "
            f"distinct timestamps; {len(moments)} available.",
            details={"required": n_splits + 1, "available": len(moments)},
        )

    timestamps = frame["timestamp"].to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=embargo_hours)

    for train_moments, test_moments in splitter.split(moments):
        # Both halves are contiguous in time, so a boundary comparison selects
        # the rows exactly and in one pass. Set membership over every row would
        # give the same answer for several times the work, and this runs once
        # per fold per model.
        train_end = moments[train_moments[-1]]
        test_start, test_end = moments[test_moments[0]], moments[test_moments[-1]]

        train_index = np.flatnonzero(timestamps <= train_end)
        test_index = np.flatnonzero(
            (timestamps >= test_start) & (timestamps <= test_end)
        )

        if len(train_index) == 0 or len(test_index) == 0:
            continue

        yield train_index, test_index


def audit(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    embargo_hours: int = 0,
) -> dict[str, Any]:
    """The facts task 7.13 asserts, computed rather than assumed.

    Returned as data rather than raised as an assertion so the training report
    can carry it: a run that claims a clean split should be able to show the
    timestamps it rests on.
    """
    timestamps = frame["timestamp"]
    train_times = timestamps.iloc[train_index]
    test_times = timestamps.iloc[test_index]

    train_end = train_times.max()
    test_start = test_times.min()
    gap_hours = (test_start - train_end).total_seconds() / 3600.0

    return {
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "train_end": train_end.isoformat(),
        "test_start": test_start.isoformat(),
        "gap_hours": round(gap_hours, 2),
        "embargo_hours": embargo_hours,
        "train_precedes_test": bool(train_end < test_start),
        "embargo_respected": bool(gap_hours >= embargo_hours),
        "no_overlapping_rows": not bool(set(train_index) & set(test_index)),
        "no_shared_timestamps": bool(
            set(train_times.unique()).isdisjoint(set(test_times.unique()))
        ),
    }


__all__ = [
    "DEFAULT_CV_SPLITS",
    "DEFAULT_TEST_FRACTION",
    "MIN_TEST_ROWS",
    "MIN_TRAIN_ROWS",
    "TimeSplit",
    "audit",
    "chronological_split",
    "expanding_window_splits",
    "split_at_timestamp",
]
