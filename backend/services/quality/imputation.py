"""Imputation (tasks 3.2, 3.3; FEAT-03, AC-4, design §7).

Two repairs, applied in that order, because design §7 assigns them to different
kinds of gap:

1. **Contiguous short gaps → forward/backward fill.** For an hour or two of a
   smooth physical series, the neighbouring value is a better estimate than any
   model: "time-series locality is the better signal". Filling only *short*
   runs is the whole point -- carrying a value across a two-day sensor outage
   would invent a flat line where there is no information.
2. **Everything left → MICE.** ``IterativeImputer`` conditions each column on
   the others, which is exactly right for the MAR mechanism the analyser
   detects, and is what FEAT-03 names.

Both operate **per station**. A fill that crosses stations would carry one
sensor's reading into another's gap, and a model fitted across a city's whole
pollution gradient would regress every station toward the city mean.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# Must precede the IterativeImputer import: scikit-learn gates it behind an
# explicit opt-in while the API is still marked experimental.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from services.datasets import MEASUREMENT_COLUMNS, STATION_COLUMN

# Gaps up to this many consecutive hours are filled locally; longer ones are
# left for MICE. Three hours is roughly how long PM2.5 stays autocorrelated
# enough for a neighbouring reading to beat a model.
DEFAULT_MAX_GAP_HOURS = 3

# Deterministic imputation matters more than it sounds: AC-4 asserts a null-free
# feature set, and a run that produced different values each time would make
# every downstream metric unreproducible.
RANDOM_STATE = 42
# 10 was not enough to converge on this data and emitted a warning on every
# station; 25 converges well inside the budget.
MAX_ITER = 25
TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class ImputationSummary:
    """What each stage repaired, per column."""

    filled_short_gaps: dict[str, int]
    imputed_by_mice: dict[str, int]
    remaining_nulls: dict[str, int]
    max_gap_hours: int
    backward_fill_used: bool
    # Stations where the chained equations were still moving when the iteration
    # budget ran out. Reported rather than warned about: the values are usable
    # and stay inside the observed range, but a reader deserves to know which
    # ones the model had not settled on. Short windows converge less readily
    # than long ones, so this rises as the slice shrinks.
    stations_not_converged: int = 0
    stations_imputed: int = 0

    @property
    def is_complete(self) -> bool:
        """AC-4: the modelled feature set has no nulls left."""
        return all(count == 0 for count in self.remaining_nulls.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "filled_short_gaps": self.filled_short_gaps,
            "imputed_by_mice": self.imputed_by_mice,
            "remaining_nulls": self.remaining_nulls,
            "max_gap_hours": self.max_gap_hours,
            "backward_fill_used": self.backward_fill_used,
            "is_complete": self.is_complete,
            "stations_imputed": self.stations_imputed,
            "stations_not_converged": self.stations_not_converged,
        }


def _short_gap_mask(series: pd.Series, max_gap: int) -> np.ndarray:
    """Positions inside a NaN run no longer than ``max_gap``.

    ``ffill(limit=n)`` is not equivalent: it fills the first *n* entries of an
    arbitrarily long gap, which quietly fabricates the start of a two-day
    outage. This selects whole runs or nothing.
    """
    null = series.isna().to_numpy()
    mask = np.zeros(len(null), dtype=bool)
    if not null.any():
        return mask

    start = None
    for index, is_null in enumerate(null):
        if is_null and start is None:
            start = index
        elif not is_null and start is not None:
            if index - start <= max_gap:
                mask[start:index] = True
            start = None
    if start is not None and len(null) - start <= max_gap:
        mask[start:] = True

    return mask


def fill_short_gaps(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    max_gap_hours: int = DEFAULT_MAX_GAP_HOURS,
    backward: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Forward/backward fill runs of at most ``max_gap_hours`` (task 3.3).

    ``backward`` exists as a switch because a backward fill reads *future*
    values. That is fine for cleaning a historical record for analysis, which is
    what this stage is for, but it is a leakage hazard if applied across a
    train/test boundary. Phase 7 must clean within its training window, or pass
    ``backward=False``.
    """
    frame = frame.copy()
    filled: dict[str, int] = {column: 0 for column in columns}

    if frame.empty:
        return frame, filled

    for column in columns:
        before = frame[column].isna().sum()
        repaired = []

        for _, group in frame.groupby(STATION_COLUMN, sort=False):
            series = group[column]
            mask = _short_gap_mask(series, max_gap_hours)
            if not mask.any():
                repaired.append(series)
                continue

            candidate = series.ffill()
            if backward:
                candidate = candidate.bfill()
            # Only the positions inside short runs are taken from the fill; the
            # long runs keep their NaNs for MICE.
            repaired.append(series.where(~mask, candidate))

        frame[column] = pd.concat(repaired).reindex(frame.index)
        filled[column] = int(before - frame[column].isna().sum())

    return frame, filled


def impute_mice(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    max_iter: int = MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, int], tuple[int, int]]:
    """MICE via ``IterativeImputer`` (task 3.2, FEAT-03).

    Fitted **per station** so each sensor's own relationships drive its
    imputation. A station with too few complete rows to fit falls back to that
    station's column medians -- which is honest about being weaker, and still
    beats leaving a null that would drop the row from every model.
    """
    frame = frame.copy()
    if frame.empty:
        return frame, {column: 0 for column in columns}, (0, 0)

    before = {column: int(frame[column].isna().sum()) for column in columns}

    if not any(before.values()):
        return frame, {column: 0 for column in columns}, (0, 0)

    repaired_blocks: list[pd.DataFrame] = []
    not_converged = 0
    stations_imputed = 0

    for _, group in frame.groupby(STATION_COLUMN, sort=False):
        block = group[list(columns)].astype("float64")

        if not block.isna().to_numpy().any():
            repaired_blocks.append(block)
            continue

        usable = [c for c in columns if block[c].notna().any()]
        complete_rows = int(block[usable].notna().all(axis=1).sum()) if usable else 0

        if len(usable) >= 2 and complete_rows >= 10:
            imputer = IterativeImputer(
                max_iter=max_iter,
                tol=TOLERANCE,
                random_state=random_state,
                # Keep imputed values inside the range actually observed: an
                # unconstrained linear model will happily predict a negative
                # PM2.5 concentration, which is not a measurement.
                min_value=block[usable].min().to_numpy(),
                max_value=block[usable].max().to_numpy(),
                keep_empty_features=True,
            )
            stations_imputed += 1
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                block[usable] = imputer.fit_transform(block[usable])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                not_converged += 1

        # Columns MICE could not touch -- entirely empty for this station, or
        # too few complete rows to fit -- fall back to the station median. A
        # column with no observation at all for this station is skipped here
        # (its median is undefined) and picked up by the global fallback below.
        for column in columns:
            if block[column].isna().any() and block[column].notna().any():
                block[column] = block[column].fillna(block[column].median())

        repaired_blocks.append(block)

    repaired = pd.concat(repaired_blocks).reindex(frame.index)
    for column in columns:
        frame[column] = repaired[column]
        # A station with no observation of a column at all still has NaNs here;
        # the city-wide median is the last resort.
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(frame[column].median())

    imputed = {
        column: int(before[column] - frame[column].isna().sum()) for column in columns
    }
    return frame, imputed, (stations_imputed, not_converged)


def impute(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    max_gap_hours: int = DEFAULT_MAX_GAP_HOURS,
    backward: bool = True,
) -> tuple[pd.DataFrame, ImputationSummary]:
    """Short-gap fill, then MICE. The full repair path (AC-4)."""
    filled_frame, filled = fill_short_gaps(
        frame, columns=columns, max_gap_hours=max_gap_hours, backward=backward
    )
    imputed_frame, imputed, (stations, not_converged) = impute_mice(
        filled_frame, columns=columns
    )

    summary = ImputationSummary(
        filled_short_gaps=filled,
        imputed_by_mice=imputed,
        remaining_nulls={
            column: int(imputed_frame[column].isna().sum()) for column in columns
        },
        max_gap_hours=max_gap_hours,
        backward_fill_used=backward,
        stations_imputed=stations,
        stations_not_converged=not_converged,
    )
    return imputed_frame, summary


__all__ = [
    "DEFAULT_MAX_GAP_HOURS",
    "MAX_ITER",
    "RANDOM_STATE",
    "TOLERANCE",
    "ImputationSummary",
    "fill_short_gaps",
    "impute",
    "impute_mice",
]
