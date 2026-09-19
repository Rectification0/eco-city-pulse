"""Feature selection (task 7.4, specs §6.3).

Two filters, in this order, and both fitted on the **training rows only** --
selection is a fitted decision like any other, and choosing columns by how they
behave on the test set is leakage wearing a respectable name.

**1. Near-constant columns go.** A column that barely varies carries no
information and, after scaling, its noise is amplified into apparent signal.

**2. One of each near-duplicate pair goes.** ``pm25`` and ``pm25_log`` are the
same measurement twice; ``pm10`` tracks ``pm25`` closely in this data. Collinear
inputs do not hurt a tree's accuracy, but they do split its importance between
two names, which makes the Phase 8 explanation say "PM2.5 mattered a bit and its
log mattered a bit" where the truth is one variable mattering a lot. For Ridge
the cost is worse: collinearity makes the coefficients unstable and their signs
unreliable.

Of a correlated pair, the column kept is the one more strongly associated with
the target. The other is dropped **with its reason recorded** -- a selection
stage that cannot say why a feature is gone is indistinguishable from a bug.

**``pm25`` is never dropped.** It is the persistence baseline's only input
(task 7.5), so removing it would make the comparison of AC-7 meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Below this standard deviation (after standardizing to the column's own mean)
# a column is effectively constant.
MIN_VARIANCE = 1e-8

# |r| above which two columns are treated as the same variable twice.
MAX_CORRELATION = 0.95

# Columns the stage may never remove, whatever the filters say.
PROTECTED: tuple[str, ...] = ("pm25",)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Which columns survived, and why the others did not."""

    kept: tuple[str, ...]
    dropped: dict[str, str] = field(default_factory=dict)
    max_correlation: float = MAX_CORRELATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": list(self.kept),
            "dropped": dict(self.dropped),
            "kept_count": len(self.kept),
            "dropped_count": len(self.dropped),
            "max_correlation": self.max_correlation,
        }


def select(
    matrix: pd.DataFrame,
    target: pd.Series,
    *,
    max_correlation: float = MAX_CORRELATION,
    min_variance: float = MIN_VARIANCE,
    protected: tuple[str, ...] = PROTECTED,
) -> SelectionResult:
    """Choose the model columns from the training rows alone (task 7.4)."""
    dropped: dict[str, str] = {}
    candidates = list(matrix.columns)

    # --- Near-constant --------------------------------------------------------
    variances = matrix.var(numeric_only=True)
    for column in list(candidates):
        if column in protected:
            continue
        variance = float(variances.get(column, 0.0))
        if not np.isfinite(variance) or variance <= min_variance:
            dropped[column] = f"near-constant on the training window (var={variance:.2e})"
            candidates.remove(column)

    if not candidates:
        return SelectionResult(kept=(), dropped=dropped, max_correlation=max_correlation)

    # --- Collinear pairs ------------------------------------------------------
    # Association with the target decides which of a pair survives, so the
    # kept column is the more useful one rather than merely the earlier one.
    with np.errstate(invalid="ignore"):
        target_correlation = {
            column: abs(float(matrix[column].corr(target))) for column in candidates
        }
    target_correlation = {
        column: 0.0 if not np.isfinite(value) else value
        for column, value in target_correlation.items()
    }

    correlations = matrix[candidates].corr().abs()

    for i, column in enumerate(candidates):
        if column in dropped:
            continue
        for other in candidates[i + 1 :]:
            if other in dropped or other in protected:
                continue
            value = float(correlations.loc[column, other])
            if not np.isfinite(value) or value <= max_correlation:
                continue
            # Keep whichever of the two tracks the target more closely; a
            # protected column always wins.
            if column in protected or target_correlation[column] >= target_correlation[other]:
                loser, winner = other, column
            else:
                loser, winner = column, other
            dropped[loser] = (
                f"|r|={value:.3f} with {winner}, which tracks the target more closely"
            )
            if loser == column:
                break

    kept = tuple(column for column in candidates if column not in dropped)
    return SelectionResult(
        kept=kept, dropped=dropped, max_correlation=max_correlation
    )


def apply(matrix: pd.DataFrame, result: SelectionResult) -> pd.DataFrame:
    """Reduce a matrix to the selected columns, in the selected order."""
    missing = [column for column in result.kept if column not in matrix.columns]
    if missing:
        raise ValueError(f"matrix is missing selected columns: {missing}")
    return matrix[list(result.kept)]


__all__ = [
    "MAX_CORRELATION",
    "MIN_VARIANCE",
    "PROTECTED",
    "SelectionResult",
    "apply",
    "select",
]
