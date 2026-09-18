"""Outlier detection (tasks 3.4-3.7, specs §5.3, design §7).

Three detectors run in parallel and vote into one ``is_anomaly`` flag:

| Detector         | Rule                                    | Sees          |
|------------------|-----------------------------------------|---------------|
| IQR              | outside ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``| one column    |
| Z-score          | ``|z| > 3``                             | one column    |
| Isolation Forest | multivariate anomaly score              | every column  |

**Flag, never delete.** design §7 calls this the critical design decision, and
it is worth restating why: a PM2.5 spike may be the single most informative
record in the dataset -- a stubble-burning night, a festival, a still winter
inversion. Deleting it destroys exactly the signal the platform exists to
explain. Nothing in this module removes a row (AC-5).

**What a flag does and does not mean.** These detectors find *statistical*
outliers. They cannot tell a faulty sensor from a genuine pollution episode --
both look identical in the numbers -- and no amount of tuning would change
that. Distinguishing them needs domain knowledge, which is precisely why the
decision is left to a human and the record is kept.

Two choices that keep the flag meaningful:

* **Bounds are computed per station.** A city-wide IQR on a real pollution
  gradient flags the dirtiest district wholesale rather than any anomaly.
* **An imputed value is never flagged.** The detectors run on the imputed
  matrix, because Isolation Forest needs a complete one, but a column can only
  contribute a vote where the original reading was actually observed.
  Otherwise the engine would flag its own invention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from services.datasets import MEASUREMENT_COLUMNS, STATION_COLUMN

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0

# specs §5.3 names three detectors and design §7 says they vote. A majority is
# the useful reading of "vote": one detector alone is noisy, and requiring all
# three would mean the multivariate detector could veto both univariate ones.
DEFAULT_MIN_VOTES = 2

RANDOM_STATE = 42
ISOLATION_FOREST_ESTIMATORS = 200

# An explicit rate, not "auto". scikit-learn's automatic threshold flagged 24%
# of this dataset -- a detector that calls a quarter of the data anomalous is
# not contributing information to the vote. 2% is a defensible prior for
# environmental telemetry and keeps the multivariate detector selective enough
# that its vote means something.
DEFAULT_CONTAMINATION = 0.02

DETECTORS = ("iqr", "zscore", "isolation_forest")


@dataclass(frozen=True, slots=True)
class AnomalySummary:
    rows: int
    flagged: int
    flagged_pct: float
    votes_by_detector: dict[str, int]
    flagged_by_column: dict[str, int]
    min_votes: int
    note: str = (
        "Statistical outliers only. A flag does not distinguish a sensor fault "
        "from a genuine pollution event; records are retained for that reason "
        "(AC-5)."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "flagged": self.flagged,
            "flagged_pct": round(self.flagged_pct, 3),
            "votes_by_detector": self.votes_by_detector,
            "flagged_by_column": self.flagged_by_column,
            "min_votes": self.min_votes,
            "note": self.note,
        }


def iqr_bounds(values: pd.Series, multiplier: float = IQR_MULTIPLIER) -> tuple[float, float]:
    """Boxplot bounds ``[Q1 - k·IQR, Q3 + k·IQR]`` (task 3.4)."""
    clean = values.dropna()
    if clean.empty:
        return (float("-inf"), float("inf"))

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    spread = q3 - q1
    if spread == 0:
        # A constant column has no spread to reason about; flagging every
        # distinct value would be noise, not detection.
        return (float("-inf"), float("inf"))

    return (q1 - multiplier * spread, q3 + multiplier * spread)


def flag_iqr(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    multiplier: float = IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Per-column IQR flags, with bounds fitted per station (task 3.4)."""
    flags = pd.DataFrame(False, index=frame.index, columns=list(columns))

    for _, group in frame.groupby(STATION_COLUMN, sort=False):
        for column in columns:
            low, high = iqr_bounds(group[column], multiplier)
            values = group[column]
            flags.loc[group.index, column] = (values < low) | (values > high)

    return flags.fillna(False).astype(bool)


def flag_zscore(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    threshold: float = ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Per-column ``|z| > threshold`` flags, per station (task 3.5)."""
    flags = pd.DataFrame(False, index=frame.index, columns=list(columns))

    for _, group in frame.groupby(STATION_COLUMN, sort=False):
        for column in columns:
            values = group[column]
            std = values.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                continue
            z = (values - values.mean()) / std
            flags.loc[group.index, column] = z.abs() > threshold

    return flags.fillna(False).astype(bool)


def flag_isolation_forest(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    contamination: float | str = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Multivariate anomaly flags (task 3.6).

    Row-level rather than column-level: the whole point of the multivariate
    detector is to catch combinations that no single column would reveal --
    heavy traffic with unusually clean air, say. It needs a complete matrix,
    which is why it runs after imputation.
    """
    usable = [c for c in columns if frame[c].notna().any()]
    if frame.empty or len(usable) < 2 or len(frame) < 10:
        return pd.Series(False, index=frame.index, dtype=bool)

    matrix = frame[usable].astype("float64")
    if matrix.isna().to_numpy().any():
        matrix = matrix.fillna(matrix.median())

    forest = IsolationForest(
        n_estimators=ISOLATION_FOREST_ESTIMATORS,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,  # determinism over speed: this dataset is small
    )
    predictions = forest.fit_predict(matrix.to_numpy())
    return pd.Series(predictions == -1, index=frame.index, dtype=bool)


def combine(
    iqr_flags: pd.DataFrame,
    zscore_flags: pd.DataFrame,
    forest_flags: pd.Series,
    *,
    observed_mask: pd.DataFrame | None = None,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> tuple[pd.Series, AnomalySummary]:
    """Vote the three detectors into one flag (task 3.7).

    ``observed_mask`` marks where the original value was present. Where it is
    False the univariate detectors are silenced for that cell, so an imputed
    value cannot be reported as an anomaly.
    """
    if observed_mask is not None:
        iqr_flags = iqr_flags & observed_mask
        zscore_flags = zscore_flags & observed_mask

    iqr_row = iqr_flags.any(axis=1)
    zscore_row = zscore_flags.any(axis=1)

    votes = iqr_row.astype(int) + zscore_row.astype(int) + forest_flags.astype(int)
    flagged = votes >= min_votes

    per_column = (iqr_flags | zscore_flags).loc[flagged]
    summary = AnomalySummary(
        rows=len(votes),
        flagged=int(flagged.sum()),
        flagged_pct=(float(flagged.sum()) / len(votes) * 100) if len(votes) else 0.0,
        votes_by_detector={
            "iqr": int(iqr_row.sum()),
            "zscore": int(zscore_row.sum()),
            "isolation_forest": int(forest_flags.sum()),
        },
        flagged_by_column={
            column: int(per_column[column].sum()) for column in iqr_flags.columns
        },
        min_votes=min_votes,
    )
    return flagged, summary


def detect(
    frame: pd.DataFrame,
    *,
    observed_mask: pd.DataFrame | None = None,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> tuple[pd.Series, AnomalySummary]:
    """Run all three detectors and vote. Never removes a row (AC-5)."""
    if frame.empty:
        return pd.Series(dtype=bool), AnomalySummary(0, 0, 0.0, {}, {}, min_votes)

    iqr_flags = flag_iqr(frame, columns=columns)
    zscore_flags = flag_zscore(frame, columns=columns)
    forest_flags = flag_isolation_forest(frame, columns=columns)

    return combine(
        iqr_flags,
        zscore_flags,
        forest_flags,
        observed_mask=observed_mask,
        min_votes=min_votes,
    )


__all__ = [
    "DEFAULT_CONTAMINATION",
    "DEFAULT_MIN_VOTES",
    "DETECTORS",
    "IQR_MULTIPLIER",
    "ZSCORE_THRESHOLD",
    "AnomalySummary",
    "combine",
    "detect",
    "flag_iqr",
    "flag_isolation_forest",
    "flag_zscore",
    "iqr_bounds",
]
