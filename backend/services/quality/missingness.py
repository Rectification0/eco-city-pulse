"""Missingness analysis (task 3.1, specs §5.3, design §7).

design §7: the MCAR/MAR/MNAR judgement is "the academic justification for the
imputation choice, not an internal detail". So it is computed, reported, and --
importantly -- bounded by what the data can actually support.

**What can and cannot be inferred.** This is the one place in the project where
the honest answer is narrower than the requirement's wording, so it is stated
plainly rather than buried:

* **MAR** is testable. If a column's missingness is associated with the
  *observed* values of other columns, the missingness is explained by data we
  have. That is exactly the condition under which MICE is the right repair.
* **MCAR** is what remains when no such association is found.
* **MNAR** -- missingness that depends on the *unobserved* value itself -- is
  **not identifiable from the data alone**. No test can distinguish it from
  MCAR, because the evidence that would separate them is precisely the data
  that is missing. This is a standard result, not a limitation of this
  implementation.

So this module never infers MNAR. It reports MCAR or MAR from the tests, always
attaches the caveat that MNAR cannot be excluded, and accepts MNAR only as an
explicitly declared domain rule with a stated justification. Labelling a column
MNAR because a statistic looked a certain way would be the kind of overclaim
ETH-1 rules out everywhere else in this platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from services.datasets import MEASUREMENT_COLUMNS, STATION_COLUMN

# Bonferroni-corrected family-wise error rate. Each column is tested against
# several candidates, so an uncorrected 0.05 would manufacture an association
# for roughly one column in four by chance alone.
ALPHA = 0.05

# How far the discrimination has to sit from 0.5 before an association counts
# as practically meaningful. With 30k rows, p < 0.05 is easy to reach; an AUC of
# 0.502 still means nothing.
MIN_EFFECT_SIZE = 0.05

# Minimum group sizes for the rank test to say anything.
MIN_GROUP_SIZE = 10


class Mechanism(str, Enum):
    COMPLETE = "complete"
    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"


@dataclass(frozen=True, slots=True)
class Association:
    """One test of "is this column's missingness predictable from that variable?".

    ``discrimination`` is the area under the ROC curve for using the variable to
    separate missing rows from observed ones: 0.5 is no association, and the
    distance from 0.5 is the effect size. It is the Mann-Whitney U statistic
    rescaled, so it comes free with the test.

    A **rank** test, not a correlation, and that choice matters. Missingness
    driven by a tail -- a sensor that saturates above some concentration -- is a
    huge effect that linear correlation barely registers: a 20x lift in the top
    few percent of a predictor can show up as r = 0.01 while the AUC moves
    decisively. Using Pearson here would have reported MCAR for a mechanism that
    is plainly not random.
    """

    related_to: str
    discrimination: float
    p_value: float
    sample_size: int
    significant: bool

    @property
    def effect(self) -> float:
        return abs(self.discrimination - 0.5)

    @property
    def direction(self) -> str:
        if self.discrimination > 0.5:
            return "higher"
        return "lower" if self.discrimination < 0.5 else "unchanged"

    def as_dict(self) -> dict[str, Any]:
        return {
            "related_to": self.related_to,
            "discrimination": round(self.discrimination, 4),
            "direction": self.direction,
            "p_value": float(f"{self.p_value:.3g}"),
            "sample_size": self.sample_size,
            "significant": self.significant,
        }


@dataclass(frozen=True, slots=True)
class ColumnMissingness:
    """Per-column missingness, its mechanism, and the evidence behind it."""

    column: str
    total: int
    missing: int
    missing_pct: float
    longest_gap_rows: int
    mechanism: Mechanism
    associations: tuple[Association, ...] = ()
    justification: str = ""
    caveat: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "total": self.total,
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 3),
            "longest_gap_rows": self.longest_gap_rows,
            "mechanism": self.mechanism.value,
            "associations": [a.as_dict() for a in self.associations],
            "justification": self.justification,
            "caveat": self.caveat,
        }


@dataclass(frozen=True, slots=True)
class GridCompleteness:
    """Hours absent from a station's grid entirely.

    Distinct from a NaN: a NaN is a row that exists with no reading, an absent
    hour is a row that was never written. Both are missing data, but only the
    first is visible to a column-wise null count, so reporting them together
    prevents an "only 3% missing" claim about a feed that skipped a fortnight.
    """

    expected_hours: int
    present_hours: int
    absent_hours: int
    completeness_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_hours": self.expected_hours,
            "present_hours": self.present_hours,
            "absent_hours": self.absent_hours,
            "completeness_pct": round(self.completeness_pct, 3),
        }


@dataclass(frozen=True, slots=True)
class CoMissingness:
    """Columns that tend to go missing together.

    A structural diagnostic rather than part of the mechanism verdict: when
    four sensors drop out in the same hours, the cause is a station outage, not
    four independent quirks. That is what tells an operator to look at the
    hardware, and it is why the pipeline fills short gaps locally before
    reaching for a model -- during an outage there is no correlated column left
    for MICE to condition on.
    """

    column: str
    also_missing: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "also_missing": {k: round(v, 4) for k, v in self.also_missing.items()},
        }


@dataclass(frozen=True, slots=True)
class MissingnessReport:
    columns: tuple[ColumnMissingness, ...]
    grid: GridCompleteness
    co_missingness: tuple[CoMissingness, ...] = ()
    caveats: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": [c.as_dict() for c in self.columns],
            "grid": self.grid.as_dict(),
            "co_missingness": [c.as_dict() for c in self.co_missingness],
            "caveats": list(self.caveats),
        }

    def by_column(self) -> dict[str, ColumnMissingness]:
        return {c.column: c for c in self.columns}


MNAR_CAVEAT = (
    "MNAR cannot be ruled out: whether missingness depends on the unobserved "
    "value is not identifiable from the observed data, and requires domain "
    "knowledge of the instrument."
)


def longest_null_run(series: pd.Series) -> int:
    """Longest run of consecutive NaNs, which is what decides whether a gap is
    short enough to fill locally (task 3.3) or belongs to MICE (task 3.2)."""
    null = series.isna().to_numpy()
    if not null.any():
        return 0

    longest = current = 0
    for is_null in null:
        current = current + 1 if is_null else 0
        longest = max(longest, current)
    return longest


def _longest_gap_per_station(frame: pd.DataFrame, column: str) -> int:
    """Gaps are per-station: a run is only contiguous within one sensor."""
    if frame.empty:
        return 0
    return int(
        max(
            (longest_null_run(group[column]) for _, group in frame.groupby(STATION_COLUMN)),
            default=0,
        )
    )


def _candidate_predictors(frame: pd.DataFrame, column: str) -> dict[str, pd.Series]:
    """Observed variables that a column's missingness might depend on.

    ``hour_of_day`` is included because a feed that drops out at the same time
    every day is emphatically not missing at random, and no measurement column
    would reveal that.
    """
    candidates: dict[str, pd.Series] = {
        other: frame[other] for other in MEASUREMENT_COLUMNS if other != column
    }
    candidates["hour_of_day"] = frame["timestamp"].dt.hour.astype("float64")
    return candidates


def _test_association(indicator: np.ndarray, values: pd.Series) -> Association | None:
    """Mann-Whitney U on the predictor, split by whether the column is missing.

    Rows where the *predictor* is itself missing are excluded: otherwise the
    test would measure the overlap of two missingness patterns rather than the
    value-based relationship the mechanism question asks about. Co-missingness
    is reported separately, by ``co_missingness``.
    """
    usable = values.notna().to_numpy()
    if usable.sum() < MIN_GROUP_SIZE * 2:
        return None

    observed = values.to_numpy()[usable]
    is_missing = indicator[usable].astype(bool)

    when_missing = observed[is_missing]
    when_present = observed[~is_missing]

    if len(when_missing) < MIN_GROUP_SIZE or len(when_present) < MIN_GROUP_SIZE:
        return None
    if np.allclose(observed, observed[0]):
        return None

    result = stats.mannwhitneyu(when_missing, when_present, alternative="two-sided")
    discrimination = float(result.statistic) / (len(when_missing) * len(when_present))
    p_value = float(result.pvalue)

    if np.isnan(discrimination) or np.isnan(p_value):
        return None

    return Association(
        related_to=str(values.name),
        discrimination=discrimination,
        p_value=p_value,
        sample_size=int(usable.sum()),
        significant=False,  # decided by the caller, which knows the test count
    )


def analyse_column(frame: pd.DataFrame, column: str) -> ColumnMissingness:
    """Counts, longest gap, and the mechanism judgement for one column."""
    total = len(frame)
    missing = int(frame[column].isna().sum())
    missing_pct = (missing / total * 100) if total else 0.0
    longest_gap = _longest_gap_per_station(frame, column)

    if missing == 0:
        return ColumnMissingness(
            column=column,
            total=total,
            missing=0,
            missing_pct=0.0,
            longest_gap_rows=0,
            mechanism=Mechanism.COMPLETE,
            justification="No missing values; no mechanism to characterise.",
        )

    indicator = frame[column].isna().to_numpy().astype(float)
    candidates = _candidate_predictors(frame, column)

    raw = [
        association
        for name, series in candidates.items()
        if (association := _test_association(indicator, series.rename(name))) is not None
    ]

    if not raw:
        return ColumnMissingness(
            column=column,
            total=total,
            missing=missing,
            missing_pct=missing_pct,
            longest_gap_rows=longest_gap,
            mechanism=Mechanism.MCAR,
            justification=(
                "No predictor had enough observed data to test against; "
                "treated as MCAR for want of evidence."
            ),
            caveat=MNAR_CAVEAT,
        )

    # Bonferroni across the tests actually run for this column.
    threshold = ALPHA / len(raw)
    assessed = tuple(
        Association(
            related_to=a.related_to,
            discrimination=a.discrimination,
            p_value=a.p_value,
            sample_size=a.sample_size,
            significant=a.p_value < threshold and a.effect >= MIN_EFFECT_SIZE,
        )
        for a in raw
    )
    significant = [a for a in assessed if a.significant]

    if significant:
        strongest = max(significant, key=lambda a: a.effect)
        return ColumnMissingness(
            column=column,
            total=total,
            missing=missing,
            missing_pct=missing_pct,
            longest_gap_rows=longest_gap,
            mechanism=Mechanism.MAR,
            associations=assessed,
            justification=(
                f"When this column is missing, observed {strongest.related_to} runs "
                f"{strongest.direction} (AUC={strongest.discrimination:.3f}, "
                f"p={strongest.p_value:.2g}). Missingness is therefore predictable "
                "from data that is present, which is the definition of MAR and the "
                "condition under which MICE is the right repair: it conditions on "
                "exactly those variables, where a mean fill would discard the "
                "relationship."
            ),
            caveat=MNAR_CAVEAT,
        )

    return ColumnMissingness(
        column=column,
        total=total,
        missing=missing,
        missing_pct=missing_pct,
        longest_gap_rows=longest_gap,
        mechanism=Mechanism.MCAR,
        associations=assessed,
        justification=(
            "No observed variable predicts this column's missingness at the "
            f"corrected threshold (alpha={ALPHA}, Bonferroni over {len(assessed)} "
            "tests), which is consistent with missing completely at random."
        ),
        caveat=MNAR_CAVEAT,
    )


def co_missingness(
    frame: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
) -> tuple[CoMissingness, ...]:
    """For each column, how often each other column is missing at the same time."""
    results = []
    for column in columns:
        indicator = frame[column].isna()
        if not indicator.any():
            continue
        results.append(
            CoMissingness(
                column=column,
                also_missing={
                    other: float(frame.loc[indicator, other].isna().mean())
                    for other in columns
                    if other != column
                },
            )
        )
    return tuple(results)


def grid_completeness(frame: pd.DataFrame) -> GridCompleteness:
    """How much of the expected hourly grid is present at all (DR-4)."""
    if frame.empty:
        return GridCompleteness(0, 0, 0, 100.0)

    expected = 0
    for _, group in frame.groupby(STATION_COLUMN):
        span = group["timestamp"].max() - group["timestamp"].min()
        expected += int(span.total_seconds() // 3600) + 1

    present = len(frame)
    absent = max(0, expected - present)
    completeness = (present / expected * 100) if expected else 100.0

    return GridCompleteness(
        expected_hours=expected,
        present_hours=present,
        absent_hours=absent,
        completeness_pct=completeness,
    )


def analyse(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    declared_mnar: dict[str, str] | None = None,
) -> MissingnessReport:
    """Full missingness report (task 3.1).

    ``declared_mnar`` maps a column to the domain justification for calling it
    MNAR. It is the *only* route to that label, because the data cannot supply
    one -- see the module docstring.
    """
    declared_mnar = declared_mnar or {}

    results = []
    for column in columns:
        result = analyse_column(frame, column)
        justification = declared_mnar.get(column)
        if justification and result.mechanism is not Mechanism.COMPLETE:
            result = ColumnMissingness(
                column=result.column,
                total=result.total,
                missing=result.missing,
                missing_pct=result.missing_pct,
                longest_gap_rows=result.longest_gap_rows,
                mechanism=Mechanism.MNAR,
                associations=result.associations,
                justification=f"Declared MNAR on domain grounds: {justification}",
                caveat=(
                    "Asserted from domain knowledge, not inferred from the data; "
                    "MNAR is not identifiable from observed values alone."
                ),
            )
        results.append(result)

    return MissingnessReport(
        columns=tuple(results),
        grid=grid_completeness(frame),
        co_missingness=co_missingness(frame, columns),
        caveats=(MNAR_CAVEAT,),
    )


__all__ = [
    "ALPHA",
    "MIN_EFFECT_SIZE",
    "MNAR_CAVEAT",
    "MIN_GROUP_SIZE",
    "Association",
    "CoMissingness",
    "ColumnMissingness",
    "GridCompleteness",
    "Mechanism",
    "MissingnessReport",
    "analyse",
    "analyse_column",
    "co_missingness",
    "grid_completeness",
    "longest_null_run",
]
