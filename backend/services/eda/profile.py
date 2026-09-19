"""Automated statistical profile — FEAT-02 (tasks 4.1, 4.2, 4.4; AC-3).

Three layers, each answering a question an analyst actually asks:

* **Univariate** (4.1) — what does each column look like on its own? Centre,
  spread, shape, and how much of it is missing.
* **Bivariate** (4.2) — what moves with what? Pearson *and* Spearman, because
  they disagree in an informative way: a large Spearman with a small Pearson is
  a monotone relationship that is not a straight line.
* **Distribution** (4.4) — which columns are skewed enough to want a log
  transform, and does the transform actually help?

Two habits carried through from earlier phases:

**Every number travels with its sample size.** A correlation computed on 40
overlapping rows and one computed on 30,000 are different claims, and a bare
coefficient hides which one you have.

**No causal language anywhere** (ETH-1). The bivariate section reports
association and says plainly that association is not causation. That is not
decoration: a correlation heatmap is the single most misread artefact in an
EDA dashboard, so the caveat ships inside the payload rather than being left to
whoever writes the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from services.datasets import MEASUREMENT_COLUMNS

# |skew| above this is "highly skewed" in the usual rule of thumb; between
# MODERATE and HIGH it is worth noting but rarely worth transforming.
HIGH_SKEW = 1.0
MODERATE_SKEW = 0.5

# A log transform has to earn its place: it is only recommended when it
# actually reduces |skew| by at least this much. Recommending one that does not
# help is cargo cult, and it costs interpretability.
MIN_SKEW_IMPROVEMENT = 0.2

CORRELATION_METHODS = ("pearson", "spearman")

CORRELATION_CAVEAT = (
    "Association only. These coefficients do not establish causation, and a "
    "confounder shared by two columns will raise their correlation without any "
    "causal link between them (ETH-1)."
)

NORMALITY_CAVEAT = (
    "With samples this large, a normality test rejects on deviations too small "
    "to matter. Read the statistic and the skew/kurtosis, not the p-value alone."
)


@dataclass(frozen=True, slots=True)
class UnivariateStats:
    """Centre, spread, shape and completeness for one column (task 4.1)."""

    column: str
    count: int
    missing: int
    missing_pct: float
    mean: float | None
    median: float | None
    std: float | None
    variance: float | None
    minimum: float | None
    maximum: float | None
    q1: float | None
    q3: float | None
    iqr: float | None
    p05: float | None
    p95: float | None
    skewness: float | None
    kurtosis: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": _round(self.missing_pct),
            "mean": _round(self.mean),
            "median": _round(self.median),
            "std": _round(self.std),
            "variance": _round(self.variance),
            "min": _round(self.minimum),
            "max": _round(self.maximum),
            "q1": _round(self.q1),
            "q3": _round(self.q3),
            "iqr": _round(self.iqr),
            "p05": _round(self.p05),
            "p95": _round(self.p95),
            "skewness": _round(self.skewness),
            "kurtosis": _round(self.kurtosis),
        }


@dataclass(frozen=True, slots=True)
class CorrelationPair:
    """One pair, both methods, and the overlap they were computed on."""

    a: str
    b: str
    pearson: float | None
    spearman: float | None
    sample_size: int

    @property
    def divergence(self) -> float | None:
        """How far the two methods disagree.

        Large divergence is the interesting case: a strong Spearman with a weak
        Pearson means the relationship is monotone but curved, which a linear
        model will under-fit.
        """
        if self.pearson is None or self.spearman is None:
            return None
        return abs(self.spearman - self.pearson)

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "pearson": _round(self.pearson),
            "spearman": _round(self.spearman),
            "sample_size": self.sample_size,
            "divergence": _round(self.divergence),
        }


@dataclass(frozen=True, slots=True)
class BivariateProfile:
    """Correlation matrices plus the pairs worth looking at (task 4.2)."""

    columns: tuple[str, ...]
    pearson: dict[str, dict[str, float | None]]
    spearman: dict[str, dict[str, float | None]]
    pairs: tuple[CorrelationPair, ...]
    caveat: str = CORRELATION_CAVEAT

    def strongest(self, limit: int = 5) -> list[CorrelationPair]:
        ranked = [p for p in self.pairs if p.pearson is not None]
        ranked.sort(key=lambda p: abs(p.pearson or 0.0), reverse=True)
        return ranked[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "pearson": self.pearson,
            "spearman": self.spearman,
            "pairs": [p.as_dict() for p in self.pairs],
            "caveat": self.caveat,
        }


@dataclass(frozen=True, slots=True)
class DistributionAssessment:
    """Shape of one column, and whether a log transform would help (task 4.4)."""

    column: str
    skewness: float | None
    kurtosis: float | None
    shape: str
    is_strictly_positive: bool
    log_skewness: float | None
    recommend_log_transform: bool
    rationale: str
    normality_statistic: float | None = None
    normality_p_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "skewness": _round(self.skewness),
            "kurtosis": _round(self.kurtosis),
            "shape": self.shape,
            "is_strictly_positive": self.is_strictly_positive,
            "log_skewness": _round(self.log_skewness),
            "recommend_log_transform": self.recommend_log_transform,
            "rationale": self.rationale,
            "normality_statistic": _round(self.normality_statistic),
            "normality_p_value": (
                None if self.normality_p_value is None else float(f"{self.normality_p_value:.3g}")
            ),
        }


@dataclass(frozen=True, slots=True)
class StatisticalProfile:
    """The full payload behind ``POST /eda/profile`` (AC-3)."""

    rows: int
    univariate: tuple[UnivariateStats, ...]
    bivariate: BivariateProfile
    distributions: tuple[DistributionAssessment, ...]
    caveats: tuple[str, ...] = field(default=())

    def by_column(self) -> dict[str, UnivariateStats]:
        return {stat.column: stat for stat in self.univariate}

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "univariate": [u.as_dict() for u in self.univariate],
            "bivariate": self.bivariate.as_dict(),
            "distributions": [d.as_dict() for d in self.distributions],
            "caveats": list(self.caveats),
        }


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _finite(series: pd.Series) -> pd.Series:
    """Observed, finite values only.

    ``dropna`` alone leaves ``inf``, which survives into a mean and turns a
    whole column of statistics into ``inf`` without any obvious cause.
    """
    clean = series.dropna()
    return clean[np.isfinite(clean)]


# --- 4.1 Univariate ---------------------------------------------------------


def univariate(frame: pd.DataFrame, column: str) -> UnivariateStats:
    """Descriptive statistics for one column, including its missingness (AC-3)."""
    total = len(frame)
    values = _finite(frame[column]) if total else pd.Series(dtype="float64")
    missing = total - len(values)

    if values.empty:
        return UnivariateStats(
            column=column,
            count=0,
            missing=missing,
            missing_pct=100.0 if total else 0.0,
            mean=None,
            median=None,
            std=None,
            variance=None,
            minimum=None,
            maximum=None,
            q1=None,
            q3=None,
            iqr=None,
            p05=None,
            p95=None,
            skewness=None,
            kurtosis=None,
        )

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))

    return UnivariateStats(
        column=column,
        count=int(len(values)),
        missing=int(missing),
        missing_pct=(missing / total * 100) if total else 0.0,
        mean=float(values.mean()),
        median=float(values.median()),
        # ddof=1: this is a sample, not the population of all possible readings.
        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        variance=float(values.var(ddof=1)) if len(values) > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        p05=float(values.quantile(0.05)),
        p95=float(values.quantile(0.95)),
        # pandas uses the bias-corrected Fisher-Pearson coefficient for skew and
        # *excess* kurtosis, so 0 is the normal distribution, not 3.
        skewness=float(values.skew()) if len(values) > 2 else None,
        kurtosis=float(values.kurt()) if len(values) > 3 else None,
    )


def univariate_profile(
    frame: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
) -> tuple[UnivariateStats, ...]:
    return tuple(univariate(frame, column) for column in columns)


# --- 4.2 Bivariate ----------------------------------------------------------


def _pairwise_overlap(frame: pd.DataFrame, a: str, b: str) -> int:
    return int((frame[a].notna() & frame[b].notna()).sum())


def bivariate_profile(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    *,
    min_overlap: int = 3,
) -> BivariateProfile:
    """Pearson and Spearman correlation matrices (task 4.2).

    Both methods, because they answer different questions: Pearson measures
    linear association and is pulled hard by outliers; Spearman measures
    monotone association on ranks and is not. Environmental data has both
    curvature and extremes, so reporting only one would be misleading.
    """
    usable = [c for c in columns if c in frame.columns]
    matrices: dict[str, dict[str, dict[str, float | None]]] = {}

    for method in CORRELATION_METHODS:
        if frame.empty or len(usable) < 2:
            matrices[method] = {a: {b: None for b in usable} for a in usable}
            continue
        # pandas computes pairwise-complete correlations, so a column with gaps
        # does not discard every row it appears in.
        matrix = frame[usable].corr(method=method, min_periods=min_overlap)
        matrices[method] = {
            a: {b: _round(matrix.loc[a, b]) for b in usable} for a in usable
        }

    pairs = []
    for index, a in enumerate(usable):
        for b in usable[index + 1 :]:
            pairs.append(
                CorrelationPair(
                    a=a,
                    b=b,
                    pearson=matrices["pearson"][a][b],
                    spearman=matrices["spearman"][a][b],
                    sample_size=_pairwise_overlap(frame, a, b) if not frame.empty else 0,
                )
            )

    return BivariateProfile(
        columns=tuple(usable),
        pearson=matrices["pearson"],
        spearman=matrices["spearman"],
        pairs=tuple(pairs),
    )


# --- 4.4 Distribution and transform advice ----------------------------------


def _shape_label(skew: float | None) -> str:
    if skew is None:
        return "unknown"
    magnitude = abs(skew)
    direction = "right" if skew > 0 else "left"
    if magnitude < MODERATE_SKEW:
        return "approximately symmetric"
    if magnitude < HIGH_SKEW:
        return f"moderately {direction}-skewed"
    return f"highly {direction}-skewed"


def assess_distribution(frame: pd.DataFrame, column: str) -> DistributionAssessment:
    """Is this column skewed, and would ``log1p`` actually fix it? (task 4.4)

    The recommendation is *verified*, not assumed: the transform is applied,
    the skew recomputed, and it is only recommended when the result is
    materially better. A log transform costs interpretability -- coefficients
    stop being in µg/m³ -- so it has to buy something.
    """
    values = _finite(frame[column]) if len(frame) else pd.Series(dtype="float64")

    if len(values) < 3:
        return DistributionAssessment(
            column=column,
            skewness=None,
            kurtosis=None,
            shape="unknown",
            is_strictly_positive=False,
            log_skewness=None,
            recommend_log_transform=False,
            rationale="Too few observations to characterise the distribution.",
        )

    skew = float(values.skew())
    kurt = float(values.kurt()) if len(values) > 3 else None
    non_negative = bool(values.min() >= 0)

    statistic: float | None = None
    p_value: float | None = None
    if len(values) >= 20:
        # D'Agostino-Pearson: combines skew and kurtosis, and unlike Shapiro-Wilk
        # it has no upper sample-size limit.
        result = stats.normaltest(values.to_numpy())
        statistic, p_value = float(result.statistic), float(result.pvalue)

    log_skew: float | None = None
    recommend = False

    if non_negative:
        # log1p, not log: it is defined at zero, and a zero concentration is a
        # legitimate reading rather than something to drop.
        log_skew = float(pd.Series(np.log1p(values.to_numpy())).skew())

    if log_skew is None:
        rationale = (
            f"{_shape_label(skew)} (skew={skew:.2f}). A log transform is not "
            "applicable: the column contains negative values."
        )
    elif abs(skew) < HIGH_SKEW:
        rationale = (
            f"{_shape_label(skew)} (skew={skew:.2f}). Not skewed enough to "
            "justify the loss of interpretability a transform costs."
        )
    elif abs(skew) - abs(log_skew) < MIN_SKEW_IMPROVEMENT:
        rationale = (
            f"{_shape_label(skew)} (skew={skew:.2f}), but log1p only moves it to "
            f"{log_skew:.2f}. Not recommended: the transform would cost "
            "interpretability without fixing the shape."
        )
    else:
        recommend = True
        rationale = (
            f"{_shape_label(skew)} (skew={skew:.2f}); log1p reduces it to "
            f"{log_skew:.2f}. Recommended before any model that assumes "
            "roughly symmetric residuals."
        )

    return DistributionAssessment(
        column=column,
        skewness=skew,
        kurtosis=kurt,
        shape=_shape_label(skew),
        is_strictly_positive=non_negative,
        log_skewness=log_skew,
        recommend_log_transform=recommend,
        rationale=rationale,
        normality_statistic=statistic,
        normality_p_value=p_value,
    )


def distribution_profile(
    frame: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
) -> tuple[DistributionAssessment, ...]:
    return tuple(assess_distribution(frame, column) for column in columns)


# --- The whole profile ------------------------------------------------------


def build(
    frame: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
) -> StatisticalProfile:
    """Univariate, bivariate and distribution analysis in one pass (FEAT-02)."""
    caveats = [CORRELATION_CAVEAT]
    if len(frame) >= 5000:
        caveats.append(NORMALITY_CAVEAT)

    return StatisticalProfile(
        rows=len(frame),
        univariate=univariate_profile(frame, columns),
        bivariate=bivariate_profile(frame, columns),
        distributions=distribution_profile(frame, columns),
        caveats=tuple(caveats),
    )


__all__ = [
    "CORRELATION_CAVEAT",
    "CORRELATION_METHODS",
    "HIGH_SKEW",
    "MIN_SKEW_IMPROVEMENT",
    "MODERATE_SKEW",
    "NORMALITY_CAVEAT",
    "BivariateProfile",
    "CorrelationPair",
    "DistributionAssessment",
    "StatisticalProfile",
    "UnivariateStats",
    "assess_distribution",
    "bivariate_profile",
    "build",
    "distribution_profile",
    "univariate",
    "univariate_profile",
]
