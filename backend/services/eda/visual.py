"""Visual EDA — bivariate and multivariate plots (tasks 12.2–12.6, specs §6.4).

The profile (Phase 4) says *how strongly* two columns move together; t-SNE and
parallel coordinates (Phases 6, 10) project every column at once. Neither shows
the raw shape of one relationship, or how a measurement differs between groups.
This module computes the five charts that do:

- ``scatter``    — VIZ-1: two measurements, OLS line, r / ρ / r², n
- ``grouped``    — VIZ-2 and VIZ-3: per-group means with 95% CIs, and box
  summaries, from one ``groupby``
- ``pair_plot``  — VIZ-4: every measurement against every other
- ``andrews``    — VIZ-5: each hour as a Fourier curve, per class

Pure functions: a frame in, a frozen dataclass with ``as_dict()`` out, no I/O.
Scope, caching and loading belong to ``service``; drawing belongs to the
browser. That split is what lets every number here be tested offline and reused
verbatim by the HTML report (design §12.1).

**Statistics on every row, points from a sample** (VIZ-6). r, OLS coefficients,
means, CIs and quartiles use every complete row in scope. Only the marks sent
to the browser are thinned, with ``manifold.subsample`` — an even, deterministic
stride — because a random sample would redraw the chart on each refresh and a
reader comparing two screenshots would see movement that is not in the data.
Every payload says how many rows it summarises and how many points it carries.

**Anomalies are drawn, not dropped** (AC-5). Each point and curve carries its
``is_anomaly`` flag and every result counts flagged rows, so the outliers the
quality engine found remain visible on the chart that is meant to show them.

**No causal language** (ETH-1). Every result's ``caveats`` opens with the
profile's association-is-not-causation sentence, then adds what is specific to
the chart — a whisker is not an anomaly flag, an Andrews curve depends on column
order, a pair plot is a sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.exceptions import InsufficientDataError, SchemaValidationError
from services.datasets import MEASUREMENT_COLUMNS, STATION_COLUMN
from services.eda import bands, manifold, profile
from services.features import temporal
from services.features.spec import DEFAULT_LOCAL_OFFSET_MINUTES

# --- Groupings (task 12.2) ---------------------------------------------------

# Every categorical axis the charts can split on. Order is the order the UI
# lists them in: clock first, calendar next, place, then the PM2.5 band.
GROUPINGS: tuple[str, ...] = (
    "hour_of_day",
    "time_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "station",
    "pm25_band",
)

# Four six-hour bands of the *local* clock (specs §6.4). Six hours each so the
# classes are the same width; "night" ends at 05:59 because the morning rush in
# the demo city starts building at 06:00 IST.
TIME_OF_DAY: tuple[tuple[str, int, int], ...] = (
    ("night", 0, 5),
    ("morning", 6, 11),
    ("afternoon", 12, 17),
    ("evening", 18, 23),
)

DAY_NAMES: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
WEEKEND_LABELS: tuple[str, ...] = ("weekday", "weekend")
HOUR_LABELS: tuple[str, ...] = tuple(f"{hour:02d}:00" for hour in range(24))

# --- Chart limits (design §12.1) ---------------------------------------------

# scattergl draws 2,000 marks without effort; past that the cloud saturates
# into a blob and further points add bytes rather than information.
SCATTER_MAX_POINTS = 2000

# Five columns is ten lower-triangle panels, so 1,500 rows is 15,000 marks —
# near the limit of what Plotly's splom redraws smoothly. The same cap t-SNE
# uses, for the same reason.
PAIRPLOT_MAX_POINTS = manifold.DEFAULT_MAX_POINTS

# Below 30 the t-interval is wide enough that two bars "differ" by noise, and
# 30 is the conventional point where the t and normal intervals converge.
THIN_GROUP = 30
CONFIDENCE = 0.95

# Tukey's whisker multiplier, the same 1.5 the quality engine's IQR detector
# uses — though here it is applied per *group*, which is why it can disagree.
WHISKER_IQR = 1.5
BOX_MAX_OUTLIERS = 50

ANDREWS_T_POINTS = 128
ANDREWS_CURVES_PER_CLASS = 60

# Fixed, and returned with every payload. Early columns take the low-frequency
# terms, which shape the curve most, so the order *is* part of the picture:
# pollutants first, then their main co-variate (traffic), then weather. Ordering
# by PC1 loading would quietly redraw every curve each time the ESI refitted.
ANDREWS_COLUMNS: tuple[str, ...] = ("pm25", "pm10", "traffic_score", "temp", "humidity")

# Not pm25_band by default (see ``andrews``), and not hour_of_day or station:
# 24 or a dozen overlapping bundles are unreadable as curves.
ANDREWS_CLASSES: tuple[str, ...] = ("time_of_day", "pm25_band", "is_weekend", "month")

MIN_SCATTER_ROWS = 3
MIN_PAIRPLOT_ROWS = 3
MIN_ANDREWS_ROWS = 2

# --- Caveats (ETH-1) ---------------------------------------------------------

SCATTER_CAVEAT = (
    "r measures straight-line association only. A curve or a set of clusters "
    "can give a small r alongside a strong relationship, which is why the "
    "points are drawn rather than r alone."
)
GROUPED_CAVEAT = (
    "Whiskers are computed per group, for display only. The quality engine's "
    "anomaly flag is per station and needs two of three detectors to agree, so "
    "a point can sit past a whisker without being flagged, and the reverse; "
    "both counts are reported. Bars with fewer than 30 rows are marked thin: "
    "their interval is too wide to compare against."
)
PAIRPLOT_CAVEAT = (
    "The points are an even sample of the rows in scope; the correlation "
    "coefficients are computed on every row, so they match the heatmap."
)
ANDREWS_CAVEAT = (
    "Each curve is one hour's standardised readings drawn as a Fourier series. "
    "The shape depends on column order — earlier columns shape the curve more — "
    "and curves show how similar whole rows are, not what any one variable "
    "contributes to them."
)


def _number(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def _numbers(values: Any, digits: int = 3) -> list[float | None]:
    return [_number(value, digits) for value in values]


def _require_measurement(name: str, role: str) -> None:
    if name not in MEASUREMENT_COLUMNS:
        raise SchemaValidationError(
            f"unknown {role} {name!r}; available: {list(MEASUREMENT_COLUMNS)}",
            details={role: name, "available": list(MEASUREMENT_COLUMNS)},
        )


def _require_grouping(name: str, role: str, allowed: tuple[str, ...] = GROUPINGS) -> None:
    if name not in allowed:
        raise SchemaValidationError(
            f"unknown {role} {name!r}; available: {list(allowed)}",
            details={role: name, "available": list(allowed)},
        )


def _finite(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Rows where every named column holds a finite number.

    ``notna`` alone would keep ``inf``, which survives into a mean and an OLS
    fit and turns the whole chart into ``inf`` with no visible cause.
    """
    if frame.empty:
        return frame
    values = frame[list(columns)].to_numpy(dtype="float64")
    return frame[np.isfinite(values).all(axis=1)]


def _flags(frame: pd.DataFrame) -> pd.Series:
    if "is_anomaly" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["is_anomaly"].fillna(False).astype(bool)


def _ordered(labels: Any, categories: tuple[str, ...], index: pd.Index) -> pd.Series:
    return pd.Series(
        pd.Categorical(labels, categories=list(categories), ordered=True), index=index
    )


def grouping_series(
    frame: pd.DataFrame,
    name: str,
    *,
    offset_minutes: int = DEFAULT_LOCAL_OFFSET_MINUTES,
) -> pd.Series:
    """One grouping's label per row, as an ordered categorical (task 12.2).

    The calendar groupings come from ``features.temporal`` — the Phase 5
    transformer — so "09:00" means 09:00 IST here exactly as it does in the
    model's ``hour_of_day`` feature. A ``.dt.hour`` on the stored UTC timestamp
    would shift every diurnal peak by five and a half hours, and the grouped
    boxplot would put the evening rush in mid-afternoon.

    Labels rather than integers because these are axis ticks: "Sat" and
    "weekend" read without a legend, 5 and 1 do not. Rows that cannot be
    labelled (no PM2.5 for ``pm25_band``) are missing, never a pseudo-group.
    """
    _require_grouping(name, "grouping")

    if name == "pm25_band":
        return bands.band_labels(frame["pm25"])

    if name == "station":
        labels = frame[STATION_COLUMN].astype(str)
        return _ordered(labels, tuple(sorted(labels.unique())), frame.index)

    if frame.empty:
        return pd.Series(pd.Categorical([]), index=frame.index)

    calendar = temporal.add_temporal(frame[["timestamp"]], offset_minutes=offset_minutes)

    if name == "hour_of_day":
        return _ordered(
            np.array(HOUR_LABELS, dtype=object)[calendar["hour_of_day"].to_numpy()],
            HOUR_LABELS,
            frame.index,
        )
    if name == "time_of_day":
        hours = calendar["hour_of_day"].to_numpy()
        names = np.array([label for label, _, _ in TIME_OF_DAY], dtype=object)
        return _ordered(names[hours // 6], tuple(names), frame.index)
    if name == "day_of_week":
        return _ordered(
            np.array(DAY_NAMES, dtype=object)[calendar["day_of_week"].to_numpy()],
            DAY_NAMES,
            frame.index,
        )
    if name == "is_weekend":
        return _ordered(
            np.array(WEEKEND_LABELS, dtype=object)[calendar["is_weekend"].to_numpy()],
            WEEKEND_LABELS,
            frame.index,
        )
    # month
    return _ordered(
        np.array(MONTH_NAMES, dtype=object)[calendar["month"].to_numpy() - 1],
        MONTH_NAMES,
        frame.index,
    )


def _present(series: pd.Series) -> list[str]:
    """Categories that actually occur, in their defined order."""
    observed = set(series.dropna().astype(str))
    return [str(category) for category in series.cat.categories if str(category) in observed]


def _labels(series: pd.Series) -> list[str | None]:
    return [None if pd.isna(value) else str(value) for value in series]


# --- VIZ-1: scatter (task 12.3) ----------------------------------------------


@dataclass(frozen=True, slots=True)
class ScatterResult:
    """Two measurements: a sample of points, and a fit over every row."""

    x_column: str
    y_column: str
    color_by: str | None
    x: tuple[float, ...]
    y: tuple[float, ...]
    is_anomaly: tuple[bool, ...]
    groups: tuple[str | None, ...] | None
    categories: tuple[str, ...] | None
    slope: float | None
    intercept: float | None
    pearson: float | None
    spearman: float | None
    r_squared: float | None
    line_x: tuple[float, ...]
    line_y: tuple[float, ...]
    rows_used: int
    points_returned: int
    sampled: bool
    anomalies_in_rows: int
    anomalies_in_points: int
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "x_column": self.x_column,
            "y_column": self.y_column,
            "color_by": self.color_by,
            "x": _numbers(self.x),
            "y": _numbers(self.y),
            "is_anomaly": list(self.is_anomaly),
            "groups": None if self.groups is None else list(self.groups),
            "categories": None if self.categories is None else list(self.categories),
            "slope": _number(self.slope, 6),
            "intercept": _number(self.intercept, 6),
            "pearson": self.pearson,
            "spearman": self.spearman,
            "r_squared": _number(self.r_squared),
            "line_x": _numbers(self.line_x, 4),
            "line_y": _numbers(self.line_y, 4),
            "n": self.rows_used,
            "rows_used": self.rows_used,
            "points_returned": self.points_returned,
            "sampled": self.sampled,
            "anomalies_in_rows": self.anomalies_in_rows,
            "anomalies_in_points": self.anomalies_in_points,
            "caveats": list(self.caveats),
        }


def scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    color_by: str | None = None,
    max_points: int = SCATTER_MAX_POINTS,
) -> ScatterResult:
    """VIZ-1: ``y`` against ``x`` with an OLS line, r, ρ and r².

    Pearson and Spearman come from ``profile.bivariate_profile`` on the same
    frame, not from a second implementation here, so the r printed on a
    scatter is the r in the heatmap cell for that pair — to the last digit.
    Two numbers for one quantity would be the first thing a reader notices.

    r² is computed from the fit's residuals rather than by squaring r. For a
    one-variable OLS they are the same number, and a test holds them to it;
    computing it independently is what makes that test mean something.
    """
    _require_measurement(x, "x")
    _require_measurement(y, "y")
    if x == y:
        raise SchemaValidationError("x and y must be different columns", details={"x": x})
    if color_by is not None:
        _require_grouping(color_by, "color_by")

    complete = _finite(frame, (x, y))
    if len(complete) < MIN_SCATTER_ROWS:
        raise InsufficientDataError(
            f"a scatter needs at least {MIN_SCATTER_ROWS} rows with both "
            f"{x} and {y}; {len(complete)} available.",
            details={"required": MIN_SCATTER_ROWS, "available": len(complete)},
        )

    pair = profile.bivariate_profile(frame, (x, y)).pairs[0]

    xs = complete[x].to_numpy(dtype="float64")
    ys = complete[y].to_numpy(dtype="float64")

    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = None
    line_x: tuple[float, ...] = ()
    line_y: tuple[float, ...] = ()
    # A constant x has no slope to fit; polyfit would warn and return noise.
    if np.ptp(xs) > 0:
        slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
        residual = ys - (slope * xs + intercept)
        total = float(((ys - ys.mean()) ** 2).sum())
        r_squared = 1.0 - float((residual**2).sum()) / total if total > 0 else None
        # Two end points: a straight line needs no more, and a fitted value per
        # point would double the payload to draw the same segment.
        line_x = (float(xs.min()), float(xs.max()))
        line_y = tuple(slope * value + intercept for value in line_x)

    points, was_sampled = manifold.subsample(complete, max_points)
    flags = _flags(points)

    groups: tuple[str | None, ...] | None = None
    categories: tuple[str, ...] | None = None
    if color_by is not None:
        labels = grouping_series(points, color_by)
        groups = tuple(_labels(labels))
        categories = tuple(_present(labels))

    caveats = [profile.CORRELATION_CAVEAT, SCATTER_CAVEAT]
    if was_sampled:
        caveats.append(
            f"Showing {len(points):,} of {len(complete):,} points, sampled evenly; "
            "the line and coefficients use every row."
        )

    return ScatterResult(
        x_column=x,
        y_column=y,
        color_by=color_by,
        x=tuple(points[x].to_numpy(dtype="float64")),
        y=tuple(points[y].to_numpy(dtype="float64")),
        is_anomaly=tuple(bool(flag) for flag in flags),
        groups=groups,
        categories=categories,
        slope=slope,
        intercept=intercept,
        pearson=pair.pearson,
        spearman=pair.spearman,
        r_squared=r_squared,
        line_x=line_x,
        line_y=line_y,
        rows_used=len(complete),
        points_returned=len(points),
        sampled=was_sampled,
        anomalies_in_rows=int(_flags(complete).sum()),
        anomalies_in_points=int(flags.sum()),
        caveats=tuple(caveats),
    )


# --- VIZ-2 / VIZ-3: grouped bars and boxes (task 12.4) -----------------------


@dataclass(frozen=True, slots=True)
class BarCell:
    """One bar: the mean of a cell, with its spread and a t-interval."""

    group: str
    split: str | None
    n: int
    mean: float
    sd: float | None
    ci_low: float | None
    ci_high: float | None
    thin: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "split": self.split,
            "n": self.n,
            "mean": _number(self.mean),
            "sd": _number(self.sd),
            "ci_low": _number(self.ci_low),
            "ci_high": _number(self.ci_high),
            "thin": self.thin,
        }


@dataclass(frozen=True, slots=True)
class BoxSummary:
    """One box: quartiles, Tukey fences and the extremes past them."""

    group: str
    n: int
    mean: float
    q1: float
    median: float
    q3: float
    lower_fence: float
    upper_fence: float
    whisker_outliers: int
    outliers: tuple[float, ...]
    anomalies_flagged: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "n": self.n,
            "mean": _number(self.mean),
            "q1": _number(self.q1),
            "median": _number(self.median),
            "q3": _number(self.q3),
            "lower_fence": _number(self.lower_fence),
            "upper_fence": _number(self.upper_fence),
            "whisker_outliers": self.whisker_outliers,
            "outliers": _numbers(self.outliers),
            "anomalies_flagged": self.anomalies_flagged,
        }


@dataclass(frozen=True, slots=True)
class GroupedResult:
    """VIZ-2 and VIZ-3 together: they share every row and every group."""

    measure: str
    group_by: str
    split_by: str | None
    categories: tuple[str, ...]
    split_categories: tuple[str, ...] | None
    bars: tuple[BarCell, ...]
    boxes: tuple[BoxSummary, ...]
    rows_used: int
    anomalies_in_rows: int
    caveats: tuple[str, ...]
    confidence: float = CONFIDENCE
    thin_threshold: int = THIN_GROUP

    def as_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "group_by": self.group_by,
            "split_by": self.split_by,
            "categories": list(self.categories),
            "split_categories": (
                None if self.split_categories is None else list(self.split_categories)
            ),
            "bars": [bar.as_dict() for bar in self.bars],
            "boxes": [box.as_dict() for box in self.boxes],
            "rows_used": self.rows_used,
            "anomalies_in_rows": self.anomalies_in_rows,
            "confidence": self.confidence,
            "thin_threshold": self.thin_threshold,
            "caveats": list(self.caveats),
        }


def mean_interval(
    values: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float | None, float | None, float | None]:
    """Mean, sample sd, and the t-based interval ``mean ± t·s/√n``.

    t rather than the normal 1.96: the thin cells this chart warns about are
    exactly where the two differ, and the normal interval would understate
    their uncertainty. With one value there is no spread to estimate, so sd and
    interval are None rather than a misleading zero.
    """
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, None, None, None
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1)) * sd / math.sqrt(n)
    return mean, sd, mean - half, mean + half


def box_summary(
    group: str,
    values: pd.Series,
    flags: pd.Series,
    *,
    max_outliers: int = BOX_MAX_OUTLIERS,
) -> BoxSummary:
    """Quartiles and Tukey fences for one group (VIZ-3).

    The fences are the most extreme *observed* values within 1.5·IQR of the
    box, which is what Plotly draws as whiskers and what a reader expects — not
    the theoretical bound, which may lie where there is no data at all.

    Only the ``max_outliers`` values furthest from the median are shipped, with
    the full count beside them: a skewed hour can have hundreds past its
    whisker, and the extremes are the ones worth a mark.
    """
    q1 = float(values.quantile(0.25))
    median = float(values.quantile(0.5))
    q3 = float(values.quantile(0.75))
    spread = q3 - q1
    low, high = q1 - WHISKER_IQR * spread, q3 + WHISKER_IQR * spread

    inside = values[(values >= low) & (values <= high)]
    beyond = values[(values < low) | (values > high)]
    extremes = beyond.iloc[
        np.argsort(-(beyond - median).abs().to_numpy(), kind="stable")[:max_outliers]
    ]

    return BoxSummary(
        group=group,
        n=int(len(values)),
        mean=float(values.mean()),
        q1=q1,
        median=median,
        q3=q3,
        # A group can hold only outliers relative to its own quartiles when it
        # is tiny; fall back to the quartiles rather than an empty min().
        lower_fence=float(inside.min()) if len(inside) else q1,
        upper_fence=float(inside.max()) if len(inside) else q3,
        whisker_outliers=int(len(beyond)),
        outliers=tuple(sorted(float(value) for value in extremes)),
        anomalies_flagged=int(flags.sum()),
    )


def grouped(
    frame: pd.DataFrame,
    measure: str,
    group_by: str,
    *,
    split_by: str | None = None,
) -> GroupedResult:
    """VIZ-2 (bars) and VIZ-3 (boxes) for one measurement (task 12.4).

    One endpoint for both because they group the same rows by the same labels;
    two would load and label the frame twice for one picture. The boxes use
    ``group_by`` only — a box per (group, split) cell would put 96 boxes on a
    station × hour chart, which is a texture, not a comparison.
    """
    _require_measurement(measure, "measure")
    _require_grouping(group_by, "group_by")
    if split_by is not None:
        _require_grouping(split_by, "split_by")
        if split_by == group_by:
            raise SchemaValidationError(
                "split_by must differ from group_by", details={"group_by": group_by}
            )

    data = _finite(frame, (measure,))
    work = pd.DataFrame(
        {
            "value": data[measure].astype("float64"),
            "group": grouping_series(data, group_by),
            "flag": _flags(data),
        },
        index=data.index,
    )
    keys = ["group"]
    if split_by is not None:
        work["split"] = grouping_series(data, split_by)
        keys.append("split")
    work = work.dropna(subset=keys)

    if work.empty:
        raise InsufficientDataError(
            f"no rows with {measure} and a {group_by} label in scope.",
            details={"required": 1, "available": 0},
        )

    bars = []
    for key, cell in work.groupby(keys, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        mean, sd, low, high = mean_interval(cell["value"].to_numpy())
        bars.append(
            BarCell(
                group=str(key[0]),
                split=str(key[1]) if split_by is not None else None,
                n=len(cell),
                mean=mean,
                sd=sd,
                ci_low=low,
                ci_high=high,
                thin=len(cell) < THIN_GROUP,
            )
        )

    boxes = tuple(
        box_summary(str(name), cell["value"], cell["flag"])
        for name, cell in work.groupby("group", observed=True, sort=True)
    )

    return GroupedResult(
        measure=measure,
        group_by=group_by,
        split_by=split_by,
        categories=tuple(_present(work["group"])),
        split_categories=tuple(_present(work["split"])) if split_by is not None else None,
        bars=tuple(bars),
        boxes=boxes,
        rows_used=len(work),
        anomalies_in_rows=int(work["flag"].sum()),
        caveats=(profile.CORRELATION_CAVEAT, GROUPED_CAVEAT),
    )


# --- VIZ-4: pair plot (task 12.5) --------------------------------------------


@dataclass(frozen=True, slots=True)
class PairPlotResult:
    """A sample of complete rows, every measurement, and the full-data r."""

    columns: tuple[str, ...]
    color_by: str
    values: dict[str, tuple[float, ...]]
    bands: tuple[str, ...]
    groups: tuple[str | None, ...]
    categories: tuple[str, ...]
    is_anomaly: tuple[bool, ...]
    pearson: dict[str, dict[str, float | None]]
    rows_used: int
    points_returned: int
    sampled: bool
    anomalies_in_rows: int
    anomalies_in_points: int
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "color_by": self.color_by,
            "values": {name: _numbers(column) for name, column in self.values.items()},
            "bands": list(self.bands),
            "groups": list(self.groups),
            "categories": list(self.categories),
            "is_anomaly": list(self.is_anomaly),
            "pearson": self.pearson,
            "rows_used": self.rows_used,
            "points_returned": self.points_returned,
            "sampled": self.sampled,
            "anomalies_in_rows": self.anomalies_in_rows,
            "anomalies_in_points": self.anomalies_in_points,
            "caveats": list(self.caveats),
        }


def pair_plot(
    frame: pd.DataFrame,
    *,
    color_by: str = "pm25_band",
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    max_points: int = PAIRPLOT_MAX_POINTS,
) -> PairPlotResult:
    """VIZ-4: every measurement against every other (task 12.5).

    Rows complete in every column, because a splom draws one row as one mark in
    every panel; a row missing humidity would appear in six panels and vanish
    from four, and the panels would no longer show the same hours.

    The coefficients are the profile's, on every row in scope — not recomputed
    on the sample — so a panel's r matches its heatmap cell.
    """
    _require_grouping(color_by, "color_by")
    for column in columns:
        _require_measurement(column, "column")

    complete = _finite(frame, columns)
    if len(complete) < MIN_PAIRPLOT_ROWS:
        raise InsufficientDataError(
            f"a pair plot needs at least {MIN_PAIRPLOT_ROWS} rows complete in every "
            f"measurement; {len(complete)} available.",
            details={"required": MIN_PAIRPLOT_ROWS, "available": len(complete)},
        )

    points, was_sampled = manifold.subsample(complete, max_points)
    flags = _flags(points)
    labels = grouping_series(points, color_by)

    return PairPlotResult(
        columns=tuple(columns),
        color_by=color_by,
        values={
            column: tuple(points[column].to_numpy(dtype="float64")) for column in columns
        },
        bands=tuple(str(label) for label in bands.band_labels(points["pm25"])),
        groups=tuple(_labels(labels)),
        categories=tuple(_present(labels)),
        is_anomaly=tuple(bool(flag) for flag in flags),
        pearson=profile.bivariate_profile(frame, columns).pearson,
        rows_used=len(complete),
        points_returned=len(points),
        sampled=was_sampled,
        anomalies_in_rows=int(_flags(complete).sum()),
        anomalies_in_points=int(flags.sum()),
        caveats=(profile.CORRELATION_CAVEAT, PAIRPLOT_CAVEAT),
    )


# --- VIZ-5: Andrews curves (task 12.6) ---------------------------------------


def andrews_t(points: int = ANDREWS_T_POINTS) -> np.ndarray:
    """The evaluation grid: ``points`` values evenly over [−π, π]."""
    return np.linspace(-np.pi, np.pi, points)


def andrews_basis(t: np.ndarray, columns: int) -> np.ndarray:
    """Rows ``1/√2, sin t, cos t, sin 2t, cos 2t, …`` — one per column.

    A curve is then ``row @ basis``, which makes *f* visibly linear in the row
    — the property the exact mean curves rely on.
    """
    terms = [np.full_like(t, 1.0 / math.sqrt(2.0))]
    for index in range(1, columns):
        frequency = (index + 1) // 2
        terms.append(np.sin(frequency * t) if index % 2 else np.cos(frequency * t))
    return np.vstack(terms)


def andrews_curves(matrix: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``f_x(t) = x₁/√2 + x₂·sin t + x₃·cos t + x₄·sin 2t + x₅·cos 2t`` per row."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype="float64"))
    return matrix @ andrews_basis(t, matrix.shape[1])


@dataclass(frozen=True, slots=True)
class AndrewsClass:
    """One class: its exact mean curve and a within-class sample of curves."""

    label: str
    n: int
    mean_curve: tuple[float, ...]
    curves: tuple[tuple[float, ...], ...]
    is_anomaly: tuple[bool, ...]
    anomalies_in_class: int
    sampled: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "mean_curve": _numbers(self.mean_curve),
            "curves": [_numbers(curve) for curve in self.curves],
            "is_anomaly": list(self.is_anomaly),
            "anomalies_in_class": self.anomalies_in_class,
            "sampled": self.sampled,
        }


@dataclass(frozen=True, slots=True)
class AndrewsResult:
    columns: tuple[str, ...]
    class_by: str
    t: tuple[float, ...]
    classes: tuple[AndrewsClass, ...]
    means: dict[str, float]
    stds: dict[str, float]
    rows_used: int
    curves_returned: int
    sampled: bool
    anomalies_in_rows: int
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "class_by": self.class_by,
            "t": _numbers(self.t, 5),
            "classes": [item.as_dict() for item in self.classes],
            "standardisation": {
                column: {"mean": _number(self.means[column]), "std": _number(self.stds[column])}
                for column in self.columns
            },
            "rows_used": self.rows_used,
            "curves_returned": self.curves_returned,
            "sampled": self.sampled,
            "anomalies_in_rows": self.anomalies_in_rows,
            "caveats": list(self.caveats),
        }


def andrews(
    frame: pd.DataFrame,
    *,
    class_by: str = "time_of_day",
    columns: tuple[str, ...] = ANDREWS_COLUMNS,
    curves_per_class: int = ANDREWS_CURVES_PER_CLASS,
    points: int = ANDREWS_T_POINTS,
) -> AndrewsResult:
    """VIZ-5: Andrews curves per class (task 12.6).

    **Standardised first**, over every complete row in scope. Unscaled, PM10
    has the largest numbers and would set every curve's shape on its own.
    Population sd (ddof=0), as ``StandardScaler`` uses for PCA, so a z-score
    here and there are the same number. A constant column is left at zero
    rather than divided by zero.

    **Sampled within class.** ``curves_per_class`` evenly spaced rows of each
    class, so a rare class such as *Severe* keeps its sixty curves instead of
    being thinned in proportion to its size and vanishing under the common
    ones.

    **Mean curves are exact.** *f* is linear in the row, so the mean of a
    class's curves is the curve of its mean row: one ``mean()`` over every row
    in the class, not an average of the sixty drawn. That is the statement the
    chart is used to make, so it must not depend on the sample.

    **Not classed by PM2.5 band by default.** Classes defined by PM2.5 would
    separate on the PM2.5 term by construction, and the chart would appear to
    discover what was put into it.
    """
    _require_grouping(class_by, "class_by", ANDREWS_CLASSES)
    for column in columns:
        _require_measurement(column, "column")

    complete = _finite(frame, columns)
    classes = grouping_series(complete, class_by) if len(complete) else pd.Series(dtype=object)
    keep = classes.notna()
    complete, classes = complete[keep], classes[keep]

    if len(complete) < MIN_ANDREWS_ROWS:
        raise InsufficientDataError(
            f"Andrews curves need at least {MIN_ANDREWS_ROWS} rows complete in every "
            f"measurement; {len(complete)} available.",
            details={"required": MIN_ANDREWS_ROWS, "available": len(complete)},
        )

    values = complete[list(columns)].astype("float64")
    means = values.mean()
    stds = values.std(ddof=0)
    z = (values - means) / stds.where(stds > 0, 1.0)
    z["__flag"] = _flags(complete)

    t = andrews_t(points)
    basis = andrews_basis(t, len(columns))

    results = []
    for label, members in z.groupby(classes, observed=True, sort=True):
        matrix = members[list(columns)].to_numpy()
        drawn, was_sampled = manifold.subsample(members, curves_per_class)
        results.append(
            AndrewsClass(
                label=str(label),
                n=len(members),
                mean_curve=tuple(matrix.mean(axis=0) @ basis),
                curves=tuple(
                    tuple(curve) for curve in drawn[list(columns)].to_numpy() @ basis
                ),
                is_anomaly=tuple(bool(flag) for flag in drawn["__flag"]),
                anomalies_in_class=int(members["__flag"].sum()),
                sampled=was_sampled,
            )
        )

    return AndrewsResult(
        columns=tuple(columns),
        class_by=class_by,
        t=tuple(t),
        classes=tuple(results),
        means={column: float(means[column]) for column in columns},
        stds={column: float(stds[column]) for column in columns},
        rows_used=len(complete),
        curves_returned=sum(len(item.curves) for item in results),
        sampled=any(item.sampled for item in results),
        anomalies_in_rows=int(z["__flag"].sum()),
        caveats=(profile.CORRELATION_CAVEAT, ANDREWS_CAVEAT),
    )


__all__ = [
    "ANDREWS_CAVEAT",
    "ANDREWS_CLASSES",
    "ANDREWS_COLUMNS",
    "ANDREWS_CURVES_PER_CLASS",
    "ANDREWS_T_POINTS",
    "BOX_MAX_OUTLIERS",
    "CONFIDENCE",
    "DAY_NAMES",
    "GROUPED_CAVEAT",
    "GROUPINGS",
    "HOUR_LABELS",
    "MONTH_NAMES",
    "PAIRPLOT_CAVEAT",
    "PAIRPLOT_MAX_POINTS",
    "SCATTER_CAVEAT",
    "SCATTER_MAX_POINTS",
    "THIN_GROUP",
    "TIME_OF_DAY",
    "WEEKEND_LABELS",
    "WHISKER_IQR",
    "AndrewsClass",
    "AndrewsResult",
    "BarCell",
    "BoxSummary",
    "GroupedResult",
    "PairPlotResult",
    "ScatterResult",
    "andrews",
    "andrews_basis",
    "andrews_curves",
    "andrews_t",
    "box_summary",
    "grouped",
    "grouping_series",
    "mean_interval",
    "pair_plot",
    "scatter",
]
