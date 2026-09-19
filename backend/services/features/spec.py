"""The feature contract (tasks 5.1-5.4; specs §6.1, design §8).

This module holds *what* the feature set is, with no code that computes it. It
is a separate module because the exit criterion of Phase 5 is a reproducibility
claim -- "byte-identical between training and inference paths" -- and that claim
is only checkable if the definition is a single serialisable object rather than
a habit spread across two call sites.

**Everything here is deterministic.** A ``FeatureSpec`` is a frozen dataclass
that round-trips through JSON, so the spec used to train a model is stored
beside its artefact and replayed verbatim at inference time. Nothing is inferred
from the data at transform time; the one data-dependent decision in the phase --
whether a pollutant is skewed enough to deserve a log transform -- is made once
during ``fit`` and then frozen into ``log_columns``.

**Naming.** specs §6.1 writes the features as ``PM2.5_lag_1h``. The database
column is ``pm25`` (specs §9), and a feature name that does not match its source
column is a rename waiting to be got wrong, so the names are derived:
``pm25_lag_1h``, ``temp_lag_3h``, ``pm25_rolling_mean_24h``. Same features,
spelled after the schema they come from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from services.datasets import MEASUREMENT_COLUMNS

# Bumped if the serialised shape changes incompatibly. A stored spec carrying a
# different version is rejected rather than guessed at: silently reading an old
# spec into new code is exactly the drift this phase exists to prevent.
SPEC_VERSION = 1

# India Standard Time, in minutes. The diurnal and weekly rhythms this phase
# encodes are human-clock effects -- rush hour, the working week -- so they are
# derived in local time while the stored timestamp stays UTC (DR-2). The offset
# travels *inside* the spec rather than living in a constant read at call time,
# so a model trained against one city's clock cannot be served against another's.
DEFAULT_LOCAL_OFFSET_MINUTES = 330

# A rolling window must be at least this full before it reports a number. Half
# is a deliberate middle: requiring a complete window would erase the feature
# for any station with a single missing hour, while a 24-hour "mean" computed
# from two readings is not a daily mean in any useful sense.
MIN_WINDOW_COVERAGE = 0.5

# Candidates for the log transform of task 5.4. specs §6.1 names CO and SO2;
# neither is in the observation schema (specs §9), so the rule applies to the
# pollutants that are. Temperature is excluded for a reason beyond scope: it
# takes negative values, and the log of a negative reading is not a number.
POLLUTANT_COLUMNS: tuple[str, ...] = ("pm25", "pm10")

ROLLING_STATISTICS: tuple[str, ...] = ("mean", "std", "min", "max", "median")

# --- Temporal (task 5.1) ----------------------------------------------------

TEMPORAL_FEATURES: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "season",
)

# Seasons as the modelled region actually experiences them (IMD convention),
# not meteorological quarters. The distinction is not pedantry: post-monsoon --
# October and November -- is the stubble-burning, low-inversion window that
# dominates North Indian PM2.5, and a Sep-Nov "autumn" would split it across two
# labels and blend its start into the monsoon.
SEASONS: tuple[str, ...] = ("winter", "summer", "monsoon", "post_monsoon")

SEASON_BY_MONTH: dict[int, int] = {
    12: 0, 1: 0, 2: 0,        # winter
    3: 1, 4: 1, 5: 1,         # summer (pre-monsoon)
    6: 2, 7: 2, 8: 2, 9: 2,   # monsoon
    10: 3, 11: 3,             # post-monsoon
}

# ``season`` is a *nominal* code: 3 is not "more" than 0. A tree splits on it
# happily; a linear model would read the ordering as real, so Phase 7 one-hots
# anything listed here before fitting Ridge (task 7.3).
NOMINAL_FEATURES: tuple[str, ...] = ("season",)

# Cyclic in the sense that hour 23 is adjacent to hour 0. Declared so Phase 7
# can add sin/cos encodings for the linear models if the residuals ask for it;
# the raw integers are kept because they are what specs §6.1 names and what a
# tree model wants.
CYCLIC_FEATURES: tuple[str, ...] = ("hour_of_day", "day_of_week", "month")


def _require_measurement(column: str) -> None:
    if column not in MEASUREMENT_COLUMNS:
        raise ValueError(
            f"unknown column {column!r}; available: {list(MEASUREMENT_COLUMNS)}"
        )


@dataclass(frozen=True, slots=True)
class LagSpec:
    """One backward shift of one column (task 5.2).

    The shift is in *hours*, not rows. On a series with a missing hour those are
    different numbers, and the row-wise one is silently wrong -- which is why
    design §8 ties lag features to the hourly resample of DR-4.
    """

    column: str
    hours: int

    def __post_init__(self) -> None:
        _require_measurement(self.column)
        if self.hours < 1:
            raise ValueError(f"lag hours must be >= 1, got {self.hours}")

    @property
    def name(self) -> str:
        return f"{self.column}_lag_{self.hours}h"

    @property
    def lookback_hours(self) -> int:
        """How far back this feature reaches. Drives the inference window."""
        return self.hours

    def as_dict(self) -> dict[str, Any]:
        return {"column": self.column, "hours": self.hours}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LagSpec:
        return cls(column=str(payload["column"]), hours=int(payload["hours"]))


@dataclass(frozen=True, slots=True)
class RollingSpec:
    """One trailing window statistic (task 5.3).

    ``include_current`` is the leakage-relevant switch. When set, the window is
    ``[t - (hours - 1), t]``: it ends at the row it describes and never reaches
    past it. That is legitimate -- the value at *t* is known when a forecast for
    *t + h* is made -- but it is a choice, so it is recorded in the spec rather
    than assumed, and clearing it shifts the whole window one hour into the past
    for a strictly-prior statistic.
    """

    column: str
    hours: int
    statistic: str = "mean"
    include_current: bool = True

    def __post_init__(self) -> None:
        _require_measurement(self.column)
        if self.hours < 2:
            raise ValueError(f"rolling window must span >= 2 hours, got {self.hours}")
        if self.statistic not in ROLLING_STATISTICS:
            raise ValueError(
                f"unknown statistic {self.statistic!r}; "
                f"available: {list(ROLLING_STATISTICS)}"
            )

    @property
    def name(self) -> str:
        return f"{self.column}_rolling_{self.statistic}_{self.hours}h"

    @property
    def lookback_hours(self) -> int:
        return self.hours if self.include_current else self.hours + 1

    def min_periods(self, coverage: float = MIN_WINDOW_COVERAGE) -> int:
        """Observations required before the window reports a number.

        ``std`` needs two: the standard deviation of a single point is zero,
        which reads as "perfectly stable traffic" when it actually means "one
        reading".
        """
        floor = 2 if self.statistic == "std" else 1
        return max(floor, math.ceil(self.hours * coverage))

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "hours": self.hours,
            "statistic": self.statistic,
            "include_current": self.include_current,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RollingSpec:
        return cls(
            column=str(payload["column"]),
            hours=int(payload["hours"]),
            statistic=str(payload.get("statistic", "mean")),
            include_current=bool(payload.get("include_current", True)),
        )


# The features specs §6.1 names, exactly. Extending the set is a spec change,
# made by constructing a different FeatureSpec -- not by editing a transform.
DEFAULT_LAGS: tuple[LagSpec, ...] = (
    LagSpec("pm25", 1),
    LagSpec("pm25", 24),
    LagSpec("temp", 3),
)

DEFAULT_ROLLINGS: tuple[RollingSpec, ...] = (
    RollingSpec("pm25", 24, "mean"),
    RollingSpec("traffic_score", 6, "std"),
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The complete, serialisable definition of the feature set (task 5.5)."""

    lags: tuple[LagSpec, ...] = DEFAULT_LAGS
    rollings: tuple[RollingSpec, ...] = DEFAULT_ROLLINGS
    # Filled by FeatureTransformer.fit from the *training* slice only. Empty
    # means the fit found no pollutant skewed enough to be worth transforming.
    log_columns: tuple[str, ...] = ()
    temporal: bool = True
    local_offset_minutes: int = DEFAULT_LOCAL_OFFSET_MINUTES
    min_window_coverage: float = MIN_WINDOW_COVERAGE
    # Compute each rolling window from its own contents rather than from a
    # running accumulator. This is what makes the phase's reproducibility claim
    # literally true rather than true to fourteen decimal places -- see
    # ``windows.add_window_features``. Clearing it is roughly two orders of
    # magnitude faster and gives up exact agreement between a full-history
    # build and a serving window.
    exact_windows: bool = True

    def __post_init__(self) -> None:
        for column in self.log_columns:
            _require_measurement(column)
        if not 0 < self.min_window_coverage <= 1:
            raise ValueError(
                f"min_window_coverage must be in (0, 1], got {self.min_window_coverage}"
            )
        names = self.feature_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names in spec: {duplicates}")

    # --- The feature set ----------------------------------------------------

    @property
    def log_feature_names(self) -> tuple[str, ...]:
        return tuple(f"{column}_log" for column in self.log_columns)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Every engineered column, in a fixed order.

        The order is part of the contract, not a detail: a matrix handed to a
        model at inference time has to carry its columns in the order the model
        was fitted on, and deriving that order from the spec removes the chance
        of the two paths agreeing on the set but not the sequence.
        """
        names: list[str] = []
        if self.temporal:
            names.extend(TEMPORAL_FEATURES)
        names.extend(spec.name for spec in self.lags)
        names.extend(spec.name for spec in self.rollings)
        names.extend(self.log_feature_names)
        return tuple(names)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Measurement columns the spec reads, in schema order."""
        used = {spec.column for spec in (*self.lags, *self.rollings)}
        used.update(self.log_columns)
        return tuple(column for column in MEASUREMENT_COLUMNS if column in used)

    @property
    def history_hours(self) -> int:
        """Hours of prior data a single row of features needs.

        Phase 9 loads exactly this much history (plus a gap-fill margin) before
        asking for a prediction, so an inference request reads a bounded window
        rather than the whole table.
        """
        windows = [spec.lookback_hours for spec in (*self.lags, *self.rollings)]
        return max(windows) if windows else 0

    # --- Serialisation ------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "lags": [spec.as_dict() for spec in self.lags],
            "rollings": [spec.as_dict() for spec in self.rollings],
            "log_columns": list(self.log_columns),
            "temporal": self.temporal,
            "local_offset_minutes": self.local_offset_minutes,
            "min_window_coverage": self.min_window_coverage,
            "exact_windows": self.exact_windows,
            # Derived, and written out on purpose: a reader of the stored spec
            # should not have to re-implement the naming rule to learn what the
            # columns are called.
            "feature_names": list(self.feature_names),
            "history_hours": self.history_hours,
            "nominal_features": list(NOMINAL_FEATURES),
            "seasons": list(SEASONS),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpec:
        version = int(payload.get("spec_version", 0))
        if version != SPEC_VERSION:
            raise ValueError(
                f"feature spec version {version} cannot be read by this build "
                f"(expected {SPEC_VERSION}); rebuild the feature set."
            )
        return cls(
            lags=tuple(LagSpec.from_dict(item) for item in payload.get("lags", ())),
            rollings=tuple(
                RollingSpec.from_dict(item) for item in payload.get("rollings", ())
            ),
            log_columns=tuple(payload.get("log_columns", ())),
            temporal=bool(payload.get("temporal", True)),
            local_offset_minutes=int(
                payload.get("local_offset_minutes", DEFAULT_LOCAL_OFFSET_MINUTES)
            ),
            min_window_coverage=float(
                payload.get("min_window_coverage", MIN_WINDOW_COVERAGE)
            ),
            exact_windows=bool(payload.get("exact_windows", True)),
        )

    def with_log_columns(self, columns: tuple[str, ...]) -> FeatureSpec:
        """A copy carrying the log decision made by ``fit``."""
        return FeatureSpec(
            lags=self.lags,
            rollings=self.rollings,
            log_columns=tuple(columns),
            temporal=self.temporal,
            local_offset_minutes=self.local_offset_minutes,
            min_window_coverage=self.min_window_coverage,
            exact_windows=self.exact_windows,
        )


DEFAULT_SPEC = FeatureSpec()


__all__ = [
    "CYCLIC_FEATURES",
    "DEFAULT_LAGS",
    "DEFAULT_LOCAL_OFFSET_MINUTES",
    "DEFAULT_ROLLINGS",
    "DEFAULT_SPEC",
    "MIN_WINDOW_COVERAGE",
    "NOMINAL_FEATURES",
    "POLLUTANT_COLUMNS",
    "ROLLING_STATISTICS",
    "SEASONS",
    "SEASON_BY_MONTH",
    "SPEC_VERSION",
    "TEMPORAL_FEATURES",
    "FeatureSpec",
    "LagSpec",
    "RollingSpec",
]
