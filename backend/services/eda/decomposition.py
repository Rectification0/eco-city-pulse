"""STL decomposition — BACSE301 Module 4 (task 4.5, design §8).

Separates a PM2.5 series into **trend**, **seasonal** and **residual**. That
split is what turns "PM2.5 was 180 last night" into something interpretable:
how much of it was the winter baseline, how much the ordinary evening peak, and
how much was genuinely unusual.

STL (Seasonal-Trend decomposition using Loess) is the right tool here rather
than a classical decomposition because it tolerates a seasonal shape that
changes over the year -- a city's diurnal pollution cycle is sharper in winter
than in monsoon -- and it is robust to the occasional spike.

Three things STL requires that raw observations do not provide, each handled
explicitly rather than assumed:

* **One series.** A decomposition of several stations stacked together is
  meaningless, so a station is chosen (or named) and the rest left alone.
* **A regular grid with no holes.** DR-4 puts every reading on the hourly grid,
  but hours can still be absent entirely. They are reindexed in and
  interpolated, and the count of interpolated points is reported -- a
  decomposition resting on 30% invented data should say so.
* **No NaNs.** Whatever the interpolation cannot reach at the ends is filled
  from the nearest observation, again counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import EcoCityPulseError, InsufficientDataError
from services.datasets import STATION_COLUMN

# Hourly data: the dominant cycle is the day.
DEFAULT_PERIOD = 24

# STL needs at least two full cycles, and asking for a trend from two days of
# data would produce a confident-looking line with nothing behind it.
MIN_CYCLES = 3

# Points returned to the caller by default. The full series can be 30k points
# per component, which is several megabytes of JSON for a chart that cannot
# render them anyway.
DEFAULT_MAX_POINTS = 720


class DecompositionUnavailableError(EcoCityPulseError):
    """statsmodels could not be loaded, so STL cannot run.

    A 503 rather than a 500: the rest of the EDA engine is unaffected, and the
    cause is an environment problem (a missing or blocked compiled extension)
    rather than a bug in the request.
    """

    status_code = 503
    code = "decomposition_unavailable"


def _load_stl() -> Any:
    """Import STL on first use.

    Lazy on purpose. statsmodels ships a compiled extension, and on a machine
    where the OS refuses to load it -- an application-control policy, a missing
    runtime -- a module-level import would take the whole application down at
    startup over one optional endpoint. This way the profile, the correlation
    matrices and the report all keep working, and only STL reports itself
    unavailable.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DecompositionUnavailableError(
            "STL decomposition is unavailable: statsmodels could not be loaded.",
            details={"error": str(exc)},
        ) from exc
    return STL


@dataclass(frozen=True, slots=True)
class DecompositionStrength:
    """Hyndman-Athanasopoulos strength measures, both in [0, 1].

    Each answers "how much of the variation does this component explain, over
    and above the noise". They make two series comparable in a way that raw
    component amplitudes do not.
    """

    trend: float
    seasonal: float

    def as_dict(self) -> dict[str, Any]:
        return {"trend": round(self.trend, 4), "seasonal": round(self.seasonal, 4)}


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """Trend / seasonal / residual for one station's series (task 4.5)."""

    column: str
    station: str
    period: int
    timestamps: tuple[datetime, ...]
    observed: tuple[float, ...]
    trend: tuple[float, ...]
    seasonal: tuple[float, ...]
    residual: tuple[float, ...]
    strength: DecompositionStrength
    points_returned: int
    points_analysed: int
    interpolated_points: int
    caveat: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "station": self.station,
            "period": self.period,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "observed": list(self.observed),
            "trend": list(self.trend),
            "seasonal": list(self.seasonal),
            "residual": list(self.residual),
            "strength": self.strength.as_dict(),
            "points_returned": self.points_returned,
            "points_analysed": self.points_analysed,
            "interpolated_points": self.interpolated_points,
            "caveat": self.caveat,
        }


def _strength(component: np.ndarray, residual: np.ndarray) -> float:
    """``max(0, 1 - Var(R) / Var(component + R))``."""
    combined = np.var(component + residual)
    if combined <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(residual) / combined))


def busiest_station(frame: pd.DataFrame, column: str) -> str:
    """The station with the most observed values for this column.

    A sensible default when the caller has not named one: the longest, most
    complete series gives the decomposition the best chance of being about the
    data rather than about the interpolation.
    """
    if frame.empty:
        raise InsufficientDataError("No observations to decompose.")

    counts = frame.groupby(STATION_COLUMN)[column].count().sort_values(ascending=False)
    if counts.empty or counts.iloc[0] == 0:
        raise InsufficientDataError(
            f"No observed values of {column!r} at any station.",
            details={"column": column},
        )
    return str(counts.index[0])


def to_regular_series(
    frame: pd.DataFrame, column: str, station: str
) -> tuple[pd.Series, int]:
    """One station's column on a gap-free hourly index.

    Returns the series and how many points had to be invented to make it
    regular. Reporting that count is the difference between a decomposition and
    a decoration.
    """
    subset = frame[frame[STATION_COLUMN] == station]
    if subset.empty:
        raise InsufficientDataError(
            f"Station {station!r} has no observations.", details={"station": station}
        )

    series = (
        subset.set_index("timestamp")[column].sort_index().groupby(level=0).mean()
    )

    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    regular = series.reindex(full_index)

    missing_before = int(regular.isna().sum())
    # Time-weighted interpolation for interior gaps, nearest-value fill for the
    # ends, where interpolation has nothing to work between.
    regular = regular.interpolate(method="time", limit_direction="both")
    regular = regular.ffill().bfill()

    return regular, missing_before


def decompose(
    frame: pd.DataFrame,
    *,
    column: str = "pm25",
    station: str | None = None,
    period: int = DEFAULT_PERIOD,
    robust: bool = True,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> DecompositionResult:
    """STL decomposition of one station's series (task 4.5, Module 4).

    ``robust=True`` by default: the demo data contains genuine pollution spikes
    and the quality engine deliberately keeps them (AC-5), so the loess fits
    need to be able to shrug them off rather than bend the trend around them.
    """
    if column not in frame.columns:
        raise InsufficientDataError(
            f"Unknown column {column!r}.", details={"column": column}
        )

    station = station or busiest_station(frame, column)
    series, interpolated = to_regular_series(frame, column, station)

    required = period * MIN_CYCLES
    if len(series) < required:
        raise InsufficientDataError(
            f"STL with period {period} needs at least {required} hourly points "
            f"({MIN_CYCLES} full cycles); {len(series)} available.",
            details={"required": required, "available": len(series), "period": period},
        )
    if series.nunique() <= 1:
        raise InsufficientDataError(
            f"{column!r} is constant at station {station!r}; nothing to decompose.",
            details={"column": column, "station": station},
        )

    stl = _load_stl()
    result = stl(series, period=period, robust=robust).fit()

    trend = np.asarray(result.trend, dtype="float64")
    seasonal = np.asarray(result.seasonal, dtype="float64")
    residual = np.asarray(result.resid, dtype="float64")

    strength = DecompositionStrength(
        trend=_strength(trend, residual), seasonal=_strength(seasonal, residual)
    )

    analysed = len(series)
    window = slice(-max_points, None) if max_points else slice(None)

    interpolation_pct = (interpolated / analysed * 100) if analysed else 0.0
    caveat = ""
    if interpolation_pct >= 10:
        caveat = (
            f"{interpolation_pct:.1f}% of the analysed points were interpolated to "
            "close gaps in the hourly grid. Treat the components as indicative."
        )

    return DecompositionResult(
        column=column,
        station=station,
        period=period,
        timestamps=tuple(series.index[window].to_pydatetime()),
        observed=tuple(np.round(series.to_numpy()[window], 4).tolist()),
        trend=tuple(np.round(trend[window], 4).tolist()),
        seasonal=tuple(np.round(seasonal[window], 4).tolist()),
        residual=tuple(np.round(residual[window], 4).tolist()),
        strength=strength,
        points_returned=len(series[window]),
        points_analysed=analysed,
        interpolated_points=interpolated,
        caveat=caveat,
    )


def is_available() -> bool:
    """Whether STL can run in this environment."""
    try:
        _load_stl()
    except DecompositionUnavailableError:
        return False
    return True


__all__ = [
    "DEFAULT_MAX_POINTS",
    "DEFAULT_PERIOD",
    "MIN_CYCLES",
    "DecompositionResult",
    "DecompositionUnavailableError",
    "DecompositionStrength",
    "busiest_station",
    "decompose",
    "is_available",
    "to_regular_series",
]
