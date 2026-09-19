"""Lag and rolling features (tasks 5.2, 5.3; specs §6.1, design §8).

Both families answer the same question -- what was this series doing before
now? -- and both are computed the same way, so they share a module and a single
pass over the data.

**The hourly grid is the whole trick.** design §8: "without a uniform grid,
``lag_1h`` is not a well-defined shift". Ingestion resamples every source to an
hourly grid (DR-4), but a grid can still have *holes*: an hour with no reading
produces no row. ``shift(1)`` moves by one **row**, so across a six-hour outage
it would quietly label a reading from six hours ago as ``pm25_lag_1h`` -- a
wrong number with a right-looking name, which no downstream test would catch.

So each station's series is first reindexed onto a complete hourly
``date_range``. Absent hours become NaN, the shift becomes a true time shift,
and the features are mapped back onto the rows that actually exist. A lag over a
gap comes back NaN, which is the honest answer: the value is unknown, not equal
to whatever the sensor last reported.

**Per station, always.** A shift that crossed stations would hand one sensor's
history to another -- the same error Phase 3 guards against when filling gaps.

**Nothing reads forward.** Lags shift strictly backward and rolling windows are
trailing, so a value at *t* is a function of ``[t - k, t]`` only. AC-8 is a
property of these two functions, and the tests assert it directly: perturbing a
future observation must leave every earlier feature untouched.

**Why the rolling statistics are computed the slow way.** pandas' rolling
aggregates run an *incremental* accumulator: it adds the entering value and
subtracts the leaving one as the window advances. That is fast and numerically
reasonable, but the running error depends on how many rows were processed
before the window arrived -- so the same hour computed over a year of history
and over a two-day serving window can differ in the last bits. Phase 5 promises
those two are identical, so by default each window is reduced from its own
contents (``rolling.apply``), which depends on nothing outside it. The cost is
real -- roughly two orders of magnitude on this step -- and ``FeatureSpec.
exact_windows`` turns it off for anyone who would rather have the speed and
accept agreement to ~1e-14.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.datasets import STATION_COLUMN
from services.features.spec import (
    MIN_WINDOW_COVERAGE,
    LagSpec,
    RollingSpec,
)

GRID_FREQ = "h"

# Window reducers that read only the window they are given. ``min_periods``
# guarantees each one sees at least one value (two for std), so none of them
# meets an all-NaN slice.
EXACT_REDUCERS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    # ddof=1 to match pandas' rolling std, which is the sample standard
    # deviation; numpy's default ddof=0 is the population one.
    "std": lambda window: np.nanstd(window, ddof=1),
}

# A defensive ceiling on the reindexed grid. One stray timestamp -- a sensor
# reporting the Unix epoch, a bad upload that slipped a year -- would otherwise
# expand a 120-day window into a multi-decade array. 20 years of hours is far
# past any real slice and still small enough to allocate.
MAX_GRID_HOURS = 24 * 365 * 20


def hourly_grid(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """One station's measurements on a gapless hourly index.

    Timestamps are floored to the hour before indexing. They should already sit
    on the hour -- the harmonizer resamples them there (DR-4) -- but flooring
    costs nothing and means a stray off-grid row lands in its hour rather than
    silently missing every lookup afterwards.
    """
    block = frame[list(columns)].astype("float64")
    block.index = frame["timestamp"].dt.floor(GRID_FREQ)

    if block.index.has_duplicates:
        # Two readings for one station-hour should not exist after DR-4. If one
        # does, averaging matches what the harmonizer would have done rather
        # than letting an arbitrary row win.
        block = block.groupby(level=0).mean()

    span = int((block.index.max() - block.index.min()).total_seconds() // 3600) + 1
    if span > MAX_GRID_HOURS:
        raise ValueError(
            f"station series spans {span} hours, beyond the {MAX_GRID_HOURS}-hour "
            "grid ceiling; narrow the window or check for a stray timestamp."
        )

    index = pd.date_range(
        block.index.min(), block.index.max(), freq=GRID_FREQ, tz=block.index.tz
    )
    return block.reindex(index)


def _assign(
    frame: pd.DataFrame, rows: pd.Index, hours: pd.Series, values: pd.Series, name: str
) -> None:
    """Map a grid-indexed series back onto the station's real rows.

    ``Series.map`` rather than ``reindex``: it looks up by value, so duplicate
    station-hours (which reindex refuses outright) simply resolve to the same
    grid entry.
    """
    frame.loc[rows, name] = hours.map(values).to_numpy(dtype="float64")


def add_window_features(
    frame: pd.DataFrame,
    *,
    lags: tuple[LagSpec, ...] = (),
    rollings: tuple[RollingSpec, ...] = (),
    min_window_coverage: float = MIN_WINDOW_COVERAGE,
    exact_windows: bool = True,
) -> pd.DataFrame:
    """Add every lag and rolling feature, one pass per station.

    Returns a copy with one float64 column per spec, aligned to the input rows.
    A feature the window cannot support -- the first hours of a series, or a
    window with too few observations -- is NaN rather than a filled-in guess;
    Phase 7 decides whether to drop those rows or impute inside its training
    window, and that decision is not this function's to make.
    """
    frame = frame.copy()
    specs: tuple[LagSpec | RollingSpec, ...] = (*lags, *rollings)

    if not specs:
        return frame

    for spec in specs:
        frame[spec.name] = np.nan

    if frame.empty:
        return frame

    columns = tuple({spec.column for spec in specs})

    for _, group in frame.groupby(STATION_COLUMN, sort=False):
        grid = hourly_grid(group, columns)
        hours = group["timestamp"].dt.floor(GRID_FREQ)

        for lag in lags:
            _assign(frame, group.index, hours, grid[lag.column].shift(lag.hours), lag.name)

        for rolling in rollings:
            window = grid[rolling.column].rolling(
                window=rolling.hours,
                min_periods=rolling.min_periods(min_window_coverage),
            )
            values = (
                window.apply(EXACT_REDUCERS[rolling.statistic], raw=True)
                if exact_windows
                else getattr(window, rolling.statistic)()
            )
            if not rolling.include_current:
                # Shift on the grid, not on the rows: one *hour* earlier, which
                # over a gap is not the same as one row earlier.
                values = values.shift(1)
            _assign(frame, group.index, hours, values, rolling.name)

    return frame


__all__ = ["GRID_FREQ", "MAX_GRID_HOURS", "add_window_features", "hourly_grid"]
