"""Temporal features (task 5.1, specs §6.1).

Five calendar features derived from one timestamp. They are cheap, they never
look forward, and they carry most of what a pollution model knows before it has
seen a single measurement: PM2.5 has a rush-hour shape, a weekday/weekend shape
and a seasonal shape, and a model given only lagged concentrations has to
rediscover all three from noise.

**Derived in local time.** The stored timestamp is UTC (DR-2) and stays that
way, but "hour of day" is a claim about human activity, not about the prime
meridian. In IST the evening traffic peak sits near 19:00 local, which is 13:30
UTC -- read in UTC, the peak lands mid-afternoon and smears across the calendar
day, splitting every weekday into two UTC dates. The offset lives in the
``FeatureSpec`` so that the value used in training is the value replayed at
inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.features.spec import (
    DEFAULT_LOCAL_OFFSET_MINUTES,
    SEASON_BY_MONTH,
    SEASONS,
    TEMPORAL_FEATURES,
)

WEEKEND_FIRST_DAY = 5  # Monday is 0, so Saturday and Sunday are 5 and 6.


def local_time(
    timestamps: pd.Series, offset_minutes: int = DEFAULT_LOCAL_OFFSET_MINUTES
) -> pd.Series:
    """Shift UTC timestamps onto the local wall clock.

    An offset rather than a named zone, deliberately. A zone would apply
    daylight-saving transitions, which is correct in the abstract and wrong
    here: the region modelled has none, and a DST jump inside a feature would
    put two different hours on the same lag distance without anything in the
    data recording that it happened. IST is a fixed +05:30 forever.
    """
    if timestamps.isna().any():
        raise ValueError(
            "temporal features need a timestamp on every row; "
            f"{int(timestamps.isna().sum())} rows have none."
        )
    # to_timedelta with an explicit unit: pd.Timedelta(minutes=...) is
    # deprecated under NumPy 2.5 and raises there in a future release.
    return timestamps + pd.to_timedelta(offset_minutes, unit="m")


def season_codes(months: pd.Series) -> pd.Series:
    """Map a month to its season code (see ``spec.SEASONS``)."""
    return months.map(SEASON_BY_MONTH).astype("int16")


def season_label(code: int) -> str:
    """The human name for a season code, for reports and the UI."""
    return SEASONS[int(code)]


def add_temporal(
    frame: pd.DataFrame,
    *,
    offset_minutes: int = DEFAULT_LOCAL_OFFSET_MINUTES,
) -> pd.DataFrame:
    """Add ``hour_of_day``, ``day_of_week``, ``is_weekend``, ``month``, ``season``.

    Every value is a pure function of the row's own timestamp: no neighbour, no
    window, nothing fitted. That makes this the one part of the feature set that
    is trivially identical between the training and inference paths, and it is
    worth saying so -- the effort in this phase is spent on the parts where it
    is not trivial.
    """
    frame = frame.copy()

    if frame.empty:
        for name in TEMPORAL_FEATURES:
            frame[name] = pd.Series(dtype="int16")
        return frame

    local = local_time(frame["timestamp"], offset_minutes)

    frame["hour_of_day"] = local.dt.hour.astype("int16")
    frame["day_of_week"] = local.dt.dayofweek.astype("int16")
    frame["is_weekend"] = (
        (local.dt.dayofweek >= WEEKEND_FIRST_DAY).astype("int16")
    )
    frame["month"] = local.dt.month.astype("int16")
    frame["season"] = season_codes(local.dt.month)

    return frame


def cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Sine/cosine encoding of a cyclic integer feature.

    Not part of the default feature set -- specs §6.1 names the raw integers,
    and a tree model splits on them perfectly well. Provided because Phase 7's
    Ridge model reads ``hour_of_day`` as a magnitude, for which 23 and 0 are as
    far apart as possible rather than adjacent.
    """
    angle = 2.0 * np.pi * values.astype("float64") / period
    return np.sin(angle), np.cos(angle)


__all__ = [
    "WEEKEND_FIRST_DAY",
    "add_temporal",
    "cyclical",
    "local_time",
    "season_codes",
    "season_label",
]
