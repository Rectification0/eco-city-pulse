"""Forecast targets (task 7.1, specs §6.3).

specs §6.3 asks for PM2.5 **1, 6 and 24 hours into the future**, as three
separate regression problems rather than one multi-output model: the 1-hour
problem is nearly persistence and the 24-hour problem is nearly climatology, and
a single model forced to do both does neither well.

**The shift is forward, in hours, per station.** It is the mirror image of the
lag features of task 5.2 and it has the same trap: ``shift(-1)`` moves by one
*row*, so across a missing hour it would pair a feature row at 09:00 with a
target from 15:00 and call it a one-hour forecast. The same hourly grid is used
here, so a target that does not exist comes back NaN and its row is dropped
rather than quietly mislabelled.

**Why the direction matters more than the mechanics.** The target is the only
value in the pipeline that is allowed to come from the future -- that is what
makes it a forecast. Every other column is backward-looking (Phase 5), so the
single place a leak could enter through the data is here, in the sign of this
shift. A ``+h`` where ``-h`` belongs would train a model to predict the past
from the present, which scores beautifully and forecasts nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.datasets import STATION_COLUMN
from services.features.windows import GRID_FREQ, hourly_grid

# specs §6.3. Hours ahead, and the order the report presents them in.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 6, 24)

TARGET_COLUMN = "pm25"


def target_name(horizon: int, column: str = TARGET_COLUMN) -> str:
    """``pm25_h1`` -- also the ``models.target`` value the registry stores."""
    return f"{column}_h{horizon}"


@dataclass(frozen=True, slots=True)
class TargetSummary:
    """How much of the frame the shift left usable."""

    name: str
    horizon: int
    column: str
    rows: int
    present: int
    missing: int

    @property
    def coverage_pct(self) -> float:
        return round(100.0 * self.present / self.rows, 2) if self.rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizon_hours": self.horizon,
            "column": self.column,
            "rows": self.rows,
            "present": self.present,
            "missing": self.missing,
            "coverage_pct": self.coverage_pct,
        }


def add_target(
    frame: pd.DataFrame,
    horizon: int,
    *,
    column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, TargetSummary]:
    """Attach ``<column>_h<horizon>``: the value ``horizon`` hours after each row.

    The last ``horizon`` hours of every station's series have no target, by
    definition -- the future they describe has not happened. Those rows are left
    NaN here and dropped by the trainer, which is the honest handling: inventing
    a target would be inventing an observation.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 hour, got {horizon}")

    frame = frame.copy()
    name = target_name(horizon, column)
    frame[name] = np.nan

    if frame.empty:
        return frame, TargetSummary(name, horizon, column, 0, 0, 0)

    for _, group in frame.groupby(STATION_COLUMN, sort=False):
        grid = hourly_grid(group, (column,))
        # Negative shift: the value that arrives `horizon` hours later. On the
        # grid this is a true time shift; on rows it would not be.
        future = grid[column].shift(-horizon)
        hours = group["timestamp"].dt.floor(GRID_FREQ)
        frame.loc[group.index, name] = hours.map(future).to_numpy(dtype="float64")

    present = int(frame[name].notna().sum())
    return frame, TargetSummary(
        name=name,
        horizon=horizon,
        column=column,
        rows=len(frame),
        present=present,
        missing=len(frame) - present,
    )


def add_targets(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    *,
    column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, tuple[TargetSummary, ...]]:
    """Attach one target column per horizon."""
    summaries = []
    for horizon in horizons:
        frame, summary = add_target(frame, horizon, column=column)
        summaries.append(summary)
    return frame, tuple(summaries)


__all__ = [
    "DEFAULT_HORIZONS",
    "TARGET_COLUMN",
    "TargetSummary",
    "add_target",
    "add_targets",
    "target_name",
]
