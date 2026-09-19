"""Feature engineering entry points (tasks 5.5, 5.6).

Two paths lead into the transformer, and this module owns both so that neither
can grow its own version of "prepare the data first":

* **Training** (``build_features``) -- the whole slice, fitted and persisted to
  the feature store. What Phase 7 trains from.
* **Inference** (``features_at``) -- one station, one hour, a stored
  transformer, and only the history that hour actually needs. What Phase 9
  serves from (task 9.1).

They share ``prepare`` and they share the ``FeatureTransformer``. That is the
whole of the guarantee in design §8: the second path is not a reimplementation
of the first, it is the first with a smaller window.

**``prepare`` is window-local on purpose.** It forward-fills gaps of a few
hours and nothing else. Forward fill is a function of the rows immediately
before the gap, so it gives the same answer whether the frame starts a year
earlier or two days earlier -- which is exactly what makes the two paths agree.
Backward fill would read the future (AC-8) and MICE would read the whole slice
(see ``transformer``), so neither belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from core.exceptions import InsufficientDataError
from services import datasets
from services.datasets import STATION_COLUMN, DatasetWindow, station_key
from services.eda import cache
from services.eda.cache import DatasetVersion
from services.features import store
from services.features.spec import FeatureSpec
from services.features.transformer import (
    FeatureTransformer,
    complete_mask,
)
from services.quality import imputation

DEFAULT_MAX_GAP_HOURS = imputation.DEFAULT_MAX_GAP_HOURS


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    """One training-path build, with the provenance to reproduce it."""

    frame: pd.DataFrame
    transformer: FeatureTransformer
    window: DatasetWindow
    version: DatasetVersion
    rows: int
    complete_rows: int
    repaired: dict[str, int]
    max_gap_hours: int
    stored: store.StoredFeatures | None
    generated_at: datetime

    @property
    def complete_pct(self) -> float:
        return round(100.0 * self.complete_rows / self.rows, 2) if self.rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Everything except the frame itself -- this is the report, not the data."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "rows": self.rows,
            "complete_rows": self.complete_rows,
            "complete_pct": self.complete_pct,
            "max_gap_hours": self.max_gap_hours,
            "repaired": self.repaired,
            "window": self.window.as_dict(),
            "dataset_version": self.version.as_dict(),
            "spec": self.transformer.as_dict(),
            "outputs": self.stored.as_dict() if self.stored else None,
        }


def prepare(
    frame: pd.DataFrame, *, max_gap_hours: int = DEFAULT_MAX_GAP_HOURS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Window-local repair, shared by both paths.

    Forward fill only, per station, for runs of at most ``max_gap_hours``. A
    longer gap is left as NaN and propagates into the features that span it,
    which is the honest outcome: an eight-hour outage leaves no basis for
    claiming what ``pm25_lag_1h`` was.
    """
    return imputation.fill_short_gaps(
        frame, max_gap_hours=max_gap_hours, backward=False
    )


def build(
    frame: pd.DataFrame,
    transformer: FeatureTransformer,
    *,
    max_gap_hours: int = DEFAULT_MAX_GAP_HOURS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Repair, then transform. The single code path both callers go through."""
    repaired, filled = prepare(frame, max_gap_hours=max_gap_hours)
    return transformer.transform(repaired), filled


# --- Training path ----------------------------------------------------------


def build_features(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    transformer: FeatureTransformer | None = None,
    spec: FeatureSpec | None = None,
    max_gap_hours: int = DEFAULT_MAX_GAP_HOURS,
    persist: bool = True,
) -> FeatureBuildResult:
    """Build the feature set over a slice and write it to the store.

    ``transformer`` is optional. Passing one replays an existing spec -- which
    is what Phase 7 does when rebuilding features for a model already trained.
    Omitting it fits a new one from this slice, which is correct only when the
    slice *is* the training window (AC-8): fit on a window, then split it, never
    the other way round.
    """
    settings = settings or get_settings()

    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    window = datasets.describe_window(frame, tuple(source_ids or ()))
    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )

    repaired, filled = prepare(frame, max_gap_hours=max_gap_hours)
    transformer = transformer or FeatureTransformer.fit(
        repaired, base_spec=spec or FeatureSpec()
    )
    featured = transformer.transform(repaired)

    stored = None
    if persist and not featured.empty:
        stored = store.write(
            featured,
            transformer,
            window=window,
            settings=settings,
            dataset_version=version.as_dict(),
        )

    return FeatureBuildResult(
        frame=featured,
        transformer=transformer,
        window=window,
        version=version,
        rows=len(featured),
        complete_rows=int(complete_mask(featured, transformer.spec).sum()),
        repaired=filled,
        max_gap_hours=max_gap_hours,
        stored=stored,
        generated_at=datetime.now(timezone.utc),
    )


# --- Inference path (the other half of task 5.5; feeds task 9.1) ------------


def history_start(
    at: datetime, spec: FeatureSpec, *, max_gap_hours: int = DEFAULT_MAX_GAP_HOURS
) -> datetime:
    """The earliest observation a single row of features can depend on.

    ``history_hours`` covers the longest lag and the longest window;
    ``max_gap_hours`` covers the repair, which needs the readings *before* a gap
    to fill it. One extra hour is added so the boundary is inclusive under any
    rounding. Loading exactly this much is what keeps an inference request a
    bounded query rather than a table scan.
    """
    return at - timedelta(hours=spec.history_hours + max_gap_hours + 1)


def features_at(
    session: Session,
    transformer: FeatureTransformer,
    *,
    at: datetime,
    lat: float | None = None,
    lon: float | None = None,
    station: str | None = None,
    source_ids: tuple[int, ...] | None = None,
    max_gap_hours: int = DEFAULT_MAX_GAP_HOURS,
) -> pd.DataFrame:
    """Features for one station at one hour, built the training way.

    Loads only ``history_start(at) … at``, runs the same ``prepare`` and the
    same transformer, and returns the row for ``at``. Because every feature
    looks strictly backward, the row is identical to the one a full-dataset
    build produces for that hour -- the property Phase 5 exists to establish and
    that ``tests/test_features_transformer.py`` asserts.

    Returned as a one-row **frame**, not a Series. A Series of a mixed row
    collapses to ``object`` dtype, and an object array is what a model rejects
    at ``predict`` time; a frame keeps every column its own type, so
    ``feature_matrix`` hands Phase 9 a float matrix without a cast.
    """
    if station is None:
        if lat is None or lon is None:
            raise ValueError("features_at needs either a station key or lat and lon.")
        station = station_key(lat, lon)

    at = _as_utc(at)
    start = history_start(at, transformer.spec, max_gap_hours=max_gap_hours)

    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=at
    )
    frame = frame[frame[STATION_COLUMN] == station].reset_index(drop=True)

    datasets.require_rows(
        frame, minimum=1, what=f"feature construction for station {station}"
    )

    featured, _ = build(frame, transformer, max_gap_hours=max_gap_hours)
    hour = at.replace(minute=0, second=0, microsecond=0)
    match = featured[featured["timestamp"].dt.floor("h") == hour]

    if match.empty:
        raise InsufficientDataError(
            f"No observation for station {station} at {hour.isoformat()}.",
            details={"station": station, "at": hour.isoformat()},
        )

    # The last row, kept as a frame: at most one row per station-hour survives
    # DR-4, and taking the newest is the right tie-break if one ever does not.
    return match.iloc[[-1]]


def _as_utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


__all__ = [
    "DEFAULT_MAX_GAP_HOURS",
    "FeatureBuildResult",
    "build",
    "build_features",
    "features_at",
    "history_start",
    "prepare",
]
