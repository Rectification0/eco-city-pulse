"""EDA service entry points (task 4.3, 4.6).

What the routes call. Each function does the same three things: fingerprint the
slice, consult the cache, compute only on a miss. The fingerprint is taken
*before* the data is loaded, so a cache hit costs one aggregate query rather
than pulling 30,000 rows to discover the answer is already known.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from services import datasets
from services.datasets import MEASUREMENT_COLUMNS, DatasetWindow
from services.eda import cache, decomposition, profile, report
from services.eda.cache import PROFILE_CACHE, DatasetVersion


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """A profile plus the provenance needed to trust it."""

    profile: profile.StatisticalProfile
    window: DatasetWindow
    version: DatasetVersion
    cached: bool
    generated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "cached": self.cached,
            "dataset_version": self.version.as_dict(),
            "window": self.window.as_dict(),
            **self.profile.as_dict(),
        }


def build_profile(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    use_cache: bool = True,
) -> ProfileResult:
    """The full statistical profile for a slice (FEAT-02, AC-3)."""
    settings = settings or get_settings()

    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )
    key = PROFILE_CACHE.key(
        version,
        kind="profile",
        source_ids=list(source_ids or ()),
        start=start,
        end=end,
        columns=list(columns),
    )

    if use_cache:
        hit = PROFILE_CACHE.get(key)
        if hit is not None:
            stored_profile, window, generated_at = hit
            return ProfileResult(
                profile=stored_profile,
                window=window,
                version=version,
                cached=True,
                generated_at=generated_at,
            )

    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    computed = profile.build(frame, columns)
    window = datasets.describe_window(frame, tuple(source_ids or ()))
    generated_at = datetime.now(tz=_utc())

    if use_cache:
        PROFILE_CACHE.set(key, (computed, window, generated_at))

    return ProfileResult(
        profile=computed,
        window=window,
        version=version,
        cached=False,
        generated_at=generated_at,
    )


def decompose_series(
    session: Session,
    settings: Settings | None = None,
    *,
    column: str = "pm25",
    station: str | None = None,
    period: int = decomposition.DEFAULT_PERIOD,
    max_points: int | None = decomposition.DEFAULT_MAX_POINTS,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    use_cache: bool = True,
) -> tuple[decomposition.DecompositionResult, DatasetVersion, bool]:
    """STL for one station's series (task 4.5)."""
    settings = settings or get_settings()

    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )
    key = PROFILE_CACHE.key(
        version,
        kind="stl",
        column=column,
        station=station,
        period=period,
        max_points=max_points,
        source_ids=list(source_ids or ()),
        start=start,
        end=end,
    )

    if use_cache:
        hit = PROFILE_CACHE.get(key)
        if hit is not None:
            return hit, version, True

    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    result = decomposition.decompose(
        frame, column=column, station=station, period=period, max_points=max_points
    )

    if use_cache:
        PROFILE_CACHE.set(key, result)

    return result, version, False


def generate_report(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    include_decomposition: bool = True,
    persist: bool = True,
) -> report.ReportResult:
    """Render the standalone HTML EDA report (task 4.7, Module 5)."""
    settings = settings or get_settings()

    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    window = datasets.describe_window(frame, tuple(source_ids or ()))
    computed = profile.build(frame)
    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )

    stl: decomposition.DecompositionResult | None = None
    stl_note = ""
    if include_decomposition and not frame.empty:
        try:
            stl = decomposition.decompose(frame, column="pm25", max_points=336)
        except Exception as exc:  # noqa: BLE001
            # A report that is missing one section is far more useful than no
            # report. Too short a series, a constant column, or statsmodels
            # unavailable are all reasons to note and carry on.
            stl_note = str(exc)

    return report.render(
        frame=frame,
        statistics=computed,
        window=window,
        version=version,
        decomposition_result=stl,
        decomposition_note=stl_note,
        settings=settings,
        persist=persist,
    )


def _utc():  # noqa: ANN202 - tiny helper, kept out of the import list
    from datetime import timezone

    return timezone.utc


__all__ = ["ProfileResult", "build_profile", "decompose_series", "generate_report"]
