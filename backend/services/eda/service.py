"""EDA service entry points (tasks 4.3, 4.6, 6.4, 6.5).

What the routes call. Each function does the same three things: fingerprint the
slice, consult the cache, compute only on a miss. The fingerprint is taken
*before* the data is loaded, so a cache hit costs one aggregate query rather
than pulling 30,000 rows to discover the answer is already known.

PCA and t-SNE are cached for a second reason as well: both are expensive enough
that an EDA Studio which recomputed them on every panel switch would feel
broken, and both are deterministic, so a cached answer is the same answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from services import datasets
from services.datasets import MEASUREMENT_COLUMNS, DatasetWindow
from services.eda import cache, decomposition, manifold, profile, reduction, report
from services.eda.cache import PROFILE_CACHE, DatasetVersion
from services.quality import imputation

# Where the fitted PCA is written (task 6.2). Explained variance and loadings
# are the parts worth keeping: they are what makes a stored ESI readable months
# later, and they are small enough to live beside the other processed artefacts.
PCA_MODEL_FILENAME = "pca_model.json"


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
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )

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
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )

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
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )

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


# --- Phase 6: dimensionality reduction and ESI ------------------------------


def _analysis_frame(
    session: Session,
    *,
    source_ids: tuple[int, ...] | None,
    start: datetime | None,
    end: datetime | None,
    max_gap_hours: int = imputation.DEFAULT_MAX_GAP_HOURS,
):  # noqa: ANN202 - a DataFrame; annotating it would add pandas to the imports
    """Load a slice and repair only its short gaps.

    Forward fill only. A backward fill would read future values, which is
    harmless for a purely historical chart but not for the ESI: design §9 has
    it scoring live conditions on the dashboard, and a number that quietly used
    tomorrow's reading would be indefensible there (AC-8). Longer gaps stay
    missing and their rows are excluded from the projection, counted rather
    than hidden.
    """
    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    repaired, _ = imputation.fill_short_gaps(
        frame, max_gap_hours=max_gap_hours, backward=False
    )
    return repaired


def persist_pca_model(
    result: reduction.ReductionResult, *, settings: Settings
) -> str:
    """Write the fitted PCA to ``data/processed`` (task 6.2)."""
    directory = settings.data_processed_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PCA_MODEL_FILENAME

    path.write_text(
        json.dumps(
            {
                "model": result.model.as_dict(),
                "components": [item.as_dict() for item in result.model.loadings],
                "esi": result.esi_summary,
                "rows_used": result.rows_used,
                "rows_dropped": result.rows_dropped,
                "caveats": list(result.caveats),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return str(path)


def reduce_dimensions(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    n_components: int | None = None,
    use_cache: bool = True,
    persist: bool = True,
) -> tuple[reduction.ReductionResult, DatasetVersion, bool, str | None]:
    """PCA components, explained variance, loadings and the ESI (tasks 6.2-6.4)."""
    settings = settings or get_settings()
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )

    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )
    key = PROFILE_CACHE.key(
        version,
        kind="pca",
        source_ids=list(source_ids or ()),
        start=start,
        end=end,
        columns=list(columns),
        n_components=n_components,
    )

    if use_cache:
        hit = PROFILE_CACHE.get(key)
        if hit is not None:
            return hit, version, True, None

    frame = _analysis_frame(session, source_ids=source_ids, start=start, end=end)
    result = reduction.reduce_frame(
        frame, columns=columns, n_components=n_components
    )

    if use_cache:
        PROFILE_CACHE.set(key, result)

    path = persist_pca_model(result, settings=settings) if persist else None
    return result, version, False, path


def project_tsne(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    perplexity: float = manifold.DEFAULT_PERPLEXITY,
    max_points: int = manifold.DEFAULT_MAX_POINTS,
    use_cache: bool = True,
) -> tuple[manifold.ProjectionResult, DatasetVersion, bool]:
    """A 2D t-SNE scatter for the EDA Studio (task 6.5).

    Nothing here is persisted and nothing returns it to a model: the embedding
    has no out-of-sample transform, so it is a picture and only a picture
    (design §9).
    """
    settings = settings or get_settings()
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )

    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )
    key = PROFILE_CACHE.key(
        version,
        kind="tsne",
        source_ids=list(source_ids or ()),
        start=start,
        end=end,
        columns=list(columns),
        perplexity=perplexity,
        max_points=max_points,
    )

    if use_cache:
        hit = PROFILE_CACHE.get(key)
        if hit is not None:
            return hit, version, True

    frame = _analysis_frame(session, source_ids=source_ids, start=start, end=end)
    result = manifold.project(
        frame, columns=columns, perplexity=perplexity, max_points=max_points
    )

    if use_cache:
        PROFILE_CACHE.set(key, result)

    return result, version, False


def _utc():  # noqa: ANN202 - tiny helper, kept out of the import list
    from datetime import timezone

    return timezone.utc


__all__ = [
    "PCA_MODEL_FILENAME",
    "ProfileResult",
    "build_profile",
    "decompose_series",
    "generate_report",
    "persist_pca_model",
    "project_tsne",
    "reduce_dimensions",
]
