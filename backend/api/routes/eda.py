"""EDA routes — FEAT-02 and FEAT-04 (tasks 4.3, 4.5, 4.7, 6.4, 6.5).

Per design §5 these handlers hold no statistics. They validate the request,
call ``services.eda``, and shape the response.

**On ``dataset_id``.** specs §8 describes the profile request as taking one.
This system has no discrete datasets to give an id to -- ``observations`` is a
continuous stream from several sources -- so a slice is named the way it
actually exists: by source, by time window, by column. What the spec wants an
id *for* (knowing which data a number describes, and being able to tell two
results apart) is served by the ``dataset_version`` fingerprint returned with
every response, which is derived from the data rather than assigned to it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field, model_validator

from api.dependencies import SessionDep, SettingsDep
from core.exceptions import InsufficientDataError
from services.datasets import MEASUREMENT_COLUMNS
from services.eda import cache, decomposition, manifold, service

router = APIRouter(prefix="/eda", tags=["eda"])


class DatasetSelector(BaseModel):
    """Which slice of ``observations`` to analyse (SEC-1).

    Every field optional: the default is "everything", which is what an analyst
    opening the EDA Studio for the first time wants.
    """

    source_ids: list[int] | None = Field(
        default=None, description="Restrict to these data sources. Omit for all."
    )
    start: datetime | None = Field(
        default=None, description="Inclusive lower bound on the observation timestamp."
    )
    end: datetime | None = Field(
        default=None, description="Inclusive upper bound on the observation timestamp."
    )
    columns: list[str] | None = Field(
        default=None,
        description=f"Numeric columns to profile. Defaults to {list(MEASUREMENT_COLUMNS)}.",
    )
    use_cache: bool = Field(
        default=True,
        description=(
            "Results are keyed by a fingerprint of the data, so a cached answer "
            "can never describe a slice that has since changed (design §11)."
        ),
    )

    @model_validator(mode="after")
    def _check_window(self) -> DatasetSelector:
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        return self

    @model_validator(mode="after")
    def _check_columns(self) -> DatasetSelector:
        if self.columns:
            unknown = sorted(set(self.columns) - set(MEASUREMENT_COLUMNS))
            if unknown:
                raise ValueError(
                    f"unknown columns {unknown}; available: {list(MEASUREMENT_COLUMNS)}"
                )
        return self

    def resolved_columns(self) -> tuple[str, ...]:
        return tuple(self.columns) if self.columns else MEASUREMENT_COLUMNS

    def source_tuple(self) -> tuple[int, ...] | None:
        return tuple(self.source_ids) if self.source_ids else None


# --- Response models (SEC-1) ------------------------------------------------


class UnivariateResponse(BaseModel):
    column: str
    count: int
    missing: int
    missing_pct: float
    mean: float | None
    median: float | None
    std: float | None
    variance: float | None
    min: float | None
    max: float | None
    q1: float | None
    q3: float | None
    iqr: float | None
    p05: float | None
    p95: float | None
    skewness: float | None
    kurtosis: float | None


class CorrelationPairResponse(BaseModel):
    a: str
    b: str
    pearson: float | None
    spearman: float | None
    sample_size: int
    divergence: float | None = Field(
        default=None,
        description=(
            "|Spearman - Pearson|. Large means the relationship is monotone but "
            "not linear, which a linear model will under-fit."
        ),
    )


class BivariateResponse(BaseModel):
    columns: list[str]
    pearson: dict[str, dict[str, float | None]]
    spearman: dict[str, dict[str, float | None]]
    pairs: list[CorrelationPairResponse]
    caveat: str = Field(description="ETH-1: association is not causation.")


class DistributionResponse(BaseModel):
    column: str
    skewness: float | None
    kurtosis: float | None
    shape: str
    is_strictly_positive: bool
    log_skewness: float | None
    recommend_log_transform: bool
    rationale: str
    normality_statistic: float | None
    normality_p_value: float | None


class ProfileResponse(BaseModel):
    """AC-3: mean, median, IQR and missingness for every numeric column."""

    generated_at: datetime
    cached: bool
    dataset_version: dict[str, Any]
    window: dict[str, Any]
    rows: int
    univariate: list[UnivariateResponse]
    bivariate: BivariateResponse
    distributions: list[DistributionResponse]
    caveats: list[str]


class DecompositionStrengthResponse(BaseModel):
    trend: float
    seasonal: float


class DecompositionResponse(BaseModel):
    column: str
    station: str
    period: int
    timestamps: list[datetime]
    observed: list[float]
    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    strength: DecompositionStrengthResponse
    points_returned: int
    points_analysed: int
    interpolated_points: int
    caveat: str
    cached: bool
    dataset_version: dict[str, Any]


class DecompositionRequest(DatasetSelector):
    column: str = Field(default="pm25", description="Series to decompose.")
    station: str | None = Field(
        default=None,
        description="Station key 'lat,lon'. Defaults to the one with most data.",
    )
    period: int = Field(
        default=decomposition.DEFAULT_PERIOD,
        ge=2,
        le=8766,
        description="Seasonal period in hours. 24 = daily, 168 = weekly.",
    )
    max_points: int | None = Field(
        default=decomposition.DEFAULT_MAX_POINTS,
        ge=24,
        le=20000,
        description="Trailing points returned. The full series can be megabytes.",
    )

    @model_validator(mode="after")
    def _check_column(self) -> DecompositionRequest:
        if self.column not in MEASUREMENT_COLUMNS:
            raise ValueError(
                f"unknown column {self.column!r}; available: {list(MEASUREMENT_COLUMNS)}"
            )
        return self


# --- Routes -----------------------------------------------------------------


@router.post(
    "/profile",
    response_model=ProfileResponse,
    summary="Univariate, bivariate and distribution statistics",
)
async def profile(
    session: SessionDep, settings: SettingsDep, selector: DatasetSelector | None = None
) -> ProfileResponse:
    """The automated statistical profile (FEAT-02, AC-3).

    A body is optional: with none, the whole dataset is profiled.
    """
    selector = selector or DatasetSelector()

    result = service.build_profile(
        session,
        settings,
        source_ids=selector.source_tuple(),
        start=selector.start,
        end=selector.end,
        columns=selector.resolved_columns(),
        use_cache=selector.use_cache,
    )
    return ProfileResponse.model_validate(result.as_dict())


@router.post(
    "/decompose",
    response_model=DecompositionResponse,
    summary="STL trend / seasonal / residual decomposition",
)
async def decompose(
    session: SessionDep,
    settings: SettingsDep,
    request: DecompositionRequest | None = None,
) -> DecompositionResponse:
    """STL for one station's series (task 4.5, Module 4).

    Returns 422 when the window is too short for the requested period, and 503
    when statsmodels cannot be loaded in this environment -- the difference
    matters, because only one of the two is fixable by changing the request.
    """
    request = request or DecompositionRequest()

    result, version, cached = service.decompose_series(
        session,
        settings,
        column=request.column,
        station=request.station,
        period=request.period,
        max_points=request.max_points,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        use_cache=request.use_cache,
    )

    return DecompositionResponse.model_validate(
        {**result.as_dict(), "cached": cached, "dataset_version": version.as_dict()}
    )


@router.get(
    "/report",
    response_class=Response,
    summary="Self-contained HTML EDA report",
    responses={200: {"content": {"text/html": {}}, "description": "The report."}},
)
async def report(
    session: SessionDep,
    settings: SettingsDep,
    include_decomposition: Annotated[
        bool, Query(description="Add the STL section. Adds a second or two.")
    ] = True,
    persist: Annotated[
        bool, Query(description="Also write it to data/processed (task 4.7).")
    ] = True,
) -> Response:
    """Render the report (Module 5).

    Returned as ``text/html`` rather than JSON because the artefact *is* the
    document: one file, no scripts, no external assets, printable to PDF.
    """
    result = service.generate_report(
        session,
        settings,
        include_decomposition=include_decomposition,
        persist=persist,
    )
    return Response(
        content=result.html,
        media_type="text/html; charset=utf-8",
        headers={"X-Report-Sections": ",".join(result.sections)},
    )


# --- Dimensionality reduction and ESI — FEAT-04 (tasks 6.4, 6.5, 6.6) -------


class ComponentResponse(BaseModel):
    index: int
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    loadings: dict[str, float] = Field(
        description=(
            "Coefficients over the standardized columns. Shipped with every "
            "response so the ESI stays interpretable rather than a black box "
            "(design §9)."
        )
    )
    drivers: list[str] = Field(
        description="Columns by absolute contribution, strongest first."
    )


class ESISummaryResponse(BaseModel):
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    latest: float | None


class ReductionResponse(BaseModel):
    """AC-6: an ESI on a 0-100 scale, with its loadings inspectable."""

    columns: list[str]
    rows_used: int
    rows_dropped: int
    components: list[ComponentResponse]
    esi: ESISummaryResponse
    pc1_oriented_by: str = Field(
        description=(
            "A principal component is defined only up to sign, so PC1 is "
            "oriented to increase with this column. 'High ESI' therefore means "
            "'dirtier air' by construction rather than by luck."
        )
    )
    pc1_sign_flipped: bool
    caveats: list[str]
    cached: bool
    dataset_version: dict[str, Any]
    artifact_path: str | None = Field(
        default=None, description="Where the fitted PCA was written (task 6.2)."
    )


class ReductionRequest(DatasetSelector):
    n_components: int | None = Field(
        default=None,
        ge=1,
        le=len(MEASUREMENT_COLUMNS),
        description="Components to retain. Defaults to all of them.",
    )
    persist: bool = Field(
        default=True, description="Also write the fitted PCA to data/processed."
    )


class ProjectionResponse(BaseModel):
    columns: list[str]
    x: list[float]
    y: list[float]
    timestamps: list[datetime]
    stations: list[str]
    hour_of_day: list[int]
    perplexity: float
    points: int
    rows_available: int
    subsampled: bool
    caveat: str
    cached: bool
    dataset_version: dict[str, Any]


class ProjectionRequest(DatasetSelector):
    perplexity: float = Field(
        default=manifold.DEFAULT_PERPLEXITY,
        ge=5.0,
        le=100.0,
        description="Neighbourhood size. Clamped below the sample size.",
    )
    max_points: int = Field(
        default=manifold.DEFAULT_MAX_POINTS,
        ge=manifold.MIN_ROWS,
        le=5000,
        description="Points embedded, sampled evenly across the window.",
    )


@router.post(
    "/reduce",
    response_model=ReductionResponse,
    summary="PCA components, explained variance, loadings and the ESI",
)
async def reduce(
    session: SessionDep, settings: SettingsDep, request: ReductionRequest | None = None
) -> ReductionResponse:
    """Dimensionality reduction and the Environmental Stress Index (FEAT-04, AC-6).

    Returns 422 when the window holds too few complete rows to estimate a
    covariance matrix from -- a request problem, and one a wider window fixes.
    """
    request = request or ReductionRequest()

    result, version, cached, path = service.reduce_dimensions(
        session,
        settings,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        columns=request.resolved_columns(),
        n_components=request.n_components,
        use_cache=request.use_cache,
        persist=request.persist,
    )

    return ReductionResponse.model_validate(
        {
            **result.as_dict(),
            "cached": cached,
            "dataset_version": version.as_dict(),
            "artifact_path": path,
        }
    )


@router.post(
    "/tsne",
    response_model=ProjectionResponse,
    summary="t-SNE 2D projection — EDA Studio only",
)
async def tsne(
    session: SessionDep, settings: SettingsDep, request: ProjectionRequest | None = None
) -> ProjectionResponse:
    """A 2D scatter for cluster inspection (task 6.5, specs §6.2).

    Deliberately has no counterpart in any inference path: t-SNE has no stable
    out-of-sample transform, so these coordinates are a picture and never a
    feature (design §9). The caveat travels in the payload.
    """
    request = request or ProjectionRequest()

    result, version, cached = service.project_tsne(
        session,
        settings,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        columns=request.resolved_columns(),
        perplexity=request.perplexity,
        max_points=request.max_points,
        use_cache=request.use_cache,
    )

    return ProjectionResponse.model_validate(
        {**result.as_dict(), "cached": cached, "dataset_version": version.as_dict()}
    )


class CacheStatsResponse(BaseModel):
    entries: int
    hits: int
    misses: int
    max_entries: int
    ttl_seconds: int


@router.get(
    "/cache",
    response_model=CacheStatsResponse,
    summary="Profile cache statistics",
)
async def cache_stats() -> CacheStatsResponse:
    """In-process, so these numbers are this worker's (design §11)."""
    return CacheStatsResponse.model_validate(cache.PROFILE_CACHE.stats())


__all__ = [
    "DatasetSelector",
    "DecompositionRequest",
    "InsufficientDataError",
    "ProfileResponse",
    "ProjectionRequest",
    "ProjectionResponse",
    "ReductionRequest",
    "ReductionResponse",
    "router",
]
