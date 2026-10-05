"""EDA routes — FEAT-02, FEAT-04 and VIZ-1 … VIZ-6 (tasks 4.3–6.5, 12.8).

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
from services.eda import cache, decomposition, manifold, service, visual

router = APIRouter(prefix="/eda", tags=["eda"])


class ScopeSelector(BaseModel):
    """Which rows of ``observations`` to read (SEC-1).

    Every field optional: the default is "everything the configured scope
    allows", which is what an analyst opening the EDA Studio wants. Shared by
    every EDA request, so a chart cannot quietly accept a narrower notion of
    scope than the profile beside it.
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
    use_cache: bool = Field(
        default=True,
        description=(
            "Results are keyed by a fingerprint of the data, so a cached answer "
            "can never describe a slice that has since changed (design §11)."
        ),
    )

    @model_validator(mode="after")
    def _check_window(self) -> ScopeSelector:
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        return self

    def source_tuple(self) -> tuple[int, ...] | None:
        return tuple(self.source_ids) if self.source_ids else None


class DatasetSelector(ScopeSelector):
    """A scope plus the numeric columns to analyse."""

    columns: list[str] | None = Field(
        default=None,
        description=f"Numeric columns to profile. Defaults to {list(MEASUREMENT_COLUMNS)}.",
    )

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


# --- Visual EDA — VIZ-1 … VIZ-6 (tasks 12.3–12.8) --------------------------
#
# Validation lives here, on the request models, so an unknown column or
# grouping is a 422 naming what *is* available before any query runs. The
# service functions check the same names again: they are callable from scripts
# and the report too, and a misspelt column there should fail as loudly.


def _measurement(value: str, role: str) -> str:
    if value not in MEASUREMENT_COLUMNS:
        raise ValueError(
            f"unknown {role} {value!r}; available: {list(MEASUREMENT_COLUMNS)}"
        )
    return value


def _grouping(
    value: str | None, role: str, allowed: tuple[str, ...] = visual.GROUPINGS
) -> str | None:
    if value is not None and value not in allowed:
        raise ValueError(f"unknown {role} {value!r}; available: {list(allowed)}")
    return value


class VisualProvenance(BaseModel):
    """What every visual chart response carries besides the chart itself."""

    caveats: list[str] = Field(
        description="ETH-1 first (association is not causation), then the chart's own."
    )
    cached: bool
    dataset_version: dict[str, Any]
    window: dict[str, Any] = Field(
        description="The slice actually read, after the source scope was resolved."
    )


class ScatterRequest(ScopeSelector):
    x: str = Field(default="traffic_score", description="Measurement on the x axis.")
    y: str = Field(default="pm25", description="Measurement on the y axis.")
    color_by: str | None = Field(
        default=None, description=f"Optional grouping: one of {list(visual.GROUPINGS)}."
    )

    @model_validator(mode="after")
    def _check(self) -> ScatterRequest:
        _measurement(self.x, "x")
        _measurement(self.y, "y")
        if self.x == self.y:
            raise ValueError("x and y must be different columns")
        _grouping(self.color_by, "color_by")
        return self


class ScatterResponse(VisualProvenance):
    """VIZ-1. Points are a sample; the fit and coefficients use every row."""

    x_column: str
    y_column: str
    color_by: str | None
    x: list[float | None]
    y: list[float | None]
    is_anomaly: list[bool]
    groups: list[str | None] | None
    categories: list[str] | None
    slope: float | None
    intercept: float | None
    pearson: float | None = Field(description="Equal to the /eda/profile cell for this pair.")
    spearman: float | None
    r_squared: float | None
    line_x: list[float | None] = Field(description="OLS line end points, x.")
    line_y: list[float | None] = Field(description="OLS line end points, y.")
    n: int
    rows_used: int
    points_returned: int
    sampled: bool
    anomalies_in_rows: int
    anomalies_in_points: int


class GroupedRequest(ScopeSelector):
    measure: str = Field(default="pm25", description="Measurement to summarise.")
    group_by: str = Field(
        default="hour_of_day", description=f"One of {list(visual.GROUPINGS)}."
    )
    split_by: str | None = Field(
        default=None,
        description="Optional second grouping for the bar chart; must differ from group_by.",
    )

    @model_validator(mode="after")
    def _check(self) -> GroupedRequest:
        _measurement(self.measure, "measure")
        _grouping(self.group_by, "group_by")
        _grouping(self.split_by, "split_by")
        if self.split_by is not None and self.split_by == self.group_by:
            raise ValueError("split_by must differ from group_by")
        return self


class BarCellResponse(BaseModel):
    group: str
    split: str | None
    n: int
    mean: float | None
    sd: float | None
    ci_low: float | None
    ci_high: float | None
    thin: bool = Field(description="n below the thin threshold: compare with care.")


class BoxSummaryResponse(BaseModel):
    group: str
    n: int
    mean: float | None
    q1: float | None
    median: float | None
    q3: float | None
    lower_fence: float | None
    upper_fence: float | None
    whisker_outliers: int = Field(description="Values past the whiskers, all of them.")
    outliers: list[float | None] = Field(description="The most extreme of them, capped.")
    anomalies_flagged: int = Field(
        description="Rows the quality engine flagged — a different test from the whiskers."
    )


class GroupedResponse(VisualProvenance):
    """VIZ-2 (bars, by group × split) and VIZ-3 (boxes, by group)."""

    measure: str
    group_by: str
    split_by: str | None
    categories: list[str]
    split_categories: list[str] | None
    bars: list[BarCellResponse]
    boxes: list[BoxSummaryResponse]
    rows_used: int
    anomalies_in_rows: int
    confidence: float
    thin_threshold: int


class PairPlotRequest(ScopeSelector):
    color_by: str = Field(
        default="pm25_band", description=f"One of {list(visual.GROUPINGS)}."
    )

    @model_validator(mode="after")
    def _check(self) -> PairPlotRequest:
        _grouping(self.color_by, "color_by")
        return self


class PairPlotResponse(VisualProvenance):
    """VIZ-4. A sample of complete rows; r over every row in scope."""

    columns: list[str]
    color_by: str
    values: dict[str, list[float | None]]
    bands: list[str]
    groups: list[str | None]
    categories: list[str]
    is_anomaly: list[bool]
    pearson: dict[str, dict[str, float | None]]
    rows_used: int
    points_returned: int
    sampled: bool
    anomalies_in_rows: int
    anomalies_in_points: int


class AndrewsRequest(ScopeSelector):
    class_by: str = Field(
        default="time_of_day", description=f"One of {list(visual.ANDREWS_CLASSES)}."
    )

    @model_validator(mode="after")
    def _check(self) -> AndrewsRequest:
        _grouping(self.class_by, "class_by", visual.ANDREWS_CLASSES)
        return self


class AndrewsClassResponse(BaseModel):
    label: str
    n: int
    mean_curve: list[float | None] = Field(
        description="Exact: the curve of the class's mean row, over every row."
    )
    curves: list[list[float | None]]
    is_anomaly: list[bool]
    anomalies_in_class: int
    sampled: bool


class StandardisationResponse(BaseModel):
    mean: float | None
    std: float | None


class AndrewsResponse(VisualProvenance):
    """VIZ-5. Column order is fixed and returned, because it shapes the curves."""

    columns: list[str]
    class_by: str
    t: list[float | None]
    classes: list[AndrewsClassResponse]
    standardisation: dict[str, StandardisationResponse]
    rows_used: int
    curves_returned: int
    sampled: bool
    anomalies_in_rows: int


@router.post(
    "/scatter",
    response_model=ScatterResponse,
    summary="Scatter plot of two measurements with OLS line and r (VIZ-1)",
)
async def scatter(
    session: SessionDep, settings: SettingsDep, request: ScatterRequest | None = None
) -> ScatterResponse:
    """VIZ-1 (task 12.3). 422 when fewer than three rows hold both columns."""
    request = request or ScatterRequest()
    result = service.scatter_plot(
        session,
        settings,
        x=request.x,
        y=request.y,
        color_by=request.color_by,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        use_cache=request.use_cache,
    )
    return ScatterResponse.model_validate(result.as_dict())


@router.post(
    "/grouped",
    response_model=GroupedResponse,
    summary="Grouped means with 95% CIs and boxplot summaries (VIZ-2, VIZ-3)",
)
async def grouped(
    session: SessionDep, settings: SettingsDep, request: GroupedRequest | None = None
) -> GroupedResponse:
    """VIZ-2 and VIZ-3 (task 12.4), from one ``groupby``."""
    request = request or GroupedRequest()
    result = service.grouped_summary(
        session,
        settings,
        measure=request.measure,
        group_by=request.group_by,
        split_by=request.split_by,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        use_cache=request.use_cache,
    )
    return GroupedResponse.model_validate(result.as_dict())


@router.post(
    "/pairplot",
    response_model=PairPlotResponse,
    summary="Sampled scatter matrix of every measurement (VIZ-4)",
)
async def pairplot(
    session: SessionDep, settings: SettingsDep, request: PairPlotRequest | None = None
) -> PairPlotResponse:
    """VIZ-4 (task 12.5). At most 1,500 evenly sampled complete rows."""
    request = request or PairPlotRequest()
    result = service.pair_plot(
        session,
        settings,
        color_by=request.color_by,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        use_cache=request.use_cache,
    )
    return PairPlotResponse.model_validate(result.as_dict())


@router.post(
    "/andrews",
    response_model=AndrewsResponse,
    summary="Andrews curves per class, with exact mean curves (VIZ-5)",
)
async def andrews(
    session: SessionDep, settings: SettingsDep, request: AndrewsRequest | None = None
) -> AndrewsResponse:
    """VIZ-5 (task 12.6). Standardised, fixed column order, ≤ 60 curves per class."""
    request = request or AndrewsRequest()
    result = service.andrews_curves(
        session,
        settings,
        class_by=request.class_by,
        source_ids=request.source_tuple(),
        start=request.start,
        end=request.end,
        use_cache=request.use_cache,
    )
    return AndrewsResponse.model_validate(result.as_dict())


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
    "AndrewsRequest",
    "AndrewsResponse",
    "DatasetSelector",
    "DecompositionRequest",
    "GroupedRequest",
    "GroupedResponse",
    "InsufficientDataError",
    "PairPlotRequest",
    "PairPlotResponse",
    "ProfileResponse",
    "ProjectionRequest",
    "ProjectionResponse",
    "ReductionRequest",
    "ReductionResponse",
    "ScatterRequest",
    "ScatterResponse",
    "ScopeSelector",
    "router",
]
