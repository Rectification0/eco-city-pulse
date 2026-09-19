"""Data-layer routes: boundaries, source health, and ingestion triggers.

Per design §5 these handlers contain no ingestion logic. They validate input,
call ``ingestion_service``, and shape the response.

The trigger endpoints are the Manual and Upload modes of design §6.3
(task 2.9); Scheduled runs from the background task in ``main.py``, and Demo
runs at container start.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import SessionDep, SettingsDep
from core.config import IngestionMode, Settings
from core.exceptions import UploadRejectedError
from db.models import (
    DataSource,
    IngestionRun,
    Observation,
    QuarantinedRecord,
    RunStatus,
)
from services import adapters as adapter_registry
from services import datasets, geo_service, ingestion_service, quality

router = APIRouter(prefix="/data", tags=["data"])

Position = Annotated[
    tuple[float, float],
    Field(description="[longitude, latitude] in decimal degrees (DR-3)."),
]


class DistrictProperties(BaseModel):
    """SEC-1: the boundary file is validated on the way out, not trusted."""

    district_id: str
    name: str
    city: str
    state: str
    country: str
    centroid_lat: float = Field(ge=-90, le=90)
    centroid_lon: float = Field(ge=-180, le=180)


class DistrictGeometry(BaseModel):
    type: Literal["Polygon"]
    # GeoJSON nests rings: [exterior, *holes], each a list of positions.
    coordinates: list[list[Position]]


class DistrictFeature(BaseModel):
    type: Literal["Feature"]
    id: str
    properties: DistrictProperties
    geometry: DistrictGeometry


class DistrictMetadata(BaseModel):
    """Provenance travels with the data.

    ``accuracy`` reaches the UI deliberately: the polygons are schematic, and a
    map that silently implies surveyed boundaries would be the same kind of
    overclaim the ethics requirement rules out elsewhere (ETH-1).
    """

    city: str
    crs: str
    units: str
    district_count: int
    bbox: tuple[float, float, float, float]
    accuracy: str
    license: str


class DistrictCollection(BaseModel):
    type: Literal["FeatureCollection"]
    name: str
    metadata: DistrictMetadata
    features: list[DistrictFeature]


@router.get(
    "/districts",
    response_model=DistrictCollection,
    summary="City district boundaries (GeoJSON)",
)
async def get_districts(settings: SettingsDep) -> DistrictCollection:
    """District polygons for the dashboard map layer (specs §5.1, task 10.4)."""
    return DistrictCollection.model_validate(geo_service.load_districts(settings))


# --- Source health (task 2.10) ----------------------------------------------


class RunSummary(BaseModel):
    """The most recent ingestion attempt for a source."""

    run_id: int
    mode: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    records_fetched: int
    records_valid: int
    records_quarantined: int
    records_written: int
    message: str | None


class SourceHealth(BaseModel):
    """specs §8: a source plus enough context to judge whether to trust it."""

    id: int
    name: str
    api_url: str | None
    status: str = Field(description="healthy · degraded · offline")
    last_run: datetime | None
    domain: str | None = Field(
        default=None, description="What the source measures, from the adapter registry."
    )
    requires_credentials: bool = Field(
        description="False for the demo bundle, uploads, and the synthetic fallback."
    )
    credentials_configured: bool = Field(
        description=(
            "Whether the source is ready to run. For a source that needs no "
            "key this is always true. SEC-2: never the key itself."
        )
    )
    observation_count: int
    last_observation_at: datetime | None
    latest_run: RunSummary | None


class SourcesResponse(BaseModel):
    sources: list[SourceHealth]
    ingestion_mode: IngestionMode
    generated_at: datetime


@router.get(
    "/sources",
    response_model=SourcesResponse,
    summary="Data sources with ingestion health",
)
async def get_sources(session: SessionDep, settings: SettingsDep) -> SourcesResponse:
    """List every registered source and how its ingestion is going (task 2.10)."""
    ingestion_service.ensure_sources(session)

    specs = {spec.name: spec for spec in adapter_registry.registered_specs()}
    configured = _configured_sources(settings)

    # Per-source aggregates in two grouped queries rather than N+1 lookups.
    counts = {
        source_id: (total, latest)
        for source_id, total, latest in session.execute(
            select(
                Observation.source_id,
                func.count(Observation.id),
                func.max(Observation.timestamp),
            ).group_by(Observation.source_id)
        )
    }

    sources = session.scalars(select(DataSource).order_by(DataSource.id)).all()
    latest_runs = _latest_runs(session, [source.id for source in sources])

    return SourcesResponse(
        ingestion_mode=settings.ingestion_mode,
        generated_at=datetime.now(timezone.utc),
        sources=[
            SourceHealth(
                id=source.id,
                name=source.name,
                api_url=source.api_url,
                status=source.status.value,
                last_run=source.last_run,
                domain=specs[source.name].domain.value if source.name in specs else None,
                requires_credentials=(
                    specs[source.name].requires_credentials
                    if source.name in specs
                    else True
                ),
                credentials_configured=configured.get(source.name, True),
                observation_count=counts.get(source.id, (0, None))[0],
                last_observation_at=counts.get(source.id, (0, None))[1],
                latest_run=latest_runs.get(source.id),
            )
            for source in sources
        ],
    )


def _configured_sources(settings: Settings) -> dict[str, bool]:
    """Which adapters are ready to run, without ever reading a key (SEC-2)."""
    return {
        adapter.name: adapter.is_configured(settings)
        for adapter in adapter_registry.live_adapters()
    }


def _latest_runs(session: Session, source_ids: list[int]) -> dict[int, RunSummary]:
    if not source_ids:
        return {}

    # DISTINCT ON is the one-query way to get "newest row per group" in
    # PostgreSQL; the composite index on (source_id, started_at DESC) serves it.
    newest = (
        select(IngestionRun)
        .where(IngestionRun.source_id.in_(source_ids))
        .order_by(IngestionRun.source_id, IngestionRun.started_at.desc())
        .distinct(IngestionRun.source_id)
    )

    return {
        run.source_id: RunSummary(
            run_id=run.id,
            mode=run.mode,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            records_fetched=run.records_fetched,
            records_valid=run.records_valid,
            records_quarantined=run.records_quarantined,
            records_written=run.records_written,
            message=run.message,
        )
        for run in session.scalars(newest)
    }


# --- Ingestion triggers (task 2.9) ------------------------------------------


class OutcomeResponse(BaseModel):
    """One source's result from a triggered run."""

    source_name: str
    mode: str
    status: RunStatus
    records_fetched: int
    records_valid: int
    records_quarantined: int
    records_written: int
    message: str | None
    run_id: int | None


class IngestionResponse(BaseModel):
    mode: IngestionMode
    outcomes: list[OutcomeResponse]
    triggered_at: datetime


def _to_response(
    mode: IngestionMode, outcomes: list[ingestion_service.IngestionOutcome]
) -> IngestionResponse:
    return IngestionResponse(
        mode=mode,
        triggered_at=datetime.now(timezone.utc),
        # asdict, not vars: IngestionOutcome uses __slots__ and so has no
        # __dict__ to read.
        outcomes=[OutcomeResponse(**asdict(outcome)) for outcome in outcomes],
    )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Trigger an ingestion run (Manual mode)",
)
async def trigger_ingestion(
    session: SessionDep,
    settings: SettingsDep,
    mode: Annotated[
        IngestionMode | None,
        Query(description="Defaults to MANUAL, which fetches every live source."),
    ] = None,
    days: Annotated[
        int | None,
        Query(ge=1, le=730, description="Demo mode only: days of history to generate."),
    ] = None,
) -> IngestionResponse:
    """The Administrator's manual trigger (specs §4, Sam).

    Returns **200 with per-source outcomes even when a source failed**. That is
    DR-1 expressed in the API: an unreachable third party degrades that source
    to ``offline`` and is reported, it does not turn into a 5xx for the caller.
    """
    chosen = mode or IngestionMode.MANUAL

    if chosen is IngestionMode.DEMO:
        outcomes = ingestion_service.run_demo(session, settings, days=days)
    elif chosen is IngestionMode.UPLOAD:
        raise UploadRejectedError(
            "Upload mode needs a file; use POST /data/upload instead."
        )
    else:
        outcomes = ingestion_service.run_live(session, settings, mode=chosen)

    return _to_response(chosen, outcomes)


@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV or JSON file (Upload mode)",
)
async def upload_observations(
    session: SessionDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="CSV or JSON observations.")],
    assume_timezone_offset_minutes: Annotated[
        int,
        Form(
            ge=-840,
            le=840,
            description=(
                "Offset applied to timestamps that carry no timezone. "
                "0 = UTC. Recorded in the ingestion log (DR-2)."
            ),
        ),
    ] = 0,
) -> IngestionResponse:
    """Parse an uploaded file and ingest it through the normal pipeline.

    The size limit is enforced on the bytes actually read rather than on the
    declared ``Content-Length``, which a client controls and can understate.
    """
    content = await file.read(settings.upload_max_bytes + 1)
    if len(content) > settings.upload_max_bytes:
        raise UploadRejectedError(
            f"The upload exceeds the {settings.upload_max_bytes} byte limit.",
            details={"filename": file.filename, "limit": settings.upload_max_bytes},
        )

    rows = adapter_registry.parse_upload(content, filename=file.filename)
    outcome = ingestion_service.run_upload(
        session,
        settings,
        rows,
        assume_timezone=timezone(timedelta(minutes=assume_timezone_offset_minutes)),
        filename=file.filename,
    )
    return _to_response(IngestionMode.UPLOAD, [outcome])


# --- Ingestion log (task 2.8) -----------------------------------------------


class QuarantineEntryResponse(BaseModel):
    id: int
    run_id: int
    reason: str
    created_at: datetime


class IngestionRunsResponse(BaseModel):
    runs: list[RunSummary]
    quarantined_sample: list[QuarantineEntryResponse]


@router.get(
    "/ingestion/runs",
    response_model=IngestionRunsResponse,
    summary="Recent ingestion runs and quarantined records",
)
async def get_ingestion_runs(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
) -> IngestionRunsResponse:
    """The log FEAT-01 names as its output; the Admin view's data (task 10.15).

    The quarantine sample deliberately omits the stored payload: it can contain
    an arbitrary third-party document, and this endpoint exists to answer "is
    ingestion healthy", not to serve raw upstream bodies to a browser.
    """
    runs = session.scalars(
        select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(limit)
    ).all()

    quarantined = session.scalars(
        select(QuarantinedRecord)
        .where(QuarantinedRecord.run_id.in_([run.id for run in runs] or [-1]))
        .order_by(QuarantinedRecord.id.desc())
        .limit(limit)
    ).all()

    return IngestionRunsResponse(
        runs=[
            RunSummary(
                run_id=run.id,
                mode=run.mode,
                status=run.status,
                started_at=run.started_at,
                finished_at=run.finished_at,
                records_fetched=run.records_fetched,
                records_valid=run.records_valid,
                records_quarantined=run.records_quarantined,
                records_written=run.records_written,
                message=run.message,
            )
            for run in runs
        ],
        quarantined_sample=[
            QuarantineEntryResponse(
                id=record.id,
                run_id=record.run_id,
                reason=record.reason,
                created_at=record.created_at,
            )
            for record in quarantined
        ],
    )


# --- Data quality (Phase 3; specs §4 — the Analyst triggers preprocessing) ---


class AssociationResponse(BaseModel):
    related_to: str
    discrimination: float = Field(
        description="AUC for separating missing rows from observed ones; 0.5 = none."
    )
    direction: str
    p_value: float
    sample_size: int
    significant: bool


class ColumnMissingnessResponse(BaseModel):
    column: str
    total: int
    missing: int
    missing_pct: float
    longest_gap_rows: int
    mechanism: str = Field(description="complete · mcar · mar · mnar")
    associations: list[AssociationResponse]
    justification: str
    caveat: str


class GridCompletenessResponse(BaseModel):
    expected_hours: int
    present_hours: int
    absent_hours: int
    completeness_pct: float


class CoMissingnessResponse(BaseModel):
    column: str
    also_missing: dict[str, float]


class MissingnessResponse(BaseModel):
    columns: list[ColumnMissingnessResponse]
    grid: GridCompletenessResponse
    co_missingness: list[CoMissingnessResponse]
    caveats: list[str]


class ImputationResponse(BaseModel):
    filled_short_gaps: dict[str, int]
    imputed_by_mice: dict[str, int]
    remaining_nulls: dict[str, int]
    max_gap_hours: int
    backward_fill_used: bool
    is_complete: bool = Field(description="AC-4: no nulls left in the feature set.")
    stations_imputed: int
    stations_not_converged: int = Field(
        description=(
            "Stations where the chained equations were still moving when the "
            "iteration budget ran out. Reported rather than hidden: the values "
            "stay inside the observed range, but a reader deserves to know "
            "which ones the model had not settled on."
        )
    )


class AnomalyResponse(BaseModel):
    rows: int
    flagged: int
    flagged_pct: float
    votes_by_detector: dict[str, int]
    flagged_by_column: dict[str, int]
    min_votes: int
    note: str


class QualityOutputsResponse(BaseModel):
    cleaned: str | None
    report: str | None


class QualityRunResponse(BaseModel):
    """The full account of a cleaning run (AC-4, AC-5)."""

    generated_at: datetime
    window: dict[str, Any]
    rows_in: int
    rows_out: int
    rows_preserved: bool = Field(
        description="AC-5: the engine flags records, it never removes one."
    )
    feature_set_is_complete: bool
    flags_written: int
    missingness: MissingnessResponse
    imputation: ImputationResponse
    anomalies: AnomalyResponse
    outputs: QualityOutputsResponse


@router.post(
    "/quality",
    response_model=QualityRunResponse,
    summary="Run the data quality pipeline",
)
async def run_quality(
    session: SessionDep,
    settings: SettingsDep,
    max_gap_hours: Annotated[
        int,
        Query(ge=0, le=48, description="Longest gap filled locally before MICE takes over."),
    ] = quality.imputation.DEFAULT_MAX_GAP_HOURS,
    min_votes: Annotated[
        int,
        Query(ge=1, le=3, description="Detectors that must agree before a row is flagged."),
    ] = quality.outliers.DEFAULT_MIN_VOTES,
    backward_fill: Annotated[
        bool,
        Query(
            description=(
                "Backward fill reads future values. Fine for cleaning a historical "
                "record; disable it when preparing a training window (AC-8)."
            )
        ),
    ] = True,
    persist: Annotated[
        bool, Query(description="Write the cleaned frame to data/processed (task 3.8).")
    ] = True,
) -> QualityRunResponse:
    """Impute, detect anomalies, and flag — never delete (AC-4, AC-5).

    The Analyst persona's "trigger preprocessing pipelines" capability
    (specs §4). Returns the full report rather than a job id: the run takes
    seconds on this dataset, and an operator wants the numbers, not a handle.
    """
    report = quality.pipeline.run(
        session,
        settings,
        max_gap_hours=max_gap_hours,
        min_votes=min_votes,
        backward_fill=backward_fill,
        persist=persist,
    )
    return QualityRunResponse.model_validate(report.as_dict())


# --- Current readings (Phase 10) --------------------------------------------
#
# The dependency note in tasks.md has the frontend reading "Phase 2/4/6/7/9
# endpoints", and building Phase 10 showed one missing: nothing served an
# *observation*. The profile returns statistics, the ESI returns an index, the
# predict endpoint returns a forecast -- but the dashboard's hero tile (10.3)
# and the map gradient (10.4) both need what the sensors currently say.
#
# Two reads, both narrow on purpose: the latest row per station, and one
# station's recent window. Neither computes anything; the analytics layers
# already exist and this is the raw material they were all built from.


class StationReading(BaseModel):
    """One station's most recent hourly reading."""

    station: str = Field(description="Stable 'lat,lon' key (services.datasets).")
    lat: float
    lon: float
    district_id: str | None = Field(
        default=None, description="District whose centroid this station sits on."
    )
    district_name: str | None = None
    timestamp: datetime
    pm25: float | None
    pm10: float | None
    temp: float | None
    humidity: float | None
    traffic_score: float | None
    is_anomaly: bool = Field(
        description="Flagged by the Phase 3 detectors. Flagged, never removed (AC-5)."
    )


class LatestObservationsResponse(BaseModel):
    readings: list[StationReading]
    observed_at: datetime | None = Field(
        description="Newest timestamp in the set; the dashboard's notion of 'now'."
    )
    stale_minutes: float | None = Field(
        default=None,
        description=(
            "Age of that newest reading. Demo data is generated up to the hour "
            "it was seeded, so this grows until the next ingestion run."
        ),
    )
    generated_at: datetime


class SeriesPoint(BaseModel):
    timestamp: datetime
    pm25: float | None
    pm10: float | None
    temp: float | None
    humidity: float | None
    traffic_score: float | None
    is_anomaly: bool


class StationSeriesResponse(BaseModel):
    station: str
    lat: float
    lon: float
    hours: int
    points: list[SeriesPoint]
    generated_at: datetime


@router.get(
    "/observations/latest",
    response_model=LatestObservationsResponse,
    summary="The most recent reading from every station",
)
async def latest_observations(
    session: SessionDep, settings: SettingsDep
) -> LatestObservationsResponse:
    """What the sensors currently say (tasks 10.3, 10.4).

    One row per station, resolved with ``DISTINCT ON`` so the database does the
    per-group work rather than the application pulling every row and filtering
    in Python.
    """
    statement = (
        select(Observation)
        .distinct(Observation.lat, Observation.lon)
        .order_by(Observation.lat, Observation.lon, Observation.timestamp.desc())
    )
    rows = list(session.scalars(statement))

    districts = {
        district_id: (lat, lon)
        for district_id, (lat, lon) in geo_service.district_centroids(settings).items()
    }
    names = {
        feature["id"]: feature["properties"]["name"]
        for feature in geo_service.load_districts(settings)["features"]
    }
    by_point = {
        (round(lat, 5), round(lon, 5)): district_id
        for district_id, (lat, lon) in districts.items()
    }

    readings = []
    for row in rows:
        district_id = by_point.get((round(row.lat, 5), round(row.lon, 5)))
        readings.append(
            StationReading(
                station=datasets.station_key(row.lat, row.lon),
                lat=row.lat,
                lon=row.lon,
                district_id=district_id,
                district_name=names.get(district_id) if district_id else None,
                timestamp=row.timestamp,
                pm25=row.pm25,
                pm10=row.pm10,
                temp=row.temp,
                humidity=row.humidity,
                traffic_score=row.traffic_score,
                is_anomaly=bool(row.is_anomaly),
            )
        )

    now = datetime.now(timezone.utc)
    observed_at = max((r.timestamp for r in readings), default=None)
    return LatestObservationsResponse(
        readings=sorted(readings, key=lambda r: r.district_name or r.station),
        observed_at=observed_at,
        stale_minutes=(
            round((now - observed_at).total_seconds() / 60, 1) if observed_at else None
        ),
        generated_at=now,
    )


@router.get(
    "/observations/series",
    response_model=StationSeriesResponse,
    summary="One station's recent hourly readings",
)
async def station_series(
    session: SessionDep,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    hours: Annotated[int, Query(ge=1, le=720)] = 48,
) -> StationSeriesResponse:
    """The observed history behind the dashboard's trendline (task 10.5)."""
    rows = list(
        session.scalars(
            select(Observation)
            .where(Observation.lat == lat, Observation.lon == lon)
            .order_by(Observation.timestamp.desc())
            .limit(hours)
        )
    )

    return StationSeriesResponse(
        station=datasets.station_key(lat, lon),
        lat=lat,
        lon=lon,
        hours=hours,
        points=[
            SeriesPoint(
                timestamp=row.timestamp,
                pm25=row.pm25,
                pm10=row.pm10,
                temp=row.temp,
                humidity=row.humidity,
                traffic_score=row.traffic_score,
                is_anomaly=bool(row.is_anomaly),
            )
            # Oldest first: a chart reads left to right.
            for row in reversed(rows)
        ],
        generated_at=datetime.now(timezone.utc),
    )
