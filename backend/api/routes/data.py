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
from typing import Annotated, Literal

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
from services import geo_service, ingestion_service

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
