"""Ingestion engine — FEAT-01 (tasks 2.7, 2.8, 2.9).

The single write path to ``observations``. Every mode of design §6.3 —
Scheduled, Manual, Upload, Demo — arrives here; only the adapter differs.

Each source goes through the same five stages:

    fetch → validate (quarantine failures) → harmonize → upsert → log

Three behaviours are load-bearing rather than incidental:

* **A failed fetch never fails the caller** (DR-1). The source is marked
  ``offline``, the run is logged as ``failed``, and the platform carries on
  with the data it already has. Availability is never coupled to a third party.
* **Malformed records are quarantined, not dropped** (task 2.7). A record that
  fails its schema is stored with the reason, because a provider changing its
  payload shape should surface as a visible pile of quarantined records rather
  than as an unexplained gap in the data three weeks later.
* **Upsert, not insert** (task 2.7). Re-ingesting an hour updates it in place,
  and a new NULL never erases a value that is already known — a partial fetch
  mid-hour must not delete what a complete fetch wrote earlier.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.config import IngestionMode, Settings, get_settings
from core.exceptions import EcoCityPulseError, UnknownSourceError
from db.models import (
    DataSource,
    IngestionRun,
    Observation,
    QuarantinedRecord,
    RunStatus,
    SourceStatus,
)
from services import adapters as adapter_registry
from services import geo_service, harmonizer
from services.adapters.base import SourceAdapter
from services.geo_service import Station
from services.harmonizer import MEASUREMENT_FIELDS, SourceReading

logger = logging.getLogger(__name__)

# Rows per upsert batch. Large enough to hide the round trips, small enough to
# stay well inside the driver's parameter limits.
BATCH_SIZE = 2_000

# A file that is malformed from the first byte should not write one quarantine
# row per line. The cap keeps a bad upload from filling the table; the run
# message records that the rest were counted but not stored.
QUARANTINE_LIMIT = 500


@dataclass(slots=True)
class IngestionOutcome:
    """What one source's run did. Returned to the caller and to the API."""

    source_name: str
    mode: str
    status: RunStatus
    records_fetched: int = 0
    records_valid: int = 0
    records_quarantined: int = 0
    records_written: int = 0
    message: str | None = None
    run_id: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in (RunStatus.SUCCESS, RunStatus.PARTIAL)


@dataclass(slots=True)
class QuarantineEntry:
    reason: str
    payload: Any = field(default=None)


# --- Source registry --------------------------------------------------------


def ensure_sources(session: Session) -> dict[str, DataSource]:
    """Reflect the adapter registry into ``data_sources``; return by name.

    Runs before every ingestion so a newly added adapter appears in the admin
    view immediately, and so ``api_url`` corrections in code reach the database
    without a migration.
    """
    existing = {source.name: source for source in session.scalars(select(DataSource))}

    for spec in adapter_registry.registered_specs():
        source = existing.get(spec.name)
        if source is None:
            source = DataSource(
                name=spec.name, api_url=spec.api_url, status=SourceStatus.OFFLINE
            )
            session.add(source)
            existing[spec.name] = source
        elif source.api_url != spec.api_url:
            source.api_url = spec.api_url

    session.flush()
    return existing


def _resolve_source(session: Session, name: str) -> DataSource:
    source = session.scalar(select(DataSource).where(DataSource.name == name))
    if source is None:
        raise UnknownSourceError(f"No data source named {name!r}.", details={"name": name})
    return source


# --- The write path (task 2.7) ---------------------------------------------


def _json_safe(value: Any) -> Any:
    """Make a payload storable as JSONB without losing the evidence.

    ``default=str`` in spirit: anything the JSON encoder cannot represent is
    stringified rather than dropped, because the point of quarantine is to keep
    what actually arrived.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def write_observations(
    session: Session, readings: Sequence[SourceReading], *, source_id: int
) -> int:
    """Upsert harmonized readings. The only place ``observations`` is written.

    On conflict the measurement columns are set to ``COALESCE(new, existing)``:
    a fresh value wins, but a NULL leaves what is already stored alone.
    ``is_anomaly`` is deliberately untouched — it belongs to the Phase 3
    quality engine, and re-ingestion must not silently unflag a record.
    """
    if not readings:
        return 0

    dialect = session.get_bind().dialect.name
    written = 0

    for start in range(0, len(readings), BATCH_SIZE):
        batch = [
            {
                "source_id": source_id,
                "timestamp": reading.timestamp,
                "lat": reading.lat,
                "lon": reading.lon,
                **reading.measurements(),
            }
            for reading in readings[start : start + BATCH_SIZE]
        ]

        if dialect == "postgresql":
            insert = pg_insert(Observation)
            statement = insert.on_conflict_do_update(
                constraint="uq_observations_source_timestamp_location",
                set_={
                    name: func.coalesce(
                        insert.excluded[name], Observation.__table__.c[name]
                    )
                    for name in MEASUREMENT_FIELDS
                },
            )
        else:  # pragma: no cover - PostgreSQL is the supported target
            statement = Observation.__table__.insert()

        session.execute(statement, batch)
        written += len(batch)

    return written


# --- One source ------------------------------------------------------------


def ingest_source(
    session: Session,
    settings: Settings,
    adapter: SourceAdapter,
    *,
    mode: IngestionMode,
    stations: Sequence[Station] | None = None,
) -> IngestionOutcome:
    """Run one source end to end. Never raises for an upstream failure (DR-1)."""
    source = _resolve_source(session, adapter.name)
    stations = list(stations if stations is not None else geo_service.stations_from_districts(settings))

    started = datetime.now(timezone.utc)
    run = IngestionRun(
        source_id=source.id,
        mode=mode.value,
        status=RunStatus.SKIPPED,
        started_at=started,
    )
    session.add(run)
    session.flush()

    outcome = IngestionOutcome(
        source_name=adapter.name, mode=mode.value, status=RunStatus.SKIPPED, run_id=run.id
    )

    # --- not switched on -----------------------------------------------------
    if not adapter.is_configured(settings):
        source.status = SourceStatus.OFFLINE
        outcome.message = (
            f"{adapter.name} has no API key configured; skipped without contacting it."
        )
        _finish(run, outcome, RunStatus.SKIPPED)
        return outcome

    # --- fetch ---------------------------------------------------------------
    try:
        payloads = list(adapter.fetch(settings, stations=stations))
    except Exception as exc:  # noqa: BLE001 - DR-1: no upstream may break us
        # Broad on purpose. A third party can fail in ways no adapter
        # anticipated (a proxy returning HTML, a DNS hijack, a TLS error); the
        # requirement is that none of them take the platform down with them.
        source.status = SourceStatus.OFFLINE
        outcome.message = f"Fetch failed: {exc}"
        logger.warning("Ingestion fetch failed for %s: %s", adapter.name, exc)
        _finish(run, outcome, RunStatus.FAILED)
        return outcome

    outcome.records_fetched = len(payloads)

    # --- validate ------------------------------------------------------------
    readings: list[SourceReading] = []
    quarantine: list[QuarantineEntry] = []

    for payload in payloads:
        try:
            readings.extend(adapter.parse(payload))
        except (ValidationError, EcoCityPulseError, ValueError, TypeError, KeyError) as exc:
            quarantine.append(QuarantineEntry(reason=_reason(exc), payload=payload))

    outcome.records_valid = len(readings)
    outcome.records_quarantined = len(quarantine)
    _store_quarantine(session, run, quarantine)

    # --- harmonize + write ---------------------------------------------------
    harmonized = harmonizer.harmonize(readings)
    outcome.records_written = write_observations(session, harmonized, source_id=source.id)

    # --- health --------------------------------------------------------------
    source.last_run = started
    if quarantine:
        source.status = SourceStatus.DEGRADED
        outcome.status = RunStatus.PARTIAL
        outcome.message = (
            f"{len(quarantine)} of {len(payloads)} records failed validation "
            "and were quarantined."
        )
        if len(quarantine) > QUARANTINE_LIMIT:
            outcome.message += f" Only the first {QUARANTINE_LIMIT} were stored."
    else:
        source.status = SourceStatus.HEALTHY
        outcome.status = RunStatus.SUCCESS

    _finish(run, outcome, outcome.status)
    return outcome


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ())) or "record"
        return f"schema: {location}: {first.get('msg', 'invalid')}"
    if isinstance(exc, EcoCityPulseError):
        return f"{exc.code}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def _store_quarantine(
    session: Session, run: IngestionRun, entries: Sequence[QuarantineEntry]
) -> None:
    for entry in entries[:QUARANTINE_LIMIT]:
        session.add(
            QuarantinedRecord(
                run_id=run.id, reason=entry.reason[:500], payload=_json_safe(entry.payload)
            )
        )


def _finish(run: IngestionRun, outcome: IngestionOutcome, status: RunStatus) -> None:
    outcome.status = status
    run.status = status
    run.finished_at = datetime.now(timezone.utc)
    run.records_fetched = outcome.records_fetched
    run.records_valid = outcome.records_valid
    run.records_quarantined = outcome.records_quarantined
    run.records_written = outcome.records_written
    run.message = outcome.message


# --- The four modes (task 2.9) ---------------------------------------------


def live_adapters_for(settings: Settings) -> list[SourceAdapter]:
    """The adapters a Scheduled or Manual run uses.

    Task 2.2: with no TomTom key, the synthetic generator takes traffic's place
    rather than the column simply going empty. PM2.5 without traffic loses the
    strongest explanatory variable in the dataset, so a modelled stand-in —
    clearly labelled as one — is better than nothing.
    """
    chosen: list[SourceAdapter] = []
    for adapter in adapter_registry.live_adapters():
        if isinstance(adapter, adapter_registry.TomTomAdapter) and not adapter.is_configured(
            settings
        ):
            chosen.append(adapter_registry.SyntheticTrafficAdapter())
            continue
        chosen.append(adapter)
    return chosen


def run_live(
    session: Session, settings: Settings, *, mode: IngestionMode
) -> list[IngestionOutcome]:
    """Scheduled and Manual modes: identical work, different trigger."""
    ensure_sources(session)
    stations = geo_service.stations_from_districts(settings)
    return [
        ingest_source(session, settings, adapter, mode=mode, stations=stations)
        for adapter in live_adapters_for(settings)
    ]


def run_demo(
    session: Session,
    settings: Settings,
    *,
    days: int | None = None,
    end: datetime | None = None,
    seed: int | None = None,
) -> list[IngestionOutcome]:
    """Demo mode: the offline bundle, through the same pipeline (AC-2)."""
    from services.adapters.demo import DemoAdapter
    from services.demo_data import DEFAULT_DAYS, DEFAULT_SEED

    ensure_sources(session)
    adapter = DemoAdapter(
        days=days if days is not None else DEFAULT_DAYS,
        end=end,
        seed=seed if seed is not None else DEFAULT_SEED,
    )
    return [ingest_source(session, settings, adapter, mode=IngestionMode.DEMO)]


def run_upload(
    session: Session,
    settings: Settings,
    rows: Iterable[Mapping[str, Any]],
    *,
    assume_timezone: timezone = timezone.utc,
    filename: str | None = None,
) -> IngestionOutcome:
    """Upload mode: rows already parsed from a CSV or JSON file."""
    from services.adapters.upload import UploadAdapter

    ensure_sources(session)
    adapter = UploadAdapter(
        list(rows), assume_timezone=assume_timezone, filename=filename
    )
    outcome = ingest_source(session, settings, adapter, mode=IngestionMode.UPLOAD)

    # DR-2: the assumption applied to naive timestamps is part of the record,
    # not an implementation detail someone has to reverse-engineer later.
    note = f"Source: {filename or 'upload'}; naive timestamps read as {assume_timezone}."
    outcome.message = f"{outcome.message} {note}" if outcome.message else note
    _update_run_message(session, outcome)
    return outcome


def _update_run_message(session: Session, outcome: IngestionOutcome) -> None:
    if outcome.run_id is None:
        return
    run = session.get(IngestionRun, outcome.run_id)
    if run is not None:
        run.message = outcome.message


def run_ingestion(
    session: Session,
    settings: Settings | None = None,
    *,
    mode: IngestionMode | None = None,
    **kwargs: Any,
) -> list[IngestionOutcome]:
    """Dispatch to the mode's runner. The one entry point callers need."""
    settings = settings or get_settings()
    mode = mode or settings.ingestion_mode

    if mode is IngestionMode.DEMO:
        return run_demo(session, settings, **kwargs)
    if mode is IngestionMode.UPLOAD:
        return [run_upload(session, settings, **kwargs)]
    return run_live(session, settings, mode=mode)


__all__ = [
    "BATCH_SIZE",
    "QUARANTINE_LIMIT",
    "IngestionOutcome",
    "ensure_sources",
    "ingest_source",
    "live_adapters_for",
    "run_demo",
    "run_ingestion",
    "run_live",
    "run_upload",
    "write_observations",
]
