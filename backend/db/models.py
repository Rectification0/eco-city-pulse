"""ORM models — the four tables required by specs §9 / design §6.1.

Column sets follow the specification exactly; nothing speculative is added.
Constraints encode the invariants the harmonizer (DR-2 ... DR-4) must uphold,
so a violation fails at the write rather than surfacing as a strange chart
several phases later.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.types import JSON

from db.base import TIMESTAMPTZ, Base, normalize_utc

# JSONB on PostgreSQL (indexable, typed) but plain JSON elsewhere, so the model
# layer stays loadable in unit tests that never reach a server.
JSON_VARIANT = JSON().with_variant(JSONB, "postgresql")


class SourceStatus(str, Enum):
    """Ingestion health of a source (design §6.3).

    ``OFFLINE`` is a normal steady state, not an error: with no API key
    configured a source never goes live, and the platform keeps running on
    persisted data (DR-1).
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class DataSource(Base):
    """A provider of observations: a live API, an upload, or the demo bundle."""

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Nullable: an uploaded CSV and the demo bundle have no endpoint.
    api_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[SourceStatus] = mapped_column(
        SAEnum(
            SourceStatus,
            name="source_status",
            native_enum=False,  # VARCHAR + CHECK: no ALTER TYPE dance on change
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
        default=SourceStatus.OFFLINE,
        server_default=SourceStatus.OFFLINE.value,
    )
    last_run: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)

    observations: Mapped[list[Observation]] = relationship(
        back_populates="source", cascade="save-update, merge", passive_deletes=True
    )
    # The log goes with the source: unlike observations, a run record has no
    # meaning once the source it describes is gone.
    runs: Mapped[list[IngestionRun]] = relationship(
        back_populates="source", cascade="all, delete-orphan", passive_deletes=True
    )

    @validates("last_run")
    def _validate_last_run(self, key: str, value: datetime | None) -> datetime | None:
        return normalize_utc(value, field=key)

    def __repr__(self) -> str:
        return f"<DataSource id={self.id} name={self.name!r} status={self.status}>"


class Observation(Base):
    """One harmonized hourly reading at one location (DR-2 ... DR-4).

    The row is deliberately wide rather than key/value: every source is
    resampled onto the same hourly grid before it lands here, so a single row
    is the joined environmental state at a point in space and time. That is
    what makes the lag and rolling features of Phase 5 well defined.
    """

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        # RESTRICT, not CASCADE: deleting a source must never silently destroy
        # the history that models were trained on.
        # No separate index on source_id: the unique constraint below already
        # leads with it, and PostgreSQL serves prefix lookups from that.
        ForeignKey("data_sources.id", ondelete="RESTRICT"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)

    # Measurements are nullable by design: missingness is the signal the data
    # quality engine analyses (Phase 3), so it must survive the write path.
    pm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    traffic_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_anomaly: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    source: Mapped[DataSource] = relationship(back_populates="observations")

    __table_args__ = (
        # DR-3: decimal degrees, so anything outside these bounds is a unit
        # error (radians, or lat/lon transposed) rather than a real place.
        CheckConstraint("lat BETWEEN -90 AND 90", name="lat_decimal_degrees"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="lon_decimal_degrees"),
        CheckConstraint(
            "humidity IS NULL OR humidity BETWEEN 0 AND 100", name="humidity_percent"
        ),
        # Pollutant values are intentionally unconstrained. A genuine spike must
        # reach the table so the quality engine can flag it (AC-5); rejecting
        # extremes here would delete exactly the records the spec protects.
        # Re-ingesting the same window must not duplicate rows: this is the
        # conflict target the Phase 2 upsert writes against.
        UniqueConstraint(
            "source_id",
            "timestamp",
            "lat",
            "lon",
            name="uq_observations_source_timestamp_location",
        ),
        # design §6.1: one composite index serving both time-series windows and
        # map queries (task 1.6).
        Index("ix_observations_timestamp_lat_lon", "timestamp", "lat", "lon"),
    )

    @validates("timestamp")
    def _validate_timestamp(self, key: str, value: datetime) -> datetime:
        normalized = normalize_utc(value, field=key)
        assert normalized is not None  # column is NOT NULL
        return normalized

    def __repr__(self) -> str:
        return (
            f"<Observation id={self.id} t={self.timestamp!r} "
            f"({self.lat}, {self.lon}) pm25={self.pm25}>"
        )


class MLModel(Base):
    """Model registry entry (specs §9 ``models``).

    Named ``MLModel`` because a class called ``Model`` sitting next to Pydantic
    ``BaseModel`` in the same import graph reads as a bug. The table name still
    follows the spec.
    """

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # e.g. pm25_h1 | pm25_h6 | pm25_h24. Left as free text: the horizon set is a
    # Phase 7 decision and the spec offers these only as examples.
    # Indexed through ix_models_target_created_at below, whose leading column
    # is target -- a standalone index on it would be redundant.
    target: Mapped[str] = mapped_column(Text, nullable=False)
    features_used: Mapped[list[str] | dict[str, Any]] = mapped_column(
        JSON_VARIANT, nullable=False
    )

    # Nullable so a run can register an artifact before scoring finishes; the
    # normal path (task 7.11 then 7.12) writes them together.
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="model", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("mae IS NULL OR mae >= 0", name="mae_non_negative"),
        CheckConstraint("rmse IS NULL OR rmse >= 0", name="rmse_non_negative"),
        # Inference resolves "the current model for this target" as the newest
        # row for that target, which is exactly this index.
        Index("ix_models_target_created_at", "target", created_at.desc()),
    )

    @validates("created_at")
    def _validate_created_at(self, key: str, value: datetime | None) -> datetime | None:
        return normalize_utc(value, field=key)

    def __repr__(self) -> str:
        return f"<MLModel id={self.id} name={self.name!r} target={self.target!r}>"


class Prediction(Base):
    """A single issued forecast, plus the outcome once it is known.

    Append-only: a re-run writes a new row rather than overwriting, because the
    sequence of predictions is itself the drift-monitoring dataset (design §6.1).
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    target_time: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    # Nullable on purpose: backfilled once the real observation lands (task 9.6).
    actual_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    model: Mapped[MLModel] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_model_id_target_time", "model_id", "target_time"),
        # The backfill job scans by time across every model.
        Index("ix_predictions_target_time", "target_time"),
    )

    @validates("target_time")
    def _validate_target_time(self, key: str, value: datetime) -> datetime:
        normalized = normalize_utc(value, field=key)
        assert normalized is not None
        return normalized

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} model_id={self.model_id} "
            f"t={self.target_time!r} value={self.predicted_value}>"
        )


class RunStatus(str, Enum):
    """Outcome of one ingestion attempt against one source.

    ``PARTIAL`` and ``FAILED`` are distinct on purpose: partial means the data
    arrived but some records were unusable, failed means nothing arrived. An
    administrator needs to tell a bad feed from a dead one (specs §4, Sam).
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class IngestionRun(Base):
    """One ingestion attempt — the ingestion log FEAT-01 names as its output.

    Beyond the four tables of specs §9 because the specification requires the
    log to exist without saying where it lives. It is a run-level record; the
    individual records that failed validation go to ``quarantined_records``.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        SAEnum(
            RunStatus,
            name="run_status",
            native_enum=False,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)

    # Four counts rather than one: "500 fetched, 500 written" and "500 fetched,
    # 3 written" are the same run as far as a single total is concerned, and
    # only one of them is healthy.
    records_fetched: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    records_valid: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    records_quarantined: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    records_written: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    # Free text for the human reading the admin view: why a run failed, or
    # which fallback was substituted.
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[DataSource] = relationship(back_populates="runs")
    quarantined: Mapped[list[QuarantinedRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_ingestion_runs_source_id_started_at", "source_id", started_at.desc()),
    )

    @validates("started_at", "finished_at")
    def _validate_times(self, key: str, value: datetime | None) -> datetime | None:
        return normalize_utc(value, field=key)

    def __repr__(self) -> str:
        return (
            f"<IngestionRun id={self.id} source_id={self.source_id} "
            f"mode={self.mode} status={self.status}>"
        )


class QuarantinedRecord(Base):
    """A record that failed its source schema, kept with the reason why.

    Quarantine rather than discard (task 2.7): a malformed record is evidence
    about an upstream feed, and silently dropping it turns a schema change at
    the provider into an unexplained gap in the data weeks later.
    """

    __tablename__ = "quarantined_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    # The payload exactly as it arrived, so the failure can be reproduced.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_VARIANT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    run: Mapped[IngestionRun] = relationship(back_populates="quarantined")

    __table_args__ = (Index("ix_quarantined_records_run_id", "run_id"),)

    def __repr__(self) -> str:
        return f"<QuarantinedRecord id={self.id} run_id={self.run_id} reason={self.reason!r}>"


__all__ = [
    "Base",
    "DataSource",
    "IngestionRun",
    "MLModel",
    "Observation",
    "Prediction",
    "QuarantinedRecord",
    "RunStatus",
    "SourceStatus",
]
