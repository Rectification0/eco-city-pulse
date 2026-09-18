"""Schema shape and invariants (specs §9, design §6.1, tasks 1.2-1.6).

These assertions run against the SQLAlchemy metadata, so they hold without a
database. The live-server behaviour is covered in ``test_persistence.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import Index, UniqueConstraint

from db.base import Base
from db.models import DataSource, MLModel, Observation, Prediction, SourceStatus

# The column sets the specification names, verbatim.
SPEC_COLUMNS = {
    "data_sources": {"id", "name", "api_url", "status", "last_run"},
    "observations": {
        "id",
        "source_id",
        "timestamp",
        "lat",
        "lon",
        "pm25",
        "pm10",
        "temp",
        "humidity",
        "traffic_score",
        "is_anomaly",
    },
    "models": {
        "id",
        "name",
        "target",
        "features_used",
        "mae",
        "rmse",
        "r2",
        "created_at",
        "artifact_path",
    },
    "predictions": {
        "id",
        "model_id",
        "target_time",
        "predicted_value",
        "actual_value",
    },
}


# Beyond specs §9, and deliberately so: FEAT-01 requires an ingestion log as
# its output without saying where it lives (tasks 2.7, 2.8).
LOG_TABLES = {"ingestion_runs", "quarantined_records"}


def test_all_four_spec_tables_are_defined() -> None:
    assert set(SPEC_COLUMNS) <= set(Base.metadata.tables)


def test_no_table_exists_beyond_the_spec_and_the_ingestion_log() -> None:
    """Keeps table sprawl a decision rather than an accident."""
    assert set(Base.metadata.tables) == set(SPEC_COLUMNS) | LOG_TABLES


@pytest.mark.parametrize(("table", "columns"), SPEC_COLUMNS.items())
def test_columns_match_the_specification(table: str, columns: set[str]) -> None:
    """Exact equality, not a subset: an extra column is drift from the spec."""
    assert {c.name for c in Base.metadata.tables[table].columns} == columns


def test_observations_carry_the_composite_index() -> None:
    """design §6.1: one index serving time-series windows and map queries (1.6)."""
    indexes = {i.name: [c.name for c in i.columns] for i in Observation.__table__.indexes}

    assert indexes["ix_observations_timestamp_lat_lon"] == ["timestamp", "lat", "lon"]


def test_observations_are_unique_per_source_time_and_place() -> None:
    """The conflict target that makes re-ingestion idempotent (Phase 2)."""
    unique = [
        c for c in Observation.__table__.constraints if isinstance(c, UniqueConstraint)
    ]

    assert len(unique) == 1
    assert [c.name for c in unique[0].columns] == ["source_id", "timestamp", "lat", "lon"]


def test_deleting_a_source_cannot_destroy_its_history() -> None:
    """RESTRICT, not CASCADE: training data must outlive a source's removal."""
    foreign_key = next(iter(Observation.__table__.foreign_keys))

    assert foreign_key.column is DataSource.__table__.c.id
    assert foreign_key.ondelete == "RESTRICT"


def test_predictions_are_removed_with_their_model() -> None:
    """A prediction without its model has no meaning, so CASCADE is correct."""
    foreign_key = next(iter(Prediction.__table__.foreign_keys))

    assert foreign_key.column is MLModel.__table__.c.id
    assert foreign_key.ondelete == "CASCADE"


def test_measurement_columns_are_nullable() -> None:
    """Missingness is the signal Phase 3 analyses; it must survive the write."""
    columns = Observation.__table__.columns

    for name in ("pm25", "pm10", "temp", "humidity", "traffic_score"):
        assert columns[name].nullable, f"{name} must accept NULL"


def test_pollutant_values_are_not_range_constrained() -> None:
    """AC-5: a genuine spike must reach the table to be flagged, not rejected."""
    checks = " ".join(
        str(c.sqltext)
        for c in Observation.__table__.constraints
        if hasattr(c, "sqltext")
    )

    assert "pm25" not in checks
    assert "pm10" not in checks


def test_actual_value_is_nullable_for_backfill() -> None:
    """design §6.1: filled once the real observation arrives (task 9.6)."""
    assert Prediction.__table__.columns["actual_value"].nullable


def test_models_are_indexed_for_latest_per_target() -> None:
    index = next(
        i for i in MLModel.__table__.indexes if i.name == "ix_models_target_created_at"
    )

    assert isinstance(index, Index)
    assert index.expressions[0].name == "target"


def test_source_status_values_match_the_design() -> None:
    assert {s.value for s in SourceStatus} == {"healthy", "degraded", "offline"}


# --- DR-2: timezone handling ------------------------------------------------


def test_naive_timestamps_are_rejected() -> None:
    """Assuming a naive datetime is UTC is how a dataset acquires a silent
    phase error that only shows up as a bad model months later."""
    with pytest.raises(ValueError, match="timezone-aware"):
        Observation(source_id=1, timestamp=datetime(2026, 1, 1, 12), lat=0.0, lon=0.0)


def test_aware_timestamps_are_normalized_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    observation = Observation(
        source_id=1, timestamp=datetime(2026, 1, 1, 17, 30, tzinfo=ist), lat=0.0, lon=0.0
    )

    assert observation.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert observation.timestamp.tzinfo is timezone.utc


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda value: DataSource(name="s", last_run=value), "last_run"),
        (
            lambda value: Prediction(model_id=1, target_time=value, predicted_value=1.0),
            "target_time",
        ),
        (
            lambda value: MLModel(
                name="m",
                target="pm25_h1",
                features_used=[],
                artifact_path="/tmp/m.pkl",
                created_at=value,
            ),
            "created_at",
        ),
    ],
)
def test_every_timestamp_column_rejects_naive_values(factory, field: str) -> None:
    """DR-2 applies to the whole schema, not just to observations."""
    with pytest.raises(ValueError, match="timezone-aware"):
        factory(datetime(2026, 1, 1, 12))


def test_null_timestamps_stay_null() -> None:
    """last_run is legitimately empty for a source that has never run."""
    assert DataSource(name="s", last_run=None).last_run is None
