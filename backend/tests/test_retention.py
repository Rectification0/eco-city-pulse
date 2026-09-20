"""Retention — bounding growth without losing the record.

The danger in a cleanup job is not that it fails to delete. It is that it
deletes something nobody notices until it is needed: the production model's
weights, the drift dataset, a measurement. Each test here names the thing that
must survive, because "it freed some space" is not the property worth asserting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import Settings
from db.models import (
    DataSource,
    IngestionRun,
    MLModel,
    Observation,
    Prediction,
    QuarantinedRecord,
    RunStatus,
)
from services import retention

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        postgres_password="test-only",
        model_artifact_dir=str(tmp_path),
        **overrides,
    )


def _model(
    session: Session, tmp_path, *, name: str, target: str, age_hours: int
) -> MLModel:
    """A registry row and the file it points at."""
    created = NOW - timedelta(hours=age_hours)
    path = tmp_path / f"{target}__{name}__{created:%Y%m%dT%H%M%S}.joblib"
    path.write_bytes(b"x" * 1024)
    row = MLModel(
        name=name,
        target=target,
        features_used=["pm25"],
        artifact_path=str(path),
        created_at=created,
    )
    session.add(row)
    session.flush()
    return row


# --- Artifacts --------------------------------------------------------------


@pytest.mark.db
def test_the_production_model_is_never_swept(db_session: Session, tmp_path) -> None:
    """`serving.load_latest` resolves a target to its newest row of any name.

    Deleting that file would break inference while leaving every row in place,
    so the failure would surface as a 404 from the forecast endpoint rather
    than as anything pointing at retention.
    """
    settings = _settings(tmp_path, artifact_keep_per_model=1)
    newest = _model(db_session, tmp_path, name="xgboost", target="pm25_h1", age_hours=1)
    _model(db_session, tmp_path, name="xgboost", target="pm25_h1", age_hours=99)

    retention.sweep_artifacts(db_session, settings)

    assert Path(newest.artifact_path).exists()


@pytest.mark.db
def test_a_superseded_artifact_goes_but_its_row_stays(
    db_session: Session, tmp_path
) -> None:
    """`predictions.model_id` is ON DELETE CASCADE, so deleting the row would
    take the drift dataset with it — and the metrics on the row are the record
    of what was trained, which costs nothing to keep."""
    settings = _settings(tmp_path, artifact_keep_per_model=1)
    old = _model(db_session, tmp_path, name="ridge", target="pm25_h1", age_hours=99)
    _model(db_session, tmp_path, name="ridge", target="pm25_h1", age_hours=1)

    result = retention.sweep_artifacts(db_session, settings)

    assert not Path(old.artifact_path).exists()
    assert result.removed == 1
    assert db_session.get(MLModel, old.id) is not None


@pytest.mark.db
def test_an_orphaned_artifact_is_removed(db_session: Session, tmp_path) -> None:
    """Every `db`-marked test that trains a ladder writes artifacts and then
    rolls its rows back, so files referenced by nothing are the normal case
    rather than a corruption."""
    settings = _settings(tmp_path)
    orphan = tmp_path / "pm25_h1__random_forest__20260101T000000.joblib"
    orphan.write_bytes(b"x" * 2048)

    retention.sweep_artifacts(db_session, settings)

    assert not orphan.exists()


@pytest.mark.db
def test_nothing_but_artifacts_is_touched(db_session: Session, tmp_path) -> None:
    """The directory is a mounted volume in the container, and a sweep that
    globbed everything would delete whatever else was mounted beside it."""
    settings = _settings(tmp_path)
    bystander = tmp_path / "notes.md"
    bystander.write_text("not an artifact")

    retention.sweep_artifacts(db_session, settings)

    assert bystander.exists()


@pytest.mark.db
def test_a_dry_run_reports_without_deleting(db_session: Session, tmp_path) -> None:
    """The habit the script recommends has to actually be safe."""
    settings = _settings(tmp_path)
    orphan = tmp_path / "pm25_h1__ridge__20260101T000000.joblib"
    orphan.write_bytes(b"x" * 512)

    result = retention.sweep_artifacts(db_session, settings, dry_run=True)

    assert result.removed == 1
    assert result.bytes_freed == 512
    assert orphan.exists()


@pytest.mark.db
def test_keeping_zero_still_keeps_one(db_session: Session, tmp_path) -> None:
    """A misconfigured `artifact_keep_per_model=0` must not empty the registry
    of every loadable model. The floor is the production model."""
    settings = _settings(tmp_path, artifact_keep_per_model=0)
    row = _model(db_session, tmp_path, name="xgboost", target="pm25_h1", age_hours=1)

    retention.sweep_artifacts(db_session, settings)

    assert Path(row.artifact_path).exists()


# --- Predictions ------------------------------------------------------------


@pytest.mark.db
def test_a_scored_prediction_is_kept_however_old(
    db_session: Session, tmp_path
) -> None:
    """Design §6.1: scored rows *are* the drift-monitoring dataset — the only
    record of how a deployed model behaved on data it never trained on. Age is
    what makes them valuable, so age must not be what removes them."""
    settings = _settings(tmp_path, unscored_prediction_retention_days=1)
    model = _model(db_session, tmp_path, name="xgboost", target="pm25_h1", age_hours=1)
    scored = Prediction(
        model_id=model.id,
        target_time=NOW - timedelta(days=900),
        predicted_value=50.0,
        actual_value=48.0,
        lat=28.61,
        lon=77.21,
    )
    db_session.add(scored)
    db_session.flush()

    retention.sweep_unscored_predictions(db_session, settings, now=NOW)

    assert db_session.get(Prediction, scored.id) is not None


@pytest.mark.db
def test_an_unscored_forecast_past_its_window_is_dropped(
    db_session: Session, tmp_path
) -> None:
    """It is not waiting for anything: the observation that would have scored
    it never arrived, and it is scanned by every future backfill."""
    settings = _settings(tmp_path, unscored_prediction_retention_days=30)
    model = _model(db_session, tmp_path, name="xgboost", target="pm25_h1", age_hours=1)
    stale = Prediction(
        model_id=model.id,
        target_time=NOW - timedelta(days=90),
        predicted_value=50.0,
        lat=28.61,
        lon=77.21,
    )
    recent = Prediction(
        model_id=model.id,
        target_time=NOW - timedelta(days=2),
        predicted_value=50.0,
        lat=28.62,
        lon=77.21,
    )
    db_session.add_all([stale, recent])
    db_session.flush()

    result = retention.sweep_unscored_predictions(db_session, settings, now=NOW)

    assert result.removed == 1
    assert db_session.get(Prediction, stale.id) is None
    assert db_session.get(Prediction, recent.id) is not None


# --- The run log ------------------------------------------------------------


@pytest.mark.db
def test_an_expired_run_takes_its_quarantined_payloads_with_it(
    db_session: Session, tmp_path
) -> None:
    """`quarantined_records` cascades from `ingestion_runs` on purpose: a
    rejected payload is evidence for the run that rejected it, and keeping it
    after the run is gone leaves a rejection nobody can trace back to anything.
    """
    settings = _settings(tmp_path, run_log_retention_days=90)
    source_id = db_session.scalar(select(DataSource.id).order_by(DataSource.id))

    old = IngestionRun(
        source_id=source_id,
        mode="manual",
        status=RunStatus.SUCCESS,
        started_at=NOW - timedelta(days=200),
    )
    recent = IngestionRun(
        source_id=source_id,
        mode="manual",
        status=RunStatus.SUCCESS,
        started_at=NOW - timedelta(days=2),
    )
    db_session.add_all([old, recent])
    db_session.flush()
    db_session.add(
        QuarantinedRecord(
            run_id=old.id, reason="bad payload", payload={"raw": "unparseable"}
        )
    )
    db_session.flush()

    result = retention.sweep_run_log(db_session, settings, now=NOW)

    assert result.removed >= 1
    assert db_session.get(IngestionRun, old.id) is None
    assert db_session.get(IngestionRun, recent.id) is not None
    assert db_session.query(QuarantinedRecord).filter_by(run_id=old.id).count() == 0


# --- The line retention does not cross --------------------------------------


@pytest.mark.db
def test_a_full_pass_never_removes_an_observation(
    db_session: Session, tmp_path
) -> None:
    """AC-4 and AC-5. The quality engine flags and never deletes; a retention
    job with database credentials is exactly the place that rule would be
    quietly broken, so it is asserted rather than assumed."""
    settings = _settings(tmp_path, run_log_retention_days=0)
    before = db_session.query(Observation).count()

    retention.run(db_session, settings, now=NOW)

    assert db_session.query(Observation).count() == before


@pytest.mark.db
def test_the_report_serialises_for_a_caller(db_session: Session, tmp_path) -> None:
    """Frozen dataclasses with `as_dict` are the convention for anything a
    route or a script prints."""
    import json

    report = retention.run(db_session, _settings(tmp_path), now=NOW, dry_run=True)
    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["dry_run"] is True
    assert {sweep["name"] for sweep in payload["sweeps"]} == {
        "artifacts",
        "ingestion_runs",
        "unscored_predictions",
    }
