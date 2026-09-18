"""The cleaning pipeline end to end (tasks 3.7, 3.8; AC-4, AC-5).

Runs against the real schema inside a rolled-back transaction, because the two
things worth proving here -- that ``is_anomaly`` is written and that no row is
removed -- are claims about the database, not about a DataFrame.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import Settings
from db.models import DataSource, Observation, SourceStatus
from services import datasets
from services.quality import pipeline

pytestmark = pytest.mark.db

START = datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.fixture
def seeded(db_session: Session) -> DataSource:
    """A source with a known series: a short gap, a long gap, and a spike."""
    source = DataSource(
        name=f"quality-test-{datetime.now(timezone.utc).timestamp()}",
        status=SourceStatus.HEALTHY,
    )
    db_session.add(source)
    db_session.flush()

    for hour in range(300):
        moment = START + timedelta(hours=hour)
        pm25: float | None = 50.0 + (hour % 24) * 0.8
        pm10: float | None = 110.0 + (hour % 24) * 1.4
        temp: float | None = 20.0 + (hour % 12) * 0.5
        humidity: float | None = 55.0 + (hour % 10)
        traffic: float | None = 40.0 + (hour % 18)

        if 20 <= hour <= 21:  # short gap -> forward/backward fill
            pm25 = None
        if 100 <= hour <= 129:  # long gap -> MICE
            temp = None
        if hour == 200:  # spike -> flagged, never deleted
            pm25, pm10 = 1500.0, 2800.0

        db_session.add(
            Observation(
                source_id=source.id,
                timestamp=moment,
                lat=28.61,
                lon=77.21,
                pm25=pm25,
                pm10=pm10,
                temp=temp,
                humidity=humidity,
                traffic_score=traffic,
            )
        )

    db_session.flush()
    return source


def _rows(db_session: Session, source: DataSource) -> int:
    return int(
        db_session.scalar(
            select(func.count())
            .select_from(Observation)
            .where(Observation.source_id == source.id)
        )
        or 0
    )


# --- AC-4 and AC-5 ----------------------------------------------------------


def test_the_pipeline_never_removes_a_row(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """AC-5. An outlier could be a valid pollution spike -- the most
    informative record in the dataset -- so nothing is deleted."""
    before = _rows(db_session, seeded)

    report = pipeline.run(
        db_session, db_settings, source_ids=(seeded.id,), persist=False
    )
    db_session.flush()

    assert report.rows_preserved
    assert report.rows_in == report.rows_out == before
    assert _rows(db_session, seeded) == before


def test_the_feature_set_has_no_nulls_afterwards(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """AC-4."""
    report = pipeline.run(
        db_session, db_settings, source_ids=(seeded.id,), persist=False
    )

    assert report.feature_set_is_complete
    assert report.imputation.remaining_nulls == {
        column: 0 for column in datasets.MEASUREMENT_COLUMNS
    }


def test_the_anomalous_row_is_retained_and_flagged(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """AC-5 in the database: the spike is still there, and now marked."""
    pipeline.run(db_session, db_settings, source_ids=(seeded.id,), persist=False)
    db_session.flush()

    spike = db_session.scalar(
        select(Observation)
        .where(Observation.source_id == seeded.id)
        .where(Observation.timestamp == START + timedelta(hours=200))
    )

    assert spike is not None
    assert spike.pm25 == 1500.0  # value untouched
    assert spike.is_anomaly is True


def test_short_and_long_gaps_are_repaired_by_different_stages(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    report = pipeline.run(
        db_session, db_settings, source_ids=(seeded.id,), persist=False
    )

    assert report.imputation.filled_short_gaps["pm25"] == 2
    assert report.imputation.imputed_by_mice["temp"] == 30


def test_imputation_does_not_write_values_back_to_observations(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """``observations`` holds what was measured. The imputed frame is a derived
    artefact and belongs in data/processed, not in the source of truth."""
    pipeline.run(db_session, db_settings, source_ids=(seeded.id,), persist=False)
    db_session.flush()

    still_null = db_session.scalar(
        select(func.count())
        .select_from(Observation)
        .where(Observation.source_id == seeded.id)
        .where(Observation.pm25.is_(None))
    )

    assert still_null == 2


# --- Write-back (task 3.7) --------------------------------------------------


def test_flags_can_be_suppressed(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    report = pipeline.run(
        db_session,
        db_settings,
        source_ids=(seeded.id,),
        persist=False,
        write_flags=False,
    )

    assert report.flags_written == 0
    assert report.anomalies.flagged >= 1  # detected, just not persisted


def test_a_second_run_writes_nothing_new(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """Deterministic detectors on unchanged data should be a no-op."""
    pipeline.run(db_session, db_settings, source_ids=(seeded.id,), persist=False)
    db_session.flush()

    second = pipeline.run(
        db_session, db_settings, source_ids=(seeded.id,), persist=False
    )

    assert second.flags_written == 0


def test_a_row_that_stops_qualifying_is_unflagged(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """The column reflects the current detectors, not the union of every run
    ever made. Unflagging changes a judgement, never the data."""
    ordinary = db_session.scalar(
        select(Observation)
        .where(Observation.source_id == seeded.id)
        .where(Observation.timestamp == START + timedelta(hours=50))
    )
    assert ordinary is not None
    ordinary.is_anomaly = True  # a stale flag from some earlier judgement
    db_session.flush()

    pipeline.run(db_session, db_settings, source_ids=(seeded.id,), persist=False)
    db_session.flush()
    db_session.expire_all()

    refreshed = db_session.get(Observation, ordinary.id)
    assert refreshed is not None
    assert refreshed.is_anomaly is False
    assert refreshed.pm25 is not None  # the row itself is untouched
    assert _rows(db_session, seeded) == 300


def test_the_quality_engine_only_writes_is_anomaly(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    """Phase 2 owns the measurement columns; this engine owns exactly one
    column, so the two write paths never fight."""
    before = {
        row.id: (row.pm25, row.pm10, row.temp, row.humidity, row.traffic_score)
        for row in db_session.scalars(
            select(Observation).where(Observation.source_id == seeded.id)
        )
    }

    pipeline.run(db_session, db_settings, source_ids=(seeded.id,), persist=False)
    db_session.flush()
    db_session.expire_all()

    after = {
        row.id: (row.pm25, row.pm10, row.temp, row.humidity, row.traffic_score)
        for row in db_session.scalars(
            select(Observation).where(Observation.source_id == seeded.id)
        )
    }

    assert before == after


# --- Persistence (task 3.8) -------------------------------------------------


def test_the_cleaned_frame_and_report_are_written(
    db_session: Session, db_settings: Settings, seeded: DataSource, tmp_path
) -> None:
    settings = db_settings.model_copy(update={"data_processed_dir": str(tmp_path)})

    report = pipeline.run(
        db_session, settings, source_ids=(seeded.id,), write_flags=False
    )

    cleaned = tmp_path / pipeline.CLEANED_FILENAME
    written_report = tmp_path / pipeline.REPORT_FILENAME

    assert cleaned.is_file()
    assert written_report.is_file()
    assert report.cleaned_path == str(cleaned)


def test_the_persisted_frame_has_one_row_per_observation(
    db_session: Session, db_settings: Settings, seeded: DataSource, tmp_path
) -> None:
    import pandas as pd

    settings = db_settings.model_copy(update={"data_processed_dir": str(tmp_path)})
    pipeline.run(db_session, settings, source_ids=(seeded.id,), write_flags=False)

    frame = pd.read_csv(tmp_path / pipeline.CLEANED_FILENAME)

    assert len(frame) == 300
    assert frame[list(datasets.MEASUREMENT_COLUMNS)].isna().to_numpy().sum() == 0


def test_the_report_sidecar_explains_the_run(
    db_session: Session, db_settings: Settings, seeded: DataSource, tmp_path
) -> None:
    """A cleaned file separated from the account of how it was produced is an
    unexplained number, which is what this project refuses to ship."""
    settings = db_settings.model_copy(update={"data_processed_dir": str(tmp_path)})
    pipeline.run(db_session, settings, source_ids=(seeded.id,), write_flags=False)

    payload = json.loads((tmp_path / pipeline.REPORT_FILENAME).read_text(encoding="utf-8"))

    assert payload["rows_preserved"] is True
    assert payload["feature_set_is_complete"] is True
    assert payload["missingness"]["columns"]
    assert payload["anomalies"]["min_votes"] == 2


def test_persistence_can_be_skipped(
    db_session: Session, db_settings: Settings, seeded: DataSource
) -> None:
    report = pipeline.run(
        db_session, db_settings, source_ids=(seeded.id,), persist=False
    )

    assert report.cleaned_path is None
    assert report.report_path is None


def test_a_window_with_no_data_is_handled(
    db_session: Session, db_settings: Settings
) -> None:
    report = pipeline.run(
        db_session,
        db_settings,
        start=datetime(1990, 1, 1, tzinfo=timezone.utc),
        end=datetime(1990, 1, 2, tzinfo=timezone.utc),
        persist=False,
    )

    assert report.rows_in == 0
    assert report.rows_preserved
