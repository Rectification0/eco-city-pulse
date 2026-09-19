"""Feature engineering entry points (tasks 5.5, 5.6).

The offline tests pin the shared preparation step, which is where the two paths
could most easily drift apart. The ``db``-marked tests run the whole thing
against a real ``observations`` table and check the same property end to end:
what the training build says about an hour is what a serving request says about
it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from core.config import Settings
from core.exceptions import InsufficientDataError
from services import datasets
from services.features import service, store
from services.features.spec import DEFAULT_SPEC, FeatureSpec, RollingSpec
from services.features.transformer import FeatureTransformer, feature_matrix
from tests.test_features_transformer import realistic_frame
from tests.test_features_windows import START, build_frame

# --- The shared preparation step --------------------------------------------


def test_a_short_gap_is_repaired_before_features_are_built() -> None:
    frame = build_frame(rows=60)
    frame.loc[20:21, "pm25"] = np.nan

    repaired, filled = service.prepare(frame, max_gap_hours=3)

    assert filled["pm25"] == 2
    assert repaired["pm25"].iloc[20] == pytest.approx(19.0)  # carried forward


def test_preparation_never_reads_backward() -> None:
    """A backward fill takes a value from the future. Fine when cleaning a
    historical record (Phase 3); a leak in a feature path (AC-8)."""
    frame = build_frame(rows=60)
    frame.loc[0, "pm25"] = np.nan  # nothing earlier to carry forward

    repaired, _ = service.prepare(frame)

    assert np.isnan(repaired["pm25"].iloc[0])


def test_a_long_outage_is_left_as_nan_rather_than_invented() -> None:
    frame = build_frame(rows=60)
    frame.loc[20:30, "pm25"] = np.nan

    repaired, filled = service.prepare(frame, max_gap_hours=3)
    featured = FeatureTransformer().transform(repaired)

    assert filled["pm25"] == 0
    assert featured["pm25_lag_1h"].iloc[25] != featured["pm25_lag_1h"].iloc[25]  # NaN


def test_build_is_preparation_then_transform() -> None:
    frame = realistic_frame(rows=100)
    transformer = FeatureTransformer.fit(frame)

    built, _ = service.build(frame, transformer)
    repaired, _ = service.prepare(frame)

    pd.testing.assert_frame_equal(
        built, transformer.transform(repaired), check_exact=True
    )


# --- The inference window ---------------------------------------------------


def test_the_history_window_covers_the_longest_lookback_and_the_repair() -> None:
    """Too short a window and the longest lag comes back NaN at serving time
    while it was present at training time -- the drift this phase rules out."""
    at = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)

    start = service.history_start(at, DEFAULT_SPEC, max_gap_hours=3)

    assert at - start == timedelta(hours=DEFAULT_SPEC.history_hours + 3 + 1)


def test_the_window_widens_with_the_spec() -> None:
    """The window is derived from the spec, never a fixed constant: a weekly
    rolling feature needs a week of history and must get one."""
    at = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
    weekly = FeatureSpec(lags=(), rollings=(RollingSpec("pm25", 168, "mean"),))

    start = service.history_start(at, weekly, max_gap_hours=3)

    assert at - start == timedelta(hours=168 + 3 + 1)


# --- End to end, against a real database ------------------------------------


@pytest.fixture
def seeded(db_session: Session) -> tuple[Session, int]:
    """A station's worth of hourly observations, rolled back afterwards."""
    from db.models import DataSource, Observation, SourceStatus

    source = DataSource(
        name="feature-test", api_url="offline://test", status=SourceStatus.OFFLINE
    )
    db_session.add(source)
    db_session.flush()

    rng = np.random.default_rng(19)
    db_session.add_all(
        [
            Observation(
                source_id=source.id,
                timestamp=START + timedelta(hours=hour),
                lat=28.61,
                lon=77.21,
                pm25=float(rng.lognormal(4.0, 0.7)),
                pm10=float(rng.lognormal(4.7, 0.6)),
                temp=float(rng.normal(22, 4)),
                humidity=float(rng.uniform(30, 80)),
                traffic_score=float(rng.uniform(20, 80)),
            )
            for hour in range(240)
        ]
    )
    db_session.flush()
    return db_session, source.id


@pytest.mark.db
def test_a_build_reaches_the_feature_store(
    seeded: tuple[Session, int], db_settings: Settings, tmp_path
) -> None:
    session, source_id = seeded
    settings = db_settings.model_copy(update={"data_processed_dir": str(tmp_path)})

    result = service.build_features(
        session, settings, source_ids=(source_id,), persist=True
    )

    assert result.rows == 240
    assert result.complete_rows == 216  # less the 24-hour warm-up
    assert result.stored is not None
    restored, transformer = store.read(settings)
    assert len(restored) == 240
    assert transformer.fingerprint == result.transformer.fingerprint


@pytest.mark.db
def test_the_serving_path_agrees_with_the_training_build(
    seeded: tuple[Session, int], db_settings: Settings, tmp_path
) -> None:
    """Phase 5's exit criterion, end to end through the database.

    The training build reads 240 hours. ``features_at`` reads only the window
    one hour needs. Both must describe that hour identically -- this is the
    property Phase 9 relies on when it serves a model trained in Phase 7.
    """
    session, source_id = seeded
    settings = db_settings.model_copy(update={"data_processed_dir": str(tmp_path)})

    result = service.build_features(
        session, settings, source_ids=(source_id,), persist=False
    )
    at = START + timedelta(hours=200)
    names = list(result.transformer.feature_names)

    trained = result.frame[result.frame["timestamp"] == at][names]
    served = service.features_at(
        session,
        result.transformer,
        at=at,
        lat=28.61,
        lon=77.21,
        source_ids=(source_id,),
    )

    pd.testing.assert_frame_equal(
        trained.reset_index(drop=True),
        served[names].reset_index(drop=True),
        check_exact=True,
    )


@pytest.mark.db
def test_a_served_row_is_a_float_matrix_not_an_object_row(
    seeded: tuple[Session, int], db_settings: Settings
) -> None:
    """A Series of a mixed row collapses to ``object``, and an object array is
    what a model rejects at predict time (task 9.2)."""
    session, source_id = seeded
    transformer = FeatureTransformer()

    served = service.features_at(
        session,
        transformer,
        at=START + timedelta(hours=200),
        lat=28.61,
        lon=77.21,
        source_ids=(source_id,),
    )
    matrix = feature_matrix(served, transformer.spec)

    assert len(matrix) == 1
    assert matrix.to_numpy().dtype == np.float64


@pytest.mark.db
def test_serving_an_hour_with_no_observation_is_a_clear_error(
    seeded: tuple[Session, int], db_settings: Settings
) -> None:
    session, source_id = seeded

    with pytest.raises(InsufficientDataError):
        service.features_at(
            session,
            FeatureTransformer(),
            at=START + timedelta(hours=5000),
            lat=28.61,
            lon=77.21,
            source_ids=(source_id,),
        )


@pytest.mark.db
def test_serving_reads_only_the_window_it_needs(
    seeded: tuple[Session, int], db_settings: Settings
) -> None:
    """A bounded query, not a table scan: the spec says how far back one row
    reaches, and the fetch honours it."""
    session, source_id = seeded
    transformer = FeatureTransformer()
    at = START + timedelta(hours=200)

    window = datasets.load_observations(
        session,
        source_ids=(source_id,),
        start=service.history_start(at, transformer.spec),
        end=at,
    )

    assert len(window) <= transformer.spec.history_hours + 5
