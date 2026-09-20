"""Loading a registered model and replaying its features (Phase 8, feeds Phase 9).

The failure this module exists to prevent is silent: a model file and a feature
spec that have drifted apart do not crash, they produce confident nonsense. So
the refusal paths are tested as carefully as the happy one, and the offline
tests cover them without needing a database — a guard that only runs where
PostgreSQL is reachable is a guard that mostly does not run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from core.config import Settings
from core.exceptions import ModelNotFoundError, ModelNotTrainedError
from db.models import MLModel
from services import demo_data, geo_service
from services.features.spec import DEFAULT_SPEC
from services.features.transformer import FeatureTransformer
from services.ml import models as model_zoo
from services.ml import registry, serving, training


def artifact_for(spec=DEFAULT_SPEC, *, fingerprint: str | None = None) -> dict:
    transformer = FeatureTransformer(spec=spec)
    return {
        "estimator": model_zoo.NaiveLag1(),
        "model_name": "naive_lag1",
        "target": "pm25_h1",
        "horizon_hours": 1,
        "columns": ["pm25"],
        "feature_spec": spec.as_dict(),
        "feature_fingerprint": fingerprint or transformer.fingerprint,
        "metrics": {"mae": 1.0},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


# --- The contract check (offline) -------------------------------------------


def test_a_model_whose_spec_no_longer_matches_its_fingerprint_is_refused(
    settings: Settings, tmp_path
) -> None:
    """The silent failure made loud. A stored spec that does not hash to the
    fingerprint recorded beside it has drifted, and replaying it would mean
    feeding the model something other than what it learned."""
    created = datetime.now(timezone.utc)
    path = registry.save_artifact(
        artifact_for(fingerprint="deadbeefdeadbeef"),
        name="naive_lag1",
        target="pm25_h1",
        created_at=created,
        settings=settings.model_copy(update={"model_artifact_dir": str(tmp_path)}),
    )
    row = MLModel(
        id=1,
        name="naive_lag1",
        target="pm25_h1",
        features_used=["pm25"],
        artifact_path=str(path),
        created_at=created,
    )

    with pytest.raises(ModelNotFoundError, match="does not match the fingerprint"):
        serving._checked(row)


def test_a_matching_artifact_loads_and_rebuilds_its_transformer(
    settings: Settings, tmp_path
) -> None:
    created = datetime.now(timezone.utc)
    path = registry.save_artifact(
        artifact_for(),
        name="naive_lag1",
        target="pm25_h1",
        created_at=created,
        settings=settings.model_copy(update={"model_artifact_dir": str(tmp_path)}),
    )
    row = MLModel(
        id=1,
        name="naive_lag1",
        target="pm25_h1",
        features_used=["pm25"],
        artifact_path=str(path),
        created_at=created,
    )

    loaded = serving._checked(row)

    assert loaded.spec == DEFAULT_SPEC
    assert loaded.horizon == 1
    assert loaded.columns == ("pm25",)
    # The transformer is rebuilt from the frozen spec, log decision included.
    assert loaded.transformer.spec.log_columns == DEFAULT_SPEC.log_columns


def test_the_loaded_summary_is_what_the_model_lab_shows(
    settings: Settings, tmp_path
) -> None:
    created = datetime.now(timezone.utc)
    path = registry.save_artifact(
        artifact_for(),
        name="naive_lag1",
        target="pm25_h1",
        created_at=created,
        settings=settings.model_copy(update={"model_artifact_dir": str(tmp_path)}),
    )
    row = MLModel(
        id=7,
        name="naive_lag1",
        target="pm25_h1",
        features_used=["pm25"],
        artifact_path=str(path),
        created_at=created,
    )

    payload = serving._checked(row).as_dict()

    assert payload["model_id"] == 7
    assert payload["horizon_hours"] == 1
    assert payload["feature_fingerprint"]
    assert payload["metrics"]["mae"] == 1.0


# --- Against a real database ------------------------------------------------


@pytest.fixture
def trained(db_session: Session, db_settings: Settings, tmp_path) -> Settings:
    """A registered ladder over one station, rolled back afterwards."""
    from db.models import DataSource, Observation, SourceStatus

    settings = db_settings.model_copy(update={"model_artifact_dir": str(tmp_path)})
    stations = geo_service.stations_from_districts(settings)[:1]

    source = DataSource(name="serving-test", status=SourceStatus.OFFLINE)
    db_session.add(source)
    db_session.flush()
    db_session.add_all(
        [
            Observation(
                source_id=source.id,
                timestamp=record.timestamp,
                lat=record.lat,
                lon=record.lon,
                pm25=record.pm25,
                pm10=record.pm10,
                temp=record.temp,
                humidity=record.humidity,
                traffic_score=record.traffic_score,
            )
            for record in demo_data.generate_observations(
                days=45, stations=stations, settings=settings
            )
        ]
    )
    db_session.flush()

    training.run(
        db_session,
        settings,
        source_ids=(source.id,),
        horizons=(1,),
        include_classical=False,
        register=True,
    )
    return settings


@pytest.mark.db
def test_serving_picks_the_production_model_not_the_newest_row(
    db_session: Session, trained: Settings
) -> None:
    """A run registers the whole ladder, so "newest" alone would make the
    served model an accident of iteration order."""
    loaded = serving.load_latest(db_session, "pm25_h1")

    assert loaded.row.name == model_zoo.PRODUCTION_MODEL


@pytest.mark.db
def test_a_target_with_no_model_is_a_clear_error(db_session: Session) -> None:
    with pytest.raises(ModelNotTrainedError):
        serving.load_latest(db_session, "pm25_h999")


@pytest.mark.db
def test_the_replayed_matrix_carries_the_artifact_columns_in_order(
    db_session: Session, trained: Settings
) -> None:
    loaded = serving.load_latest(db_session, "pm25_h1")

    featured, matrix = serving.build_matrix(db_session, loaded)

    assert list(matrix.columns) == list(loaded.columns)
    assert len(featured) == len(matrix)
    assert matrix.notna().all().all()


@pytest.mark.db
def test_a_prediction_arrives_with_its_reasoning(
    db_session: Session, trained: Settings
) -> None:
    """Task 8.3, end to end: the features come from the Phase 5 inference path
    and the attribution adds up to the forecast."""
    from db.models import DataSource, Observation

    loaded = serving.load_latest(db_session, "pm25_h1")
    # Scoped to the fixture's own source. Unfiltered, this asked for the newest
    # row *in the database*, which is the fixture's only while nothing else has
    # written more recently -- a real ingest puts live readings on top, at
    # coordinates carrying a single hour of history, and feature construction
    # for a station with one reading fails on a fixture that had nothing to do
    # with it.
    latest = (
        db_session.query(Observation)
        .join(DataSource, DataSource.id == Observation.source_id)
        .filter(DataSource.name == "serving-test")
        .order_by(Observation.timestamp.desc())
        .first()
    )
    at = latest.timestamp - timedelta(hours=2)

    prediction, attribution = serving.explain_at(
        db_session, loaded, at=at, lat=latest.lat, lon=latest.lon
    )

    assert attribution.is_additive
    assert attribution.prediction == pytest.approx(prediction)
    top = attribution.top_features(3)
    assert len(top) == 3
    assert all(isinstance(value, float) for value in top.values())


@pytest.mark.db
def test_global_importance_is_cached_on_the_dataset_fingerprint(
    db_session: Session, trained: Settings
) -> None:
    from services.eda import cache

    cache.PROFILE_CACHE.clear()
    loaded = serving.load_latest(db_session, "pm25_h1")

    _, first, cached_first = serving.global_importance(
        db_session, trained, model_id=loaded.row.id, sample_rows=100
    )
    _, second, cached_second = serving.global_importance(
        db_session, trained, model_id=loaded.row.id, sample_rows=100
    )

    assert not cached_first
    assert cached_second
    assert first.features[0].feature == second.features[0].feature


@pytest.mark.db
def test_the_explained_model_ranks_pm25_first(
    db_session: Session, trained: Settings
) -> None:
    """On hourly PM2.5 the current reading dominates any one-hour forecast; a
    summary that said otherwise would be a signal something is wrong."""
    loaded = serving.load_latest(db_session, "pm25_h1")

    _, importance, _ = serving.global_importance(
        db_session, trained, model_id=loaded.row.id, sample_rows=200, use_cache=False
    )

    assert importance.features[0].feature == "pm25"
    assert isinstance(importance.base_value, float)
    assert importance.method.startswith("shap_tree:")


@pytest.mark.db
def test_explaining_a_model_that_is_not_registered_is_a_404(
    db_session: Session, trained: Settings
) -> None:
    with pytest.raises(ModelNotFoundError):
        serving.load(db_session, 99_999)
