"""The training pipeline end to end (tasks 7.13, 7.14; AC-7, AC-8).

The two tests this phase exists to pass live here, and neither is allowed to
depend on a database being reachable -- an acceptance criterion that skips is an
acceptance criterion that is not checked.

``train_horizon`` needs a session only to register, so with ``register=False``
the whole pipeline -- split, embargo, selection, four models, cross-validation,
scoring -- runs offline. Registration itself is covered by the ``db``-marked
tests at the end.

**The data is the demo generator's, not a convenient fiction.** Persistence is
a *strong* forecast for hourly PM2.5 precisely because the series is
autocorrelated, and beating it on white noise would prove nothing. ``demo_data``
carries an AR(1) residual for exactly this reason (task 1.9), so AC-7 is being
tested against the hard case.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from core.config import Settings
from core.exceptions import InsufficientDataError
from services import demo_data, geo_service
from services.datasets import prepare_frame
from services.features import service as feature_service
from services.features.transformer import FeatureTransformer
from services.ml import models as model_zoo
from services.ml import registry, splitting, targets, training

HORIZON = 1
STATIONS = 2
DAYS = 60


@pytest.fixture(scope="module")
def observations() -> pd.DataFrame:
    """Two stations of hourly demo history, with its AR(1) structure intact."""
    settings = Settings(_env_file=None, postgres_password="test-only")
    stations = geo_service.stations_from_districts(settings)[:STATIONS]
    records = list(
        demo_data.generate_observations(days=DAYS, stations=stations, settings=settings)
    )
    return prepare_frame(
        pd.DataFrame(
            [
                {
                    "id": index + 1,
                    "source_id": 1,
                    "timestamp": record.timestamp,
                    "lat": record.lat,
                    "lon": record.lon,
                    "pm25": record.pm25,
                    "pm10": record.pm10,
                    "temp": record.temp,
                    "humidity": record.humidity,
                    "traffic_score": record.traffic_score,
                    "is_anomaly": False,
                }
                for index, record in enumerate(records)
            ]
        )
    )


@pytest.fixture(scope="module")
def trained(observations: pd.DataFrame) -> training.HorizonReport:
    """One horizon trained the way ``run`` trains it, without a database."""
    cutoff = training.training_cutoff(
        observations, test_fraction=0.2, max_horizon=HORIZON
    )
    train_only = observations[observations["timestamp"] < cutoff]
    repaired, _ = feature_service.prepare(train_only)
    transformer = FeatureTransformer.fit(repaired)

    featured, matrix, _ = training.prepare_matrix(
        observations, transformer, horizons=(HORIZON,)
    )

    return training.train_horizon(
        None,  # unused when register=False
        featured,
        matrix,
        transformer,
        horizon=HORIZON,
        test_fraction=0.2,
        cv_splits=3,
        ladder=model_zoo.build_ladder(),
        include_classical=False,
        include_prophet=False,
        register=False,
        settings=Settings(_env_file=None, postgres_password="test-only"),
    )


# --- Task 7.13: the leakage audit -------------------------------------------


def test_every_training_timestamp_precedes_every_test_timestamp(
    trained: training.HorizonReport,
) -> None:
    """AC-8's first half, asserted on the run that actually produced the
    metrics rather than on a splitter in isolation."""
    audit = trained.leakage_audit

    assert audit["train_precedes_test"]
    assert audit["no_overlapping_rows"]
    assert audit["no_shared_timestamps"]
    assert audit["train_end"] < audit["test_start"]


def test_the_embargo_holds_on_the_real_run(trained: training.HorizonReport) -> None:
    """A training row's target must not land inside the test period."""
    audit = trained.leakage_audit

    assert audit["embargo_respected"]
    assert audit["gap_hours"] >= HORIZON


def test_the_scaler_never_saw_test_data(trained: training.HorizonReport) -> None:
    """AC-8's second half. Checked against the fitted object: its mean equals
    the training mean and differs from the full set's."""
    scaled = [model for model in trained.models if model.scaler_audit]
    assert scaled, "no model in the ladder reported a scaler to audit"

    for model in scaled:
        assert model.scaler_audit["fitted_on_train_only"]
        assert model.scaler_audit["differs_from_full_set_mean"]
        assert model.scaler_audit["n_samples_seen"] == model.scaler_audit["train_rows"]


def test_feature_selection_is_fitted_on_training_rows_only(
    observations: pd.DataFrame, trained: training.HorizonReport
) -> None:
    """Choosing columns by how they behave on the test set is leakage wearing a
    respectable name."""
    from services.ml import selection

    transformer = FeatureTransformer.fit(observations.iloc[:100])
    featured, matrix, _ = training.prepare_matrix(
        observations, transformer, horizons=(HORIZON,)
    )
    usable = matrix.notna().all(axis=1) & featured[targets.target_name(HORIZON)].notna()
    frame = featured.loc[usable].reset_index(drop=True)
    model_matrix = matrix.loc[usable].reset_index(drop=True)
    split = splitting.chronological_split(
        frame, test_fraction=0.2, embargo_hours=HORIZON
    )

    on_train = selection.select(
        model_matrix.iloc[split.train_index],
        frame[targets.target_name(HORIZON)].iloc[split.train_index],
    )

    # What the run recorded is what selecting on the training rows alone gives.
    assert list(on_train.kept) == trained.selection["kept"]
    # And it is a real decision, not a pass-through that keeps everything.
    assert trained.selection["dropped_count"] > 0


def test_the_feature_transformer_is_fitted_before_the_split(
    observations: pd.DataFrame,
) -> None:
    """The Phase 5 log decision is data-dependent, so it is fitted on the
    training side of the cut -- not on everything and then applied."""
    cutoff = training.training_cutoff(
        observations, test_fraction=0.2, max_horizon=24
    )
    train_only = observations[observations["timestamp"] < cutoff]
    transformer = FeatureTransformer.fit(train_only)

    assert transformer.fit_window is not None
    assert transformer.fit_window.end < observations["timestamp"].max().to_pydatetime()
    assert cutoff < observations["timestamp"].max().to_pydatetime()


# --- Task 7.14: the baseline-beating test -----------------------------------


def test_xgboost_beats_the_naive_baseline_at_one_hour(
    trained: training.HorizonReport,
) -> None:
    """AC-7, on autocorrelated data where persistence is genuinely hard to beat."""
    baseline = trained.baseline
    production = trained.production

    assert baseline is not None and production is not None
    assert production.unavailable is None, production.unavailable
    assert production.metrics.mae < baseline.metrics.mae
    assert trained.beats_baseline


def test_the_baseline_is_a_serious_forecast_not_a_straw_man(
    trained: training.HorizonReport,
) -> None:
    """If persistence scored terribly, beating it would prove nothing. At one
    hour it explains most of the variance, which is the point of AC-7."""
    baseline = trained.baseline

    assert baseline is not None
    assert baseline.metrics.r2 > 0.5


def test_every_model_reports_its_skill_against_the_baseline(
    trained: training.HorizonReport,
) -> None:
    for model in trained.models:
        if model.unavailable:
            continue
        assert model.skill_vs_baseline is not None
    assert trained.baseline.skill_vs_baseline == pytest.approx(0.0)


# --- The rest of the pipeline -----------------------------------------------


def test_the_whole_ladder_is_trained_and_scored(
    trained: training.HorizonReport,
) -> None:
    names = [model.name for model in trained.models]

    assert names == [spec.name for spec in model_zoo.build_ladder()]
    for model in trained.models:
        if model.unavailable:
            continue
        assert model.metrics.rows > 0
        assert model.metrics.mae > 0


def test_cross_validation_ran_on_the_training_portion(
    trained: training.HorizonReport,
) -> None:
    for model in trained.models:
        if model.unavailable:
            continue
        assert model.cv["folds"] == 3
        assert model.cv["mae_mean"] is not None
        assert len(model.cv["mae_per_fold"]) == 3


def test_rows_without_a_target_are_dropped_and_counted(
    trained: training.HorizonReport,
) -> None:
    """The warm-up hours and the tail of each station's series. Counted rather
    than silently absent."""
    assert trained.rows_dropped > 0
    assert trained.rows_modelled > trained.rows_dropped


def test_the_report_serialises_for_the_api(trained: training.HorizonReport) -> None:
    payload = trained.as_dict()

    assert payload["target"] == "pm25_h1"
    assert payload["beats_baseline"]
    assert payload["leakage_audit"]["train_precedes_test"]
    assert payload["models"][0]["is_baseline"]


def test_a_window_too_short_to_train_is_refused(observations: pd.DataFrame) -> None:
    transformer = FeatureTransformer.fit(observations)
    featured, matrix, _ = training.prepare_matrix(
        observations.iloc[:80], transformer, horizons=(HORIZON,)
    )

    with pytest.raises(InsufficientDataError):
        training.train_horizon(
            None,
            featured,
            matrix,
            transformer,
            horizon=HORIZON,
            test_fraction=0.2,
            cv_splits=3,
            ladder=model_zoo.build_ladder(),
            include_classical=False,
            include_prophet=False,
            register=False,
            settings=Settings(_env_file=None, postgres_password="test-only"),
        )


# --- Registration, against a real database ----------------------------------


@pytest.fixture
def seeded(db_session: Session) -> int:
    from db.models import DataSource, Observation, SourceStatus

    settings = Settings(_env_file=None, postgres_password="test-only")
    stations = geo_service.stations_from_districts(settings)[:1]
    source = DataSource(name="ml-train-test", status=SourceStatus.OFFLINE)
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
    return source.id


@pytest.mark.db
def test_a_run_registers_every_model_with_its_metrics(
    db_session: Session, db_settings: Settings, seeded: int, tmp_path
) -> None:
    """Task 7.12: the artifact and the row are written together, so a registry
    row never points at a file that does not exist."""
    settings = db_settings.model_copy(update={"model_artifact_dir": str(tmp_path)})

    report = training.run(
        db_session,
        settings,
        source_ids=(seeded,),
        horizons=(1,),
        include_classical=False,
        register=True,
    )

    registered = [model.registered for model in report.horizons[0].models]
    assert all(entry is not None for entry in registered)

    for entry in registered:
        assert entry.mae is not None
        assert Path(entry.artifact_path).exists()

    rows = registry.list_models(db_session, target="pm25_h1")
    assert len(rows) == len(registered)


@pytest.mark.db
def test_the_artifact_carries_its_feature_contract(
    db_session: Session, db_settings: Settings, seeded: int, tmp_path
) -> None:
    """A model file and a feature spec that have drifted apart produce
    confident nonsense, so serving compares the fingerprint before predicting."""
    settings = db_settings.model_copy(update={"model_artifact_dir": str(tmp_path)})

    report = training.run(
        db_session,
        settings,
        source_ids=(seeded,),
        horizons=(1,),
        model_names=("ridge",),
        include_classical=False,
        register=True,
    )
    entry = next(
        model.registered
        for model in report.horizons[0].models
        if model.name == "ridge"
    )
    artifact = registry.load_artifact(entry.artifact_path)

    assert artifact["feature_fingerprint"] == report.transformer.fingerprint
    assert artifact["horizon_hours"] == 1
    assert artifact["target"] == "pm25_h1"
    assert artifact["columns"]
    assert artifact["estimator"] is not None


@pytest.mark.db
def test_serving_resolves_the_production_model_by_name(
    db_session: Session, db_settings: Settings, seeded: int, tmp_path
) -> None:
    """The registry holds the whole ladder so the Model Lab can compare it, so
    "newest row" alone would make the served model an accident of order."""
    settings = db_settings.model_copy(update={"model_artifact_dir": str(tmp_path)})

    training.run(
        db_session,
        settings,
        source_ids=(seeded,),
        horizons=(1,),
        include_classical=False,
        register=True,
    )

    chosen = registry.latest_for_target(
        db_session, "pm25_h1", name=model_zoo.PRODUCTION_MODEL
    )

    assert chosen is not None
    assert chosen.name == model_zoo.PRODUCTION_MODEL


@pytest.mark.db
def test_a_missing_artifact_is_a_domain_error(tmp_path) -> None:
    from core.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        registry.load_artifact(tmp_path / "nothing.joblib")


def test_the_artifact_filename_cannot_escape_its_directory() -> None:
    """A model name reaches the filesystem, and a path separator inside one
    would write outside the artifact directory."""
    from datetime import datetime, timezone

    name = registry.artifact_filename(
        "../../etc/passwd", "pm25_h1", datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    assert "/" not in name
    assert "\\" not in name
    assert name.endswith(registry.ARTIFACT_SUFFIX)
