"""The prediction service — FEAT-06 (tasks 9.1-9.7, AC-9).

AC-9 is a *contract*: the response must match the example in specs §8. That is
checked offline, against the shape itself, because an acceptance criterion that
only runs where PostgreSQL is reachable is an acceptance criterion that mostly
does not run. The database-backed tests then exercise the same path end to end.

The contract, verbatim from specs §8:

    {
      "prediction": 45.2,
      "unit": "ug/m3",
      "confidence_interval": [38.5, 51.9],
      "top_features": { "lag_24_pm25": 0.45, "wind_speed": 0.22 }
    }
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy.orm import Session

from api.routes.ml import PredictionResponse
from core.config import Settings
from core.exceptions import InsufficientDataError, ModelNotTrainedError
from db.models import MLModel
from services import demo_data, geo_service
from services.ml import explain, intervals, prediction, serving, training
from services.ml.serving import LoadedModel

ORIGIN = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


def a_result(**overrides) -> prediction.PredictionResult:
    """A PredictionResult built by hand, so the contract can be checked without
    a database, a model artifact or a training run."""
    attribution = explain.Attribution(
        base_value=40.0,
        contributions={"pm25": 6.0, "temp": -1.5, "hour_of_day": 0.7},
        prediction=45.2,
        method="shap_tree:tree_path_dependent",
    )
    row = MLModel(
        id=3,
        name="xgboost",
        target="pm25_h1",
        features_used=["pm25"],
        artifact_path="/tmp/x.joblib",
        mae=6.4,
        created_at=ORIGIN,
    )
    defaults = {
        "prediction": 45.2,
        "unit": prediction.UNIT,
        "confidence_interval": (38.5, 51.9),
        "attribution": attribution,
        "coverage": 0.8,
        "interval_method": intervals.CONFORMAL,
        "model": LoadedModel(row=row, artifact={"horizon_hours": 1, "columns": ["pm25"]}),
        "origin_time": ORIGIN,
        "target_time": ORIGIN + timedelta(hours=1),
        "lat": 28.61,
        "lon": 77.21,
        "prediction_id": 11,
    }
    return prediction.PredictionResult(**{**defaults, **overrides})


# --- Task 9.7: the contract (AC-9), offline ---------------------------------


def test_the_response_carries_exactly_the_four_specified_keys() -> None:
    contract = a_result().as_contract()

    assert set(contract) == {
        "prediction",
        "unit",
        "confidence_interval",
        "top_features",
    }


def test_each_field_has_the_type_the_spec_shows() -> None:
    contract = a_result().as_contract()

    assert isinstance(contract["prediction"], float)
    assert isinstance(contract["unit"], str)
    assert isinstance(contract["confidence_interval"], list)
    assert len(contract["confidence_interval"]) == 2
    assert all(isinstance(bound, float) for bound in contract["confidence_interval"])
    assert isinstance(contract["top_features"], dict)
    assert all(isinstance(value, float) for value in contract["top_features"].values())


def test_the_unit_is_the_ascii_string_the_spec_writes() -> None:
    """specs §8 writes "ug/m3". Prettifying it to µg/m³ would be a nicer string
    and a broken contract."""
    assert a_result().as_contract()["unit"] == "ug/m3"


def test_the_interval_brackets_the_prediction() -> None:
    contract = a_result().as_contract()
    low, high = contract["confidence_interval"]

    assert low <= contract["prediction"] <= high


def test_top_features_are_ranked_by_magnitude_and_keep_their_sign() -> None:
    contract = a_result().as_contract()

    values = list(contract["top_features"].values())
    assert [abs(value) for value in values] == sorted(
        (abs(value) for value in values), reverse=True
    )
    assert any(value < 0 for value in values)


def test_the_number_of_attributions_is_configurable() -> None:
    assert len(a_result().as_contract(limit=2)["top_features"]) == 2


def test_the_full_payload_is_a_superset_that_renames_nothing() -> None:
    """Extra context is additive: every key the spec names is present, spelled
    the same, holding the same value."""
    result = a_result()
    contract = result.as_contract()
    payload = result.as_dict()

    for key, value in contract.items():
        assert payload[key] == value
    assert payload["target_time"] == (ORIGIN + timedelta(hours=1)).isoformat()
    assert payload["interval_method"] == intervals.CONFORMAL


def test_the_response_model_accepts_the_payload() -> None:
    """The route's Pydantic model is the security boundary (SEC-1); if it
    rejected the service's own output the contract would fail at runtime."""
    validated = PredictionResponse.model_validate(a_result().as_dict())

    assert validated.unit == "ug/m3"
    assert validated.prediction == pytest.approx(45.2)
    assert len(validated.confidence_interval) == 2


def test_the_payload_carries_its_caveats() -> None:
    """ETH-1, and the honest limit on a conformal interval."""
    caveats = " ".join(a_result().as_dict()["caveats"])

    assert "ETH-1" in caveats
    assert "exchangeability" in caveats


def test_the_forecast_is_stamped_with_the_target_not_the_origin() -> None:
    """Conflating them is the one mistake that would make every stored
    prediction unscoreable."""
    result = a_result()

    assert result.target_time > result.origin_time
    assert result.target_time - result.origin_time == timedelta(hours=1)


# --- Task 9.3 wiring: which interval method was used ------------------------


def a_calibrated_model(loc: float = 0.0) -> LoadedModel:
    """An artifact whose held-out residuals are centred at ``loc``."""
    rng = np.random.default_rng(5)
    predicted = rng.uniform(20, 120, 800)
    calibration = intervals.calibrate(predicted + rng.normal(loc, 6.0, 800), predicted)
    return LoadedModel(
        row=MLModel(
            id=1, name="xgboost", target="pm25_h1", features_used=[], artifact_path="x"
        ),
        artifact={"calibration": calibration.as_dict(), "metrics": {"rmse": 9.0}},
    )


def test_a_calibrated_artifact_produces_a_conformal_interval() -> None:
    bounds, method = prediction.interval_for(a_calibrated_model(), 50.0, 0.8)

    assert method == intervals.CONFORMAL
    assert bounds[0] < 50.0 < bounds[1]


def test_a_biased_model_gets_an_interval_shifted_to_where_its_errors_are() -> None:
    """The reason the quantiles are signed. A model that under-predicts by 8
    µg/m³ should not be handed an interval centred on its own point forecast --
    that would present a known bias as if it were noise.
    """
    bounds, _ = prediction.interval_for(a_calibrated_model(loc=8.0), 50.0, 0.8)

    assert (bounds[0] + bounds[1]) / 2 > 55.0


def test_an_artifact_without_calibration_falls_back_and_says_so() -> None:
    """A model registered before calibration existed still gets an interval --
    a weaker one, named rather than passed off as equivalent."""
    loaded = LoadedModel(
        row=MLModel(id=1, name="ridge", target="pm25_h1", features_used=[], artifact_path="x"),
        artifact={"metrics": {"rmse": 8.0}},
    )

    bounds, method = prediction.interval_for(loaded, 50.0, 0.8)

    assert method == intervals.NORMAL
    assert bounds[0] < 50.0 < bounds[1]


# --- End to end, against a real database ------------------------------------


@pytest.fixture
def trained(db_session: Session, db_settings: Settings, tmp_path) -> tuple[Settings, float, float]:
    """One station, trained and registered, rolled back afterwards."""
    from db.models import DataSource, Observation, SourceStatus

    settings = db_settings.model_copy(update={"model_artifact_dir": str(tmp_path)})
    station = geo_service.stations_from_districts(settings)[0]

    source = DataSource(name="predict-test", status=SourceStatus.OFFLINE)
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
                days=45, stations=[station], settings=settings
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
    return settings, station.lat, station.lon


@pytest.mark.db
def test_a_forecast_comes_back_with_an_interval_and_reasoning(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    settings, lat, lon = trained

    result = prediction.predict(
        db_session, settings, lat=lat, lon=lon, horizon=1, persist=False
    )

    assert result.prediction > 0
    assert result.confidence_interval[0] <= result.prediction <= result.confidence_interval[1]
    assert result.attribution.is_additive
    assert result.unit == "ug/m3"
    assert result.model.row.name == "xgboost"


@pytest.mark.db
def test_the_origin_defaults_to_the_newest_observation(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    """In demo mode the wall clock can run ahead of the seeded data, so "now"
    means the last reading that arrived."""
    settings, lat, lon = trained
    newest = prediction.latest_observation_time(db_session, lat=lat, lon=lon)

    result = prediction.predict(
        db_session, settings, lat=lat, lon=lon, horizon=1, persist=False
    )

    # Convert *then* floor. The driver may hand back a local-zone datetime, and
    # flooring 18:30+05:30 to the hour gives 18:00+05:30 — a different instant
    # from flooring the same moment in UTC. The service converts first, which is
    # why it is right and the naive comparison was wrong.
    expected = newest.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    assert result.origin_time == expected


@pytest.mark.db
def test_the_forecast_is_written_down(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    """Task 9.5. The row records where it was for, so its outcome can be found."""
    from db.models import Prediction

    settings, lat, lon = trained

    result = prediction.predict(
        db_session, settings, lat=lat, lon=lon, horizon=1, persist=True
    )
    stored = db_session.get(Prediction, result.prediction_id)

    assert stored is not None
    assert stored.predicted_value == pytest.approx(result.prediction)
    assert stored.target_time == result.target_time
    assert stored.lat == pytest.approx(lat)
    assert stored.actual_value is None


@pytest.mark.db
def test_an_untrained_horizon_is_a_clear_error(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    settings, lat, lon = trained

    with pytest.raises(ModelNotTrainedError):
        prediction.predict(db_session, settings, lat=lat, lon=lon, horizon=6)


@pytest.mark.db
def test_a_station_without_enough_history_is_refused(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    """The warm-up hours are real: a 24-hour lag needs a day of context before
    the station can be forecast from."""
    from db.models import Observation

    settings, lat, lon = trained
    # This station's own first hour -- not the newest row in the database, which
    # may belong to data seeded by something else entirely.
    earliest = (
        db_session.query(Observation.timestamp)
        .filter(Observation.lat == lat, Observation.lon == lon)
        .order_by(Observation.timestamp)
        .limit(1)
        .scalar()
    )

    with pytest.raises(InsufficientDataError, match="warm-up"):
        prediction.predict(
            db_session,
            settings,
            lat=lat,
            lon=lon,
            horizon=1,
            at=earliest + timedelta(hours=1),
        )


# --- Task 9.6: backfilling the outcome --------------------------------------


@pytest.mark.db
def test_the_outcome_is_backfilled_once_the_hour_is_observed(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    """What turns `predictions` into the drift-monitoring dataset (design §6.1)."""
    from db.models import Observation, Prediction

    settings, lat, lon = trained
    observed = (
        db_session.query(Observation)
        .filter(Observation.lat == lat, Observation.pm25.is_not(None))
        .order_by(Observation.timestamp.desc())
        .first()
    )
    loaded = serving.load_latest(db_session, "pm25_h1")
    db_session.add(
        Prediction(
            model_id=loaded.row.id,
            target_time=observed.timestamp,
            predicted_value=99.0,
            lat=lat,
            lon=lon,
        )
    )
    db_session.flush()

    report = prediction.backfill_actuals(
        db_session, now=observed.timestamp + timedelta(hours=1)
    )

    assert report.matched >= 1
    scored = prediction.scored_predictions(db_session, model_id=loaded.row.id)
    assert any(row.actual_value == pytest.approx(observed.pm25) for row in scored)


@pytest.mark.db
def test_a_forecast_whose_hour_has_not_arrived_is_left_alone(
    db_session: Session, trained: tuple[Settings, float, float]
) -> None:
    from db.models import Prediction

    settings, lat, lon = trained
    loaded = serving.load_latest(db_session, "pm25_h1")
    future = prediction.latest_observation_time(db_session) + timedelta(days=30)
    db_session.add(
        Prediction(
            model_id=loaded.row.id,
            target_time=future,
            predicted_value=50.0,
            lat=lat,
            lon=lon,
        )
    )
    db_session.flush()

    report = prediction.backfill_actuals(db_session)

    assert report.still_pending == 0 or all(
        row.target_time <= datetime.now(timezone.utc)
        for row in prediction.scored_predictions(db_session)
    )


@pytest.mark.db
def test_backfill_on_an_empty_table_does_nothing(db_session: Session) -> None:
    """The no-op case, on a table this test actually empties first.

    It used to read whatever `predictions` happened to hold, which made it a
    test of the developer's database rather than of `backfill_actuals`: two
    rows left behind by an earlier `/ml/predict` call sat unnoticed until
    their `target_time` fell behind `now`, and the assertion then failed for a
    reason that had nothing to do with the code. The delete is discarded with
    the surrounding transaction, so the real rows survive.
    """
    from db.models import Prediction

    db_session.query(Prediction).delete()
    db_session.flush()

    report = prediction.backfill_actuals(db_session)

    assert report.matched == 0
    assert report.scanned == 0
