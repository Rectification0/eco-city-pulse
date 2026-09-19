"""ML endpoints (Phase 7).

Against a real database inside a rolled-back transaction. The contract worth
checking here is not the arithmetic -- that is covered offline in
``test_ml_training.py`` -- but that the evidence reaches the caller: a training
response that did not carry its leakage audit and its baseline comparison would
be a number without a warrant.

Requests are kept narrow (one horizon, one or two rungs, no classical
baselines) so the suite stays fast; the full run is the CLI's job.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from core.config import Settings, get_settings
from db.models import DataSource, SourceStatus
from main import create_app
from services import demo_data, geo_service

pytestmark = pytest.mark.db


@pytest.fixture
def ml_settings(db_settings: Settings, tmp_path) -> Settings:
    """Live database, but artifacts land in a temp directory.

    Registration is append-only by design, so a test run that used the real
    artifact directory would leave model files behind on every pass.
    """
    return db_settings.model_copy(update={"model_artifact_dir": str(tmp_path)})


@pytest.fixture
def ml_client(db_session: Session, ml_settings: Settings) -> Iterator[TestClient]:
    """A TestClient sharing the rolled-back session and the temp artifact dir."""
    app = create_app(ml_settings)
    app.dependency_overrides[get_settings] = lambda: ml_settings
    app.dependency_overrides[get_db_session] = lambda: db_session
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def seeded(db_session: Session, ml_settings: Settings) -> DataSource:
    """One station, 45 days: enough to split, train and score quickly."""
    from db.models import Observation

    stations = geo_service.stations_from_districts(ml_settings)[:1]
    source = DataSource(
        name=f"ml-route-{id(db_session)}", status=SourceStatus.OFFLINE
    )
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
                days=45, stations=stations, settings=ml_settings
            )
        ]
    )
    db_session.flush()
    return source


def _url(settings: Settings, path: str) -> str:
    return f"{settings.api_v1_prefix}{path}"


def _train(client: TestClient, settings: Settings, source: DataSource, **overrides):
    body = {
        "source_ids": [source.id],
        "horizons": [1],
        "models": ["ridge"],
        "include_classical": False,
        "persist": False,
        **overrides,
    }
    return client.post(_url(settings, "/ml/train"), json=body)


# --- POST /ml/train ---------------------------------------------------------


def test_training_returns_a_report_per_horizon(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    response = _train(ml_client, ml_settings, seeded)

    assert response.status_code == 200
    body = response.json()
    assert len(body["horizons"]) == 1
    assert body["horizons"][0]["target"] == "pm25_h1"


def test_the_response_carries_the_leakage_audit(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    """AC-8 reaching the caller. A claim of no leakage that cannot be inspected
    is a claim to be taken on faith."""
    body = _train(ml_client, ml_settings, seeded).json()

    audit = body["horizons"][0]["leakage_audit"]
    assert audit["train_precedes_test"]
    assert audit["embargo_respected"]
    assert audit["no_shared_timestamps"]
    assert audit["train_end"] < audit["test_start"]
    assert body["leakage_clean"]


def test_the_baseline_is_always_included_even_if_not_requested(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    """A score without the floor it has to clear cannot be judged (AC-7)."""
    body = _train(ml_client, ml_settings, seeded, models=["ridge"]).json()

    names = [model["name"] for model in body["horizons"][0]["models"]]
    assert "naive_lag1" in names
    assert "ridge" in names


def test_every_model_reports_metrics_and_skill(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    body = _train(ml_client, ml_settings, seeded).json()

    for model in body["horizons"][0]["models"]:
        assert model["metrics"]["mae"] >= 0
        assert model["metrics"]["rows"] > 0
        assert model["skill_vs_baseline"] is not None
        assert model["cross_validation"]["folds"] >= 2


def test_the_scaled_model_reports_its_scaler_audit(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    body = _train(ml_client, ml_settings, seeded).json()

    ridge = next(m for m in body["horizons"][0]["models"] if m["name"] == "ridge")
    assert ridge["scaler_audit"]["fitted_on_train_only"]


def test_the_response_states_which_features_were_dropped(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    """A selection stage that cannot say why a feature is gone is
    indistinguishable from a bug."""
    body = _train(ml_client, ml_settings, seeded).json()

    selection = body["horizons"][0]["feature_selection"]
    assert selection["kept"]
    assert isinstance(selection["dropped"], dict)


def test_the_response_disclaims_what_r2_hides(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    body = _train(ml_client, ml_settings, seeded).json()

    assert any("autocorrelated" in caveat for caveat in body["caveats"])


def test_too_narrow_a_window_is_a_422_not_a_500(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    start = demo_data.floor_to_hour(datetime.now(timezone.utc))
    response = _train(
        ml_client,
        ml_settings,
        seeded,
        start=(start - timedelta(hours=6)).isoformat(),
        end=start.isoformat(),
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


def test_an_unknown_model_name_is_rejected(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    response = _train(ml_client, ml_settings, seeded, models=["transformer"])

    assert response.status_code == 422


def test_an_out_of_range_horizon_is_rejected(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    response = _train(ml_client, ml_settings, seeded, horizons=[0])

    assert response.status_code == 422


# --- GET /ml/models ---------------------------------------------------------


def test_the_registry_is_empty_before_anything_is_trained(
    ml_client: TestClient, ml_settings: Settings
) -> None:
    body = ml_client.get(_url(ml_settings, "/ml/models")).json()

    assert body == []


def test_a_persisted_run_reaches_the_registry(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    """Task 7.12 through the API: metrics and artifact path both land."""
    _train(ml_client, ml_settings, seeded, persist=True)

    body = ml_client.get(_url(ml_settings, "/ml/models")).json()

    assert len(body) >= 2  # the baseline and ridge
    entry = body[0]
    assert entry["target"] == "pm25_h1"
    assert entry["mae"] is not None
    assert entry["artifact_path"]
    assert entry["features_used"]


def test_the_registry_can_be_filtered_by_target(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    _train(ml_client, ml_settings, seeded, persist=True)

    matching = ml_client.get(
        _url(ml_settings, "/ml/models"), params={"target": "pm25_h1"}
    ).json()
    other = ml_client.get(
        _url(ml_settings, "/ml/models"), params={"target": "pm25_h24"}
    ).json()

    assert matching
    assert other == []


def test_the_registry_lists_the_whole_ladder_not_just_the_winner(
    ml_client: TestClient, ml_settings: Settings, seeded: DataSource
) -> None:
    """The comparison *is* the point; a table showing only the winner would
    hide the baseline it has to beat."""
    _train(ml_client, ml_settings, seeded, persist=True)

    names = {entry["name"] for entry in ml_client.get(_url(ml_settings, "/ml/models")).json()}

    assert "naive_lag1" in names


# --- Contract ---------------------------------------------------------------


def test_every_phase_seven_route_is_documented(
    ml_client: TestClient, ml_settings: Settings
) -> None:
    paths = ml_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/ml/train" in paths
    assert "/api/v1/ml/models" in paths
