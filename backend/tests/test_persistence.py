"""Session wiring (task 1.7) and behaviour against a live database.

Everything below the ``db`` marker needs a reachable PostgreSQL. Those tests
skip rather than fail when one is absent, so the suite stays runnable on a
laptop with nothing running, while still exercising the real schema in CI and
in compose. Point them at the compose database with:

    POSTGRES_HOST=localhost POSTGRES_PORT=5433 pytest -m db
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from api import dependencies as api_dependencies
from api.dependencies import SessionDep, get_db_session
from core.config import Settings
from db import session as db_session
from db.models import DataSource, MLModel, Observation, Prediction, SourceStatus
from tests import conftest

UTC_NOON = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


# --- Engine and session plumbing (no server required) ------------------------


@pytest.fixture
def clean_engines() -> Iterator[None]:
    """Isolate engine-cache assertions without disposing the live session's
    engine, which is module-scoped and shared."""
    db_session.dispose_engines()
    yield
    db_session.dispose_engines()


def test_engine_uses_the_configured_url(clean_engines: None, settings: Settings) -> None:
    engine = db_session.get_engine(settings)

    assert engine.url.host == settings.postgres_host
    assert engine.url.database == settings.postgres_db


def test_engine_is_reused_per_url(clean_engines: None, settings: Settings) -> None:
    """A fresh engine per request would mean a fresh pool per request."""
    assert db_session.get_engine(settings) is db_session.get_engine(settings)


def test_different_databases_get_different_engines(clean_engines: None, settings: Settings) -> None:
    other = settings.model_copy(update={"postgres_db": "somewhere_else"})

    assert db_session.get_engine(settings) is not db_session.get_engine(other)


def test_engine_does_not_echo_sql_by_default(clean_engines: None, settings: Settings) -> None:
    """Echo drowns out bulk-load output; it is opt-in via DB_ECHO."""
    assert db_session.get_engine(settings).echo is False


def test_importing_the_app_does_not_require_a_database() -> None:
    """The engine is lazy. If it were not, this import would already have
    tried to connect -- and the rest of the suite could not run offline."""
    from main import create_app

    assert create_app(
        Settings(_env_file=None, postgres_host="nonexistent.invalid")
    ) is not None


def test_session_dependency_commits_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[str] = []

    class FakeSession:
        def commit(self) -> None:
            recorded.append("commit")

        def rollback(self) -> None:
            recorded.append("rollback")

        def close(self) -> None:
            recorded.append("close")

    monkeypatch.setattr(
        api_dependencies, "get_sessionmaker", lambda settings=None: FakeSession
    )

    generator = get_db_session(Settings(_env_file=None))
    next(generator)
    with pytest.raises(StopIteration):
        next(generator)

    assert recorded == ["commit", "close"]


def test_session_dependency_rolls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A route that raises mid-write must not leave a partial transaction."""
    recorded: list[str] = []

    class FakeSession:
        def commit(self) -> None:
            recorded.append("commit")

        def rollback(self) -> None:
            recorded.append("rollback")

        def close(self) -> None:
            recorded.append("close")

    monkeypatch.setattr(
        api_dependencies, "get_sessionmaker", lambda settings=None: FakeSession
    )

    generator = get_db_session(Settings(_env_file=None))
    next(generator)
    with pytest.raises(RuntimeError):
        generator.throw(RuntimeError("route blew up"))

    assert recorded == ["rollback", "close"]


def test_session_dependency_is_injectable() -> None:
    """SessionDep must resolve through FastAPI's override machinery, which is
    how every later phase will test a database-backed route."""
    app = FastAPI()

    @app.get("/probe")
    async def _probe(session: SessionDep) -> dict[str, bool]:
        return {"injected": session is sentinel}

    sentinel = object()
    app.dependency_overrides[get_db_session] = lambda: sentinel

    with TestClient(app) as client:
        assert client.get("/probe").json() == {"injected": True}


# --- Live database -----------------------------------------------------------


REQUIRED_TABLES = frozenset(
    {"data_sources", "observations", "models", "predictions", "alembic_version"}
)


@pytest.fixture(scope="module")
def db(request: pytest.FixtureRequest) -> Session:
    """A session against the configured database, or a skip if none is ready.

    Two distinct reasons to skip, and the message says which: no server
    answering at all, or a server that has never been migrated. Treating the
    second as a failure would punish anyone who has PostgreSQL installed for
    some other project.
    """
    settings = conftest.LIVE_SETTINGS
    if settings is None:  # pragma: no cover - environment dependent
        pytest.skip("no database configuration in the environment")

    target = f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    try:
        engine = db_session.get_engine(settings)
        with engine.connect() as connection:
            present = set(
                connection.scalars(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                ).all()
            )
    except (OperationalError, ProgrammingError) as exc:  # pragma: no cover
        pytest.skip(f"no database at {target} ({exc.__class__.__name__})")

    missing = REQUIRED_TABLES - present
    if missing:  # pragma: no cover - environment dependent
        pytest.skip(f"{target} is not migrated (run: alembic upgrade head)")

    session = db_session.get_sessionmaker(settings)()
    request.addfinalizer(session.close)
    return session


@pytest.fixture
def source(db: Session) -> Iterator[DataSource]:
    """A throwaway source, removed with its observations afterwards."""
    record = DataSource(name=f"pytest-{datetime.now(timezone.utc).timestamp()}")
    db.add(record)
    db.flush()
    yield record
    db.rollback()


@pytest.mark.db
def test_migrations_have_been_applied(db: Session) -> None:
    """Guards against running the suite against an empty database."""
    tables = set(
        db.scalars(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        ).all()
    )

    assert {"data_sources", "observations", "models", "predictions"} <= tables
    assert "alembic_version" in tables


@pytest.mark.db
def test_composite_index_exists_on_the_server(db: Session) -> None:
    """Task 1.6 -- verified where it matters, not just in the metadata."""
    definition = db.scalar(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE indexname = 'ix_observations_timestamp_lat_lon'"
        )
    )

    assert definition is not None
    assert '"timestamp", lat, lon' in definition.replace("  ", " ")


@pytest.mark.db
def test_observation_round_trips_as_utc(db: Session, source: DataSource) -> None:
    """DR-2 survives the driver, not only the ORM validator."""
    ist = timezone(timedelta(hours=5, minutes=30))
    db.add(
        Observation(
            source_id=source.id,
            timestamp=datetime(2026, 1, 1, 17, 30, tzinfo=ist),
            lat=28.61,
            lon=77.21,
            pm25=88.4,
        )
    )
    db.flush()
    db.expire_all()

    stored = db.scalar(
        select(Observation).where(Observation.source_id == source.id)
    )
    assert stored is not None
    assert stored.timestamp == UTC_NOON
    assert stored.is_anomaly is False  # server default, never pre-judged


@pytest.mark.db
def test_duplicate_observations_are_rejected(db: Session, source: DataSource) -> None:
    """The constraint that makes re-ingestion idempotent (Phase 2)."""
    for _ in range(2):
        db.add(
            Observation(
                source_id=source.id, timestamp=UTC_NOON, lat=28.61, lon=77.21, pm25=10.0
            )
        )

    with pytest.raises(IntegrityError):
        db.flush()


@pytest.mark.db
@pytest.mark.parametrize(
    ("lat", "lon"),
    [(91.0, 77.21), (-91.0, 77.21), (28.61, 181.0), (28.61, -181.0)],
)
def test_coordinates_outside_decimal_degrees_are_rejected(
    db: Session, source: DataSource, lat: float, lon: float
) -> None:
    """DR-3: out-of-range values are unit errors, not exotic locations."""
    db.add(Observation(source_id=source.id, timestamp=UTC_NOON, lat=lat, lon=lon))

    with pytest.raises(IntegrityError):
        db.flush()


@pytest.mark.db
def test_extreme_pollutant_values_are_accepted(db: Session, source: DataSource) -> None:
    """AC-5: an outlier must reach the table so it can be flagged, not dropped."""
    db.add(
        Observation(
            source_id=source.id, timestamp=UTC_NOON, lat=28.61, lon=77.21, pm25=987.0
        )
    )
    db.flush()

    assert db.scalar(
        select(func.count())
        .select_from(Observation)
        .where(Observation.source_id == source.id)
    ) == 1


@pytest.mark.db
def test_model_registry_round_trips(db: Session) -> None:
    """features_used is JSONB: it must come back as a list, not a string."""
    model = MLModel(
        name="xgboost-pytest",
        target="pm25_h1",
        features_used=["pm25_lag_1h", "traffic_score"],
        mae=4.2,
        rmse=6.1,
        r2=0.87,
        artifact_path="artifacts/pytest.json",
    )
    db.add(model)
    db.flush()

    db.add(
        Prediction(model_id=model.id, target_time=UTC_NOON, predicted_value=45.2)
    )
    db.flush()
    db.expire_all()

    stored = db.get(MLModel, model.id)
    assert stored is not None
    assert stored.features_used == ["pm25_lag_1h", "traffic_score"]
    assert stored.created_at.tzinfo is not None
    # design §6.1: nullable until the real observation is available (task 9.6).
    assert stored.predictions[0].actual_value is None

    db.rollback()


@pytest.mark.db
def test_demo_data_is_loaded(db: Session) -> None:
    """Phase 1 exit criterion: the demo seed has populated the database."""
    from services.demo_data import DEMO_SOURCE_NAME

    bundle = db.scalar(select(DataSource).where(DataSource.name == DEMO_SOURCE_NAME))
    if bundle is None:
        pytest.skip("demo seed has not been run (python -m scripts.seed_demo)")

    count = db.scalar(
        select(func.count())
        .select_from(Observation)
        .where(Observation.source_id == bundle.id)
    )

    assert count and count > 1000
    assert bundle.status is SourceStatus.HEALTHY
    assert bundle.last_run is not None
