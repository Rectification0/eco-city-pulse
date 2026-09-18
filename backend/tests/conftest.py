"""Shared test fixtures.

Hermeticity is the whole point here. ``Settings`` draws from three places --
constructor arguments, the process environment, and the ``.env`` file -- and
``_env_file=None`` only closes the third. A developer who has ``POSTGRES_USER``
exported for their local server (a normal thing to have) would otherwise see
assertions about the assembled connection URL fail for reasons that have
nothing to do with the code. So the environment is stripped too.

The real environment is captured once, before any stripping, so the
``db``-marked tests can still find whatever server the developer configured.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from core.config import Settings, get_settings
from db.session import get_engine
from main import create_app

# Settings is case-insensitive, so both spellings have to go.
_SETTINGS_ENV_NAMES = tuple(
    spelling
    for field in Settings.model_fields
    for spelling in (field, field.upper())
)


def _settings_from_real_environment() -> Settings | None:
    """Settings as the developer's machine actually configures them.

    Built at import time, before isolation kicks in, and only used by the
    ``db``-marked tests to locate a live server. Returns None if that
    configuration is itself invalid -- collection should not fail over it.
    """
    try:
        return Settings()
    except Exception:  # pragma: no cover - depends on the developer's env
        return None


LIVE_SETTINGS = _settings_from_real_environment()


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SETTINGS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    """Configuration pointing at a real database, or a skip if there is none."""
    if LIVE_SETTINGS is None:
        pytest.skip("no valid database configuration in the environment")
    return LIVE_SETTINGS


@pytest.fixture
def settings() -> Settings:
    """Hermetic settings: no env file, no environment, no credentials."""
    return Settings(
        _env_file=None,
        app_env="test",
        postgres_password="test-only",
        frontend_origins="http://localhost:5173",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    # create_app passes settings to the factory, but routes resolve settings
    # through the get_settings dependency -- override it to match.
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --- Live database ----------------------------------------------------------

REQUIRED_TABLES = frozenset(
    {
        "data_sources",
        "observations",
        "models",
        "predictions",
        "ingestion_runs",
        "quarantined_records",
        "alembic_version",
    }
)


def _describe_database(settings: Settings) -> str:
    return f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"


@pytest.fixture(scope="session")
def live_engine(live_settings: Settings) -> Engine:
    """A migrated database, or a skip that says which of the two is missing.

    Distinguishing "nothing answering" from "answering but never migrated"
    matters: the second is a one-command fix, and reporting it as a failure
    would punish anyone who happens to run PostgreSQL for another project.
    """
    target = _describe_database(live_settings)
    try:
        engine = get_engine(live_settings)
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

    return engine


@pytest.fixture
def db_session(live_engine: Engine) -> Iterator[Session]:
    """A session whose every write is rolled back when the test ends.

    The session joins an outer transaction on a dedicated connection and
    creates a savepoint for each nested commit, so code under test can call
    ``commit()`` normally -- the ingestion service does -- while the outer
    rollback still leaves the developer's database exactly as it was.
    """
    connection = live_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def db_settings(live_settings: Settings) -> Settings:
    """Live connection details, but hermetic in every other respect.

    Credentials for the upstream APIs are blanked so a developer who *does*
    have a real AQICN key exported cannot make the test suite call it.
    """
    return live_settings.model_copy(
        update={
            "app_env": "test",
            "aqicn_api_key": SecretStr(""),
            "openweather_api_key": SecretStr(""),
            "tomtom_api_key": SecretStr(""),
        }
    )


@pytest.fixture
def db_client(db_session: Session, db_settings: Settings) -> Iterator[TestClient]:
    """A TestClient whose routes share the rolled-back session."""
    app = create_app(db_settings)
    app.dependency_overrides[get_settings] = lambda: db_settings
    app.dependency_overrides[get_db_session] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend() -> str:
    """anyio's pytest plugin ships with FastAPI; asyncio is the only backend
    this project runs on, so trio is not worth parametrising over."""
    return "asyncio"
