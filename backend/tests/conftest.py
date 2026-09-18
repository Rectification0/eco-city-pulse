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

from core.config import Settings, get_settings
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
