"""Shared test fixtures.

Tests must not read a developer's real ``.env``, so every fixture builds
Settings explicitly and the cache is cleared around each test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from core.config import Settings, get_settings
from main import create_app


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Hermetic settings: no env file, no credentials."""
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
