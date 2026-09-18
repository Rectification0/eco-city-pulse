"""Settings behaviour: SEC-2 (env-sourced secrets) and DR-1 (offline default)."""

from __future__ import annotations

import pytest

from core.config import IngestionMode, Settings


def test_database_url_is_assembled_from_parts(settings: Settings) -> None:
    assert settings.database_url == (
        "postgresql+psycopg://ecocity:test-only@db:5432/ecocitypulse"
    )


@pytest.fixture
def leaky_settings() -> Settings:
    return Settings(
        _env_file=None,
        postgres_password="super-secret",
        aqicn_api_key="key-aqicn",
        openweather_api_key="key-ow",
        tomtom_api_key="key-tomtom",
    )


SECRET_VALUES = ("super-secret", "key-aqicn", "key-ow", "key-tomtom")


@pytest.mark.parametrize("render", [repr, str])
def test_secrets_are_not_exposed_when_settings_are_rendered(
    leaky_settings: Settings, render: object
) -> None:
    """SEC-2: a settings object in a log or traceback must not print secrets.

    Covers repr and str, and by extension the computed database_url, which is
    the surface that actually leaked first: a computed field is included in
    the model repr even when the underlying field sets repr=False.
    """
    rendered = render(leaky_settings)  # type: ignore[operator]

    for secret in SECRET_VALUES:
        assert secret not in rendered


def test_secrets_are_not_exposed_in_model_dump(leaky_settings: Settings) -> None:
    """A dump is the other easy way a secret reaches a log or an API response."""
    rendered = str(leaky_settings.model_dump())

    for secret in SECRET_VALUES:
        assert secret not in rendered


def test_database_url_still_carries_the_real_password(leaky_settings: Settings) -> None:
    """Masking must not break the connection string itself."""
    assert "super-secret" in leaky_settings.database_url


def test_cors_origins_parses_comma_separated_list() -> None:
    settings = Settings(
        _env_file=None,
        frontend_origins="http://localhost:5173, http://localhost:8080 ,",
    )

    assert settings.cors_origins == ["http://localhost:5173", "http://localhost:8080"]


def test_cors_allowlist_never_contains_a_wildcard(settings: Settings) -> None:
    """SEC-3: '*' would defeat the origin restriction the spec requires."""
    assert "*" not in settings.cors_origins


def test_default_ingestion_mode_is_demo() -> None:
    """DR-1 / AC-2: the platform must work with every live API disabled."""
    assert Settings(_env_file=None).ingestion_mode is IngestionMode.DEMO


def test_settings_read_ingestion_mode_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_MODE", "scheduled")

    assert Settings(_env_file=None).ingestion_mode is IngestionMode.SCHEDULED


def test_invalid_ingestion_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGESTION_MODE", "streaming")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ({}, False),
        ({"aqicn_api_key": ""}, False),
        ({"aqicn_api_key": "k"}, True),
        ({"openweather_api_key": "k"}, True),
        ({"tomtom_api_key": "k"}, True),
    ],
)
def test_has_live_credentials(keys: dict[str, str], expected: bool) -> None:
    """An empty SecretStr is still truthy as an object, so the check must look
    at the wrapped value -- otherwise a blank key would read as configured."""
    assert Settings(_env_file=None, **keys).has_live_credentials() is expected
