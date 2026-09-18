"""Application settings.

SEC-2: every secret arrives through an environment variable and is parsed
exactly once, here. Nothing in this module carries a real credential as a
default -- the defaults only cover non-secret operational values.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionMode(str, Enum):
    """DR-1: the platform must run with no live API reachable at all.

    ``DEMO`` is the default precisely so a fresh clone with no API keys still
    demonstrates the full pipeline (AC-2).
    """

    SCHEDULED = "scheduled"
    MANUAL = "manual"
    UPLOAD = "upload"
    DEMO = "demo"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "Eco-City Pulse"
    app_version: str = "0.1.0"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # --- CORS (SEC-3: allowlist only, never "*") ---
    # Comma-separated. Kept as a string because pydantic-settings parses list
    # fields as JSON, which makes a plain comma-separated env var fail.
    frontend_origins: str = "http://localhost:5173,http://localhost:8080"

    # --- Database ---
    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_db: str = "ecocitypulse"
    postgres_user: str = "ecocity"
    # SecretStr, not str: it keeps the value out of reprs, logs, and tracebacks
    # even when a settings object is nested inside another structure.
    postgres_password: SecretStr = SecretStr("")

    # --- Ingestion ---
    ingestion_mode: IngestionMode = IngestionMode.DEMO

    # --- Third-party API keys (optional by design: absent key => offline source) ---
    aqicn_api_key: SecretStr = SecretStr("")
    openweather_api_key: SecretStr = SecretStr("")
    tomtom_api_key: SecretStr = SecretStr("")

    # --- Storage paths ---
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    model_artifact_dir: str = "artifacts"

    # Both derived values below are plain properties, deliberately NOT
    # pydantic computed fields. A computed field is rendered in the model repr
    # *and* included in model_dump(), which would push the assembled password
    # into every log line and dump of a Settings object. Neither value needs
    # to be part of the serialised shape.
    @property
    def cors_origins(self) -> list[str]:
        """Parsed CORS allowlist with blanks stripped."""
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL assembled from parts so the password stays in env."""
        return (
            f"postgresql+psycopg://{self.postgres_user}"
            f":{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def has_live_credentials(self) -> bool:
        """True when at least one upstream API key is configured.

        Used by the ingestion layer to decide whether a live fetch is even
        worth attempting before falling back to offline data (DR-1).
        """
        return any(
            key.get_secret_value()
            for key in (
                self.aqicn_api_key,
                self.openweather_api_key,
                self.tomtom_api_key,
            )
        )


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Also the FastAPI dependency for settings injection."""
    return Settings()
