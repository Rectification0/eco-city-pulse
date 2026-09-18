"""Application settings.

SEC-2: every secret arrives through an environment variable and is parsed
exactly once, here. Nothing in this module carries a real credential as a
default -- the defaults only cover non-secret operational values.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent

# Resolved from this file, not the working directory: the .env sits at the repo
# root (where docker compose also reads it), but the backend is normally run
# from backend/, so a bare ".env" would silently never be found. A backend/.env
# still wins if one exists, since later entries take precedence.
_ENV_FILES = (_REPO_ROOT / ".env", _BACKEND_DIR / ".env")


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
        env_file=_ENV_FILES,
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
    # Separate from `debug` on purpose: SQL echo prints every statement and its
    # bound parameters, which drowns out the output of a bulk load and is rarely
    # what someone flipping DEBUG actually wants.
    db_echo: bool = False
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

    # --- Paths -------------------------------------------------------------
    # Configured as relative strings, resolved by _resolve_dir below. The two
    # candidate bases are not interchangeable: in the container the backend is
    # copied to /app and ./data is mounted inside it, while in a local checkout
    # data/ is a sibling of backend/. Probing both is what lets one value work
    # in both places.
    def _resolve_dir(self, configured: str) -> Path:
        path = Path(configured)
        if path.is_absolute():
            return path
        for base in (_BACKEND_DIR, _REPO_ROOT):
            candidate = base / path
            if candidate.exists():
                return candidate
        # Nothing exists yet (first write): the repo root is the right place
        # to create it.
        return _REPO_ROOT / path

    @property
    def data_raw_path(self) -> Path:
        """Immutable landing zone (design §5)."""
        return self._resolve_dir(self.data_raw_dir)

    @property
    def data_processed_path(self) -> Path:
        """Cleaned / feature-engineered output (design §5)."""
        return self._resolve_dir(self.data_processed_dir)

    @property
    def model_artifact_path(self) -> Path:
        """Model registry storage backing ``models.artifact_path``."""
        return self._resolve_dir(self.model_artifact_dir)

    @property
    def districts_geojson_path(self) -> Path:
        """City district boundaries (task 1.8)."""
        return self.data_raw_path / "districts.geojson"

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
        """SQLAlchemy URL assembled from parts so the password stays in env.

        User and password are percent-encoded: a password containing "@", ":",
        "/", "?" or "#" -- all common in generated credentials -- would
        otherwise produce a URL that parses into the wrong host or fails
        outright. safe="" so that "/" is encoded too.
        """
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql+psycopg://{user}:{password}"
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
