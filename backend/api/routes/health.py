"""Liveness endpoint.

Scope note: this is app liveness only. Database readiness is deliberately not
checked here -- the SQLAlchemy engine and session dependency arrive with the
data layer (task 1.7), and the ``db`` container carries its own pg_isready
healthcheck in docker-compose. AC-1 is satisfied by the frontend reaching
this route through the Nginx proxy.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from core.config import IngestionMode, Settings, get_settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """SEC-1: responses are Pydantic models, not bare dicts."""

    status: Literal["ok"] = "ok"
    app: str = Field(description="Application name.")
    version: str = Field(description="Application version.")
    environment: str = Field(description="Deployment environment.")
    ingestion_mode: IngestionMode = Field(
        description="Active ingestion mode; 'demo' needs no network (DR-1)."
    )
    live_credentials_configured: bool = Field(
        description="Whether any upstream API key is present. Never exposes the key itself."
    )


@router.get("/health", response_model=HealthResponse, summary="Service liveness")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        ingestion_mode=settings.ingestion_mode,
        live_credentials_configured=settings.has_live_credentials(),
    )
