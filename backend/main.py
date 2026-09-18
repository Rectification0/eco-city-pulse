"""FastAPI application factory.

Kept deliberately thin: configure, mount, register handlers. All analytics
logic lives in ``services/`` (design §5).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import api_v1_router
from core.config import Settings, get_settings
from core.exceptions import register_exception_handlers


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        summary="Urban environmental intelligence: EDA and interpretable PM2.5 forecasting.",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # SEC-3: explicit origin allowlist. Never "*" -- that would defeat the
    # restriction the spec requires.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
