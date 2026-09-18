"""Domain exceptions and their HTTP mapping.

design §5 rule: routers validate input and call a service. Services raise the
exceptions below; the handlers registered here turn them into HTTP responses,
so no router needs a try/except block.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY to _CONTENT and deprecated
# the old name. A literal is version-proof across the range in requirements.txt.
HTTP_422 = 422


class EcoCityPulseError(Exception):
    """Base class for every expected domain failure.

    ``status_code`` and ``code`` are class attributes so a service can raise a
    subclass with just a message and still produce a well-formed response.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


# --- Ingestion (Phase 2) ---


class DataSourceUnavailableError(EcoCityPulseError):
    """A live upstream could not be reached.

    Note this is *not* raised during normal ingestion: DR-1 requires a failed
    live fetch to degrade the source to ``offline`` and continue. It is raised
    only when a caller explicitly demands fresh live data.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "data_source_unavailable"


class SchemaValidationError(EcoCityPulseError):
    """An uploaded or fetched record failed its source schema (SEC-1)."""

    status_code = HTTP_422
    code = "schema_validation_failed"


# --- Data / EDA (Phases 3-4) ---


class DatasetNotFoundError(EcoCityPulseError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "dataset_not_found"


class InsufficientDataError(EcoCityPulseError):
    """Not enough observations for the requested statistic or lag window."""

    status_code = HTTP_422
    code = "insufficient_data"


# --- ML (Phases 7-9) ---


class ModelNotFoundError(EcoCityPulseError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "model_not_found"


class ModelNotTrainedError(EcoCityPulseError):
    """Inference requested before any model was registered for the target."""

    status_code = status.HTTP_409_CONFLICT
    code = "model_not_trained"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so every error response shares one envelope shape."""

    @app.exception_handler(EcoCityPulseError)
    async def _domain_error(_: Request, exc: EcoCityPulseError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # SEC-1: Pydantic rejection reported in the same envelope as domain errors.
        return JSONResponse(
            status_code=HTTP_422,
            content={
                "error": {
                    "code": "request_validation_failed",
                    "message": "Request failed schema validation.",
                    # jsonable_encoder: error ctx can hold exception objects
                    # that json.dumps cannot serialise.
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )
