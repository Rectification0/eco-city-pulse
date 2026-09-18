"""Domain exceptions map to HTTP responses without router try/except."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from core.exceptions import (
    DatasetNotFoundError,
    EcoCityPulseError,
    ModelNotTrainedError,
    SchemaValidationError,
    register_exception_handlers,
)


@pytest.fixture
def error_client() -> TestClient:
    """A throwaway app whose routes only raise, to exercise the handlers."""
    app = FastAPI()
    register_exception_handlers(app)

    class Payload(BaseModel):
        count: int

    @app.get("/dataset")
    async def _dataset() -> None:
        raise DatasetNotFoundError("No dataset 42.", details={"dataset_id": 42})

    @app.get("/model")
    async def _model() -> None:
        raise ModelNotTrainedError("No model registered for pm25_h1.")

    @app.get("/schema")
    async def _schema() -> None:
        raise SchemaValidationError("Column 'pm25' missing.")

    @app.post("/echo")
    async def _echo(payload: Payload) -> Payload:
        return payload

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("path", "status_code", "code"),
    [
        ("/dataset", 404, "dataset_not_found"),
        ("/model", 409, "model_not_trained"),
        ("/schema", 422, "schema_validation_failed"),
    ],
)
def test_domain_errors_map_to_status_codes(
    error_client: TestClient, path: str, status_code: int, code: str
) -> None:
    response = error_client.get(path)

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code


def test_error_envelope_shape_is_consistent(error_client: TestClient) -> None:
    error = error_client.get("/dataset").json()["error"]

    assert set(error) == {"code", "message", "details"}
    assert error["details"] == {"dataset_id": 42}


def test_request_validation_uses_the_same_envelope(error_client: TestClient) -> None:
    """SEC-1: Pydantic rejection is reported like any other domain error."""
    response = error_client.post("/echo", json={"count": "not-a-number"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "request_validation_failed"
    assert error["details"]["errors"]


def test_base_error_defaults_to_500() -> None:
    assert EcoCityPulseError.status_code == 500
    assert EcoCityPulseError("boom").to_payload()["error"]["details"] == {}
