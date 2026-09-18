"""Phase 0 exit criteria: the app boots and /api/v1/health answers (AC-1)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from core.config import Settings


def test_health_returns_ok(client: TestClient, settings: Settings) -> None:
    response = client.get(f"{settings.api_v1_prefix}/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == settings.app_name
    assert body["version"] == settings.app_version
    assert body["environment"] == "test"


def test_health_reports_demo_ingestion_mode_by_default(
    client: TestClient, settings: Settings
) -> None:
    """DR-1: a fresh install must default to the offline-capable mode."""
    body = client.get(f"{settings.api_v1_prefix}/health").json()

    assert body["ingestion_mode"] == "demo"
    assert body["live_credentials_configured"] is False


def test_health_never_leaks_credentials(client: TestClient, settings: Settings) -> None:
    """SEC-2: the payload reports only whether keys exist, never their values."""
    body = client.get(f"{settings.api_v1_prefix}/health").json()

    assert set(body) == {
        "status",
        "app",
        "version",
        "environment",
        "ingestion_mode",
        "live_credentials_configured",
    }


def test_health_is_mounted_under_the_versioned_prefix(client: TestClient) -> None:
    """The route must not also be reachable unversioned."""
    assert client.get("/health").status_code == 404


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]
