"""Data routes: source health, triggers, upload, log (tasks 2.9, 2.10).

Runs against a real database inside a rolled-back transaction, because the
things worth asserting here -- that a keyless source reports as offline rather
than vanishing, that a failed source still returns 200 -- only exist once the
service layer has actually written a run.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from core.config import Settings
from services import geo_service, ingestion_service
from services.adapters import aqicn, openweather, tomtom

pytestmark = pytest.mark.db

CSV_UPLOAD = (
    b"timestamp,latitude,longitude,PM2.5,pm10,temperature,rh,traffic\n"
    b"2026-04-01T00:30:00,28.61,77.21,55.2,110.4,18.0,60,45\n"
    b"2026-04-01T00:45:00,28.61,77.21,61.0,,19.0,,50\n"
    b"not-a-time,28.61,77.21,10,20,21,50,30\n"
)


@pytest.fixture(autouse=True)
def _no_upstream_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """No route test may reach an upstream, whatever the developer's env holds.

    Patched per adapter module rather than on ``httpx.Client`` itself:
    ``TestClient`` *is* an httpx client, so a global block would stop the
    requests these tests are made of.
    """

    class Blocked:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("route tests must not contact an upstream API")

    for module in (aqicn, openweather, tomtom):
        monkeypatch.setattr(module.httpx, "Client", Blocked, raising=True)


def _url(settings: Settings, path: str) -> str:
    return f"{settings.api_v1_prefix}{path}"


# --- GET /data/sources (task 2.10) ------------------------------------------


def test_sources_lists_every_registered_source(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.get(_url(db_settings, "/data/sources"))

    assert response.status_code == 200
    names = {source["name"] for source in response.json()["sources"]}
    assert {"AQICN", "OpenWeather", "TomTom Traffic"} <= names


def test_a_keyless_source_is_listed_as_offline_not_omitted(
    db_client: TestClient, db_settings: Settings
) -> None:
    """"AQICN is offline for want of a key" is information; a missing row is
    indistinguishable from a bug."""
    body = db_client.get(_url(db_settings, "/data/sources")).json()
    aqicn = next(s for s in body["sources"] if s["name"] == "AQICN")

    assert aqicn["requires_credentials"] is True
    assert aqicn["credentials_configured"] is False
    assert aqicn["status"] == "offline"


def test_a_keyless_source_needing_no_credential_is_not_reported_as_missing_one(
    db_client: TestClient, db_settings: Settings
) -> None:
    """The demo bundle is ready to run; it has no key because it needs none."""
    body = db_client.get(_url(db_settings, "/data/sources")).json()
    demo = next(s for s in body["sources"] if "Demo" in s["name"])

    assert demo["requires_credentials"] is False
    assert demo["credentials_configured"] is True


def test_sources_never_leak_a_credential(
    db_client: TestClient, db_settings: Settings
) -> None:
    """SEC-2: the payload reports only whether a key exists."""
    body = db_client.get(_url(db_settings, "/data/sources")).json()

    for source in body["sources"]:
        assert set(source) == {
            "id",
            "name",
            "api_url",
            "status",
            "last_run",
            "domain",
            "requires_credentials",
            "credentials_configured",
            "observation_count",
            "last_observation_at",
            "latest_run",
        }


def test_sources_report_the_active_ingestion_mode(
    db_client: TestClient, db_settings: Settings
) -> None:
    body = db_client.get(_url(db_settings, "/data/sources")).json()

    assert body["ingestion_mode"] == db_settings.ingestion_mode.value


def test_sources_carry_ingestion_health_after_a_run(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    ingestion_service.run_demo(db_session, db_settings, days=1)
    db_session.flush()

    body = db_client.get(_url(db_settings, "/data/sources")).json()
    demo = next(s for s in body["sources"] if "Demo" in s["name"])

    assert demo["status"] == "healthy"
    assert demo["last_run"] is not None
    assert demo["observation_count"] > 0
    assert demo["latest_run"]["status"] == "success"
    assert demo["latest_run"]["records_written"] > 0


# --- POST /data/ingest (task 2.9) -------------------------------------------


def test_manual_trigger_returns_per_source_outcomes(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(_url(db_settings, "/data/ingest"))

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "manual"
    assert body["outcomes"]


def test_a_keyless_run_still_returns_200(
    db_client: TestClient, db_settings: Settings
) -> None:
    """DR-1 expressed in the API: a source that cannot run is reported, not
    turned into a 5xx for the caller."""
    body = db_client.post(_url(db_settings, "/data/ingest")).json()
    statuses = {o["source_name"]: o["status"] for o in body["outcomes"]}

    assert statuses["AQICN"] == "skipped"
    assert statuses["OpenWeather"] == "skipped"


def test_manual_trigger_falls_back_to_synthetic_traffic(
    db_client: TestClient, db_settings: Settings
) -> None:
    """task 2.2, end to end through the API."""
    body = db_client.post(_url(db_settings, "/data/ingest")).json()
    names = [o["source_name"] for o in body["outcomes"]]

    assert any("Synthetic" in name for name in names)


def test_demo_mode_can_be_triggered_explicitly(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(
        _url(db_settings, "/data/ingest"), params={"mode": "demo", "days": 1}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "demo"
    assert body["outcomes"][0]["records_written"] > 0


def test_upload_mode_cannot_be_triggered_without_a_file(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(_url(db_settings, "/data/ingest"), params={"mode": "upload"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "upload_rejected"


def test_an_unknown_mode_is_rejected_by_validation(
    db_client: TestClient, db_settings: Settings
) -> None:
    """SEC-1: the query parameter is a Pydantic enum, not free text."""
    response = db_client.post(
        _url(db_settings, "/data/ingest"), params={"mode": "streaming"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"


# --- POST /data/upload (task 2.3) -------------------------------------------


def test_uploading_a_csv_ingests_it(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("observations.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
    )

    assert response.status_code == 201
    outcome = response.json()["outcomes"][0]
    assert outcome["records_fetched"] == 3
    assert outcome["records_written"] >= 1


def test_a_malformed_row_is_quarantined_not_fatal(
    db_client: TestClient, db_settings: Settings
) -> None:
    """One bad line in an export is not a reason to reject the export."""
    response = db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("observations.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
    )

    outcome = response.json()["outcomes"][0]
    assert outcome["status"] == "partial"
    assert outcome["records_quarantined"] == 1
    assert outcome["records_valid"] == 2


def test_the_timezone_assumption_is_echoed_back(
    db_client: TestClient, db_settings: Settings
) -> None:
    """DR-2: the caller states the offset and it is recorded, not guessed."""
    response = db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("observations.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
        data={"assume_timezone_offset_minutes": 330},
    )

    assert "UTC+05:30" in response.json()["outcomes"][0]["message"]


def test_an_empty_upload_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "upload_rejected"


def test_an_oversized_upload_is_rejected(
    db_client: TestClient, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit is enforced on bytes actually read, not on a client-supplied
    Content-Length."""
    small = db_settings.model_copy(update={"upload_max_bytes": 32})
    db_client.app.dependency_overrides[  # type: ignore[attr-defined]
        next(
            dep
            for dep in db_client.app.dependency_overrides  # type: ignore[attr-defined]
            if dep.__name__ == "get_settings"
        )
    ] = lambda: small

    response = db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("big.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
    )

    assert response.status_code == 422
    assert "limit" in response.json()["error"]["details"]


def test_an_invalid_timezone_offset_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("observations.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
        data={"assume_timezone_offset_minutes": 5000},
    )

    assert response.status_code == 422


# --- GET /data/ingestion/runs (task 2.8) ------------------------------------


def test_the_ingestion_log_is_readable(
    db_client: TestClient, db_settings: Settings
) -> None:
    db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("observations.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
    )

    body = db_client.get(_url(db_settings, "/data/ingestion/runs")).json()

    assert body["runs"]
    assert body["runs"][0]["mode"] == "upload"
    assert body["quarantined_sample"]


def test_the_log_does_not_serve_raw_upstream_payloads(
    db_client: TestClient, db_settings: Settings
) -> None:
    """A quarantined payload is an arbitrary third-party document; this
    endpoint answers "is ingestion healthy", not "show me that body"."""
    db_client.post(
        _url(db_settings, "/data/upload"),
        files={"file": ("observations.csv", io.BytesIO(CSV_UPLOAD), "text/csv")},
    )

    entry = db_client.get(_url(db_settings, "/data/ingestion/runs")).json()[
        "quarantined_sample"
    ][0]

    assert set(entry) == {"id", "run_id", "reason", "created_at"}


def test_the_log_respects_its_limit(
    db_client: TestClient, db_settings: Settings
) -> None:
    db_client.post(_url(db_settings, "/data/ingest"))

    body = db_client.get(
        _url(db_settings, "/data/ingestion/runs"), params={"limit": 1}
    ).json()

    assert len(body["runs"]) == 1


# --- Contract ---------------------------------------------------------------


def test_every_phase_two_route_is_documented(
    db_client: TestClient, db_settings: Settings
) -> None:
    paths = db_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/data/sources" in paths
    assert "/api/v1/data/ingest" in paths
    assert "/api/v1/data/upload" in paths
    assert "/api/v1/data/ingestion/runs" in paths


# --- POST /data/quality (Phase 3) -------------------------------------------


def test_the_quality_endpoint_reports_a_full_run(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    ingestion_service.run_demo(db_session, db_settings, days=4)
    db_session.flush()

    response = db_client.post(
        _url(db_settings, "/data/quality"), params={"persist": False}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rows_in"] > 0
    assert body["rows_preserved"] is True  # AC-5
    assert body["feature_set_is_complete"] is True  # AC-4
    assert body["imputation"]["remaining_nulls"] == {
        "pm25": 0,
        "pm10": 0,
        "temp": 0,
        "humidity": 0,
        "traffic_score": 0,
    }


def test_the_quality_response_explains_each_mechanism(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    """design §7: the MCAR/MAR judgement is the justification for the
    imputation choice, so it has to reach the caller."""
    ingestion_service.run_demo(db_session, db_settings, days=4)
    db_session.flush()

    body = db_client.post(
        _url(db_settings, "/data/quality"), params={"persist": False}
    ).json()

    for column in body["missingness"]["columns"]:
        assert column["mechanism"] in {"complete", "mcar", "mar", "mnar"}
        assert column["justification"]


def test_the_quality_response_never_claims_mnar_was_excluded(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    ingestion_service.run_demo(db_session, db_settings, days=4)
    db_session.flush()

    body = db_client.post(
        _url(db_settings, "/data/quality"), params={"persist": False}
    ).json()

    assert any("MNAR cannot be ruled out" in c for c in body["missingness"]["caveats"])


def test_the_quality_response_qualifies_what_a_flag_means(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    """ETH-1's habit applied to the quality engine: no number without its
    limits."""
    ingestion_service.run_demo(db_session, db_settings, days=4)
    db_session.flush()

    body = db_client.post(
        _url(db_settings, "/data/quality"), params={"persist": False}
    ).json()

    assert "sensor fault" in body["anomalies"]["note"]


def test_the_vote_threshold_is_rejected_when_out_of_range(
    db_client: TestClient, db_settings: Settings
) -> None:
    """SEC-1: query parameters are validated, not trusted."""
    response = db_client.post(
        _url(db_settings, "/data/quality"), params={"min_votes": 9}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"


def test_the_quality_route_is_documented(
    db_client: TestClient, db_settings: Settings
) -> None:
    paths = db_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/data/quality" in paths


def test_the_quality_response_reports_mice_convergence(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    """A run that did not settle is information the caller should have, not a
    warning on a console nobody is reading."""
    ingestion_service.run_demo(db_session, db_settings, days=4)
    db_session.flush()

    body = db_client.post(
        _url(db_settings, "/data/quality"), params={"persist": False}
    ).json()

    imputation = body["imputation"]
    assert imputation["stations_imputed"] >= 1
    assert imputation["stations_not_converged"] <= imputation["stations_imputed"]


# --- Current readings (Phase 10) --------------------------------------------
#
# These two reads exist because building the frontend found nothing that served
# an *observation*: the profile returns statistics, the ESI an index, predict a
# forecast. The dashboard's hero tile (10.3) and map gradient (10.4) need what
# the sensors currently say.


def _seed_two_stations(db_session: Session, settings: Settings) -> None:
    from db.models import DataSource, Observation, SourceStatus

    source = DataSource(name=f"obs-{id(db_session)}", status=SourceStatus.OFFLINE)
    db_session.add(source)
    db_session.flush()

    stations = geo_service.stations_from_districts(settings)[:2]
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db_session.add_all(
        [
            Observation(
                source_id=source.id,
                timestamp=start + timedelta(hours=hour),
                lat=station.lat,
                lon=station.lon,
                pm25=40.0 + hour + index * 10,
                pm10=80.0,
                temp=22.0,
                humidity=55.0,
                traffic_score=30.0,
            )
            for index, station in enumerate(stations)
            for hour in range(6)
        ]
    )
    db_session.flush()


def test_latest_returns_one_row_per_station(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    _seed_two_stations(db_session, db_settings)

    body = db_client.get(_url(db_settings, "/data/observations/latest")).json()

    stations = [reading["station"] for reading in body["readings"]]
    assert len(stations) == len(set(stations))
    # The newest hour, not an arbitrary one.
    for reading in body["readings"]:
        assert reading["timestamp"] == body["observed_at"] or reading["timestamp"] <= body["observed_at"]


def test_latest_names_the_district_a_station_sits_in(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    """The map keys its polygons by district, so a reading without one cannot
    colour anything."""
    _seed_two_stations(db_session, db_settings)

    body = db_client.get(_url(db_settings, "/data/observations/latest")).json()

    assert any(reading["district_name"] for reading in body["readings"])


def test_latest_reports_how_stale_it_is(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    """Demo data stops at the hour it was seeded; a dashboard that showed it as
    'now' without qualification would be lying by omission."""
    _seed_two_stations(db_session, db_settings)

    body = db_client.get(_url(db_settings, "/data/observations/latest")).json()

    assert body["stale_minutes"] is not None
    assert body["stale_minutes"] > 0


def test_series_comes_back_oldest_first(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    """A chart reads left to right."""
    _seed_two_stations(db_session, db_settings)
    station = geo_service.stations_from_districts(db_settings)[0]

    body = db_client.get(
        _url(db_settings, "/data/observations/series"),
        params={"lat": station.lat, "lon": station.lon, "hours": 4},
    ).json()

    stamps = [point["timestamp"] for point in body["points"]]
    assert stamps == sorted(stamps)
    assert len(stamps) == 4


def test_series_returns_nothing_for_a_place_with_no_station(
    db_client: TestClient, db_settings: Settings, db_session: Session
) -> None:
    _seed_two_stations(db_session, db_settings)

    body = db_client.get(
        _url(db_settings, "/data/observations/series"),
        params={"lat": 0.0, "lon": 0.0},
    ).json()

    assert body["points"] == []


def test_an_impossible_coordinate_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.get(
        _url(db_settings, "/data/observations/series"),
        params={"lat": 999, "lon": 77.2},
    )

    assert response.status_code == 422
