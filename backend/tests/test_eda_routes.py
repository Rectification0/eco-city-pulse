"""EDA endpoints (tasks 4.3, 4.5, 4.7, 6.4, 6.5).

Against a real database inside a rolled-back transaction, because the contract
worth checking -- that AC-3's four statistics reach the caller for every
column -- is about the response, not about a DataFrame.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from core.config import Settings
from db.models import DataSource, Observation, SourceStatus
from services.datasets import MEASUREMENT_COLUMNS
from services.eda import cache, decomposition

pytestmark = pytest.mark.db

START = datetime(2026, 7, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    """Each test starts with an empty cache, so hit/miss assertions mean
    something regardless of what ran before."""
    cache.PROFILE_CACHE.clear()


@pytest.fixture
def seeded(db_session: Session) -> DataSource:
    """A source with a clean daily cycle: enough for STL, small enough to be fast."""
    import math

    source = DataSource(
        name=f"eda-route-{datetime.now(timezone.utc).timestamp()}",
        status=SourceStatus.HEALTHY,
    )
    db_session.add(source)
    db_session.flush()

    for hour in range(24 * 12):
        value = 60 + 18 * math.sin(2 * math.pi * hour / 24) + (hour % 7)
        db_session.add(
            Observation(
                source_id=source.id,
                timestamp=START + timedelta(hours=hour),
                lat=28.61,
                lon=77.21,
                pm25=None if hour % 50 == 0 else value,
                pm10=value * 2,
                temp=22 + (hour % 12) * 0.4,
                humidity=55 + (hour % 9),
                traffic_score=40 + (hour % 17),
            )
        )
    db_session.flush()
    return source


def _url(settings: Settings, path: str) -> str:
    return f"{settings.api_v1_prefix}{path}"


# --- POST /eda/profile (task 4.3) -------------------------------------------


def test_the_profile_endpoint_answers_with_no_body(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """An analyst opening the EDA Studio wants "everything" by default."""
    response = db_client.post(_url(db_settings, "/eda/profile"))

    assert response.status_code == 200
    assert response.json()["rows"] > 0


def test_ac3_statistics_are_present_for_every_numeric_column(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """AC-3, stated literally: mean, median, IQR and missingness, per column."""
    body = db_client.post(
        _url(db_settings, "/eda/profile"),
        json={"source_ids": [seeded.id]},
    ).json()

    reported = {column["column"] for column in body["univariate"]}
    assert reported == set(MEASUREMENT_COLUMNS)

    for column in body["univariate"]:
        for field in ("mean", "median", "iqr", "missing_pct"):
            assert field in column, f"{field} missing for {column['column']}"


def test_the_profile_carries_both_correlation_methods(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/profile"), json={"source_ids": [seeded.id]}
    ).json()

    assert body["bivariate"]["pearson"]
    assert body["bivariate"]["spearman"]
    assert body["bivariate"]["pairs"]


def test_the_profile_disclaims_causation(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """ETH-1 reaches the API payload, not only the report."""
    body = db_client.post(
        _url(db_settings, "/eda/profile"), json={"source_ids": [seeded.id]}
    ).json()

    assert "causation" in body["bivariate"]["caveat"].lower()
    assert any("causation" in c.lower() for c in body["caveats"])


def test_the_profile_reports_its_dataset_version(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """specs §8 asks for a dataset id; the fingerprint serves that purpose and
    is derived from the data rather than assigned to it."""
    body = db_client.post(
        _url(db_settings, "/eda/profile"), json={"source_ids": [seeded.id]}
    ).json()

    assert len(body["dataset_version"]["fingerprint"]) == 16
    assert body["dataset_version"]["rows"] == 24 * 12


def test_the_second_call_is_served_from_cache(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    payload = {"source_ids": [seeded.id]}

    first = db_client.post(_url(db_settings, "/eda/profile"), json=payload).json()
    second = db_client.post(_url(db_settings, "/eda/profile"), json=payload).json()

    assert first["cached"] is False
    assert second["cached"] is True


def test_a_time_window_narrows_the_profile(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/profile"),
        json={
            "source_ids": [seeded.id],
            "end": (START + timedelta(hours=47)).isoformat(),
        },
    ).json()

    assert body["rows"] == 48


def test_specific_columns_can_be_requested(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/profile"),
        json={"source_ids": [seeded.id], "columns": ["pm25", "temp"]},
    ).json()

    assert {c["column"] for c in body["univariate"]} == {"pm25", "temp"}


def test_an_unknown_column_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    """SEC-1: the request body is validated, not trusted."""
    response = db_client.post(
        _url(db_settings, "/eda/profile"), json={"columns": ["ozone"]}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"


def test_a_backwards_window_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(
        _url(db_settings, "/eda/profile"),
        json={"start": "2026-08-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"},
    )

    assert response.status_code == 422


# --- POST /eda/decompose (task 4.5) -----------------------------------------


@pytest.mark.skipif(
    not decomposition.is_available(),
    reason="statsmodels could not be loaded in this environment",
)
def test_decomposition_returns_all_three_components(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    response = db_client.post(
        _url(db_settings, "/eda/decompose"), json={"source_ids": [seeded.id]}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["trend"]) == len(body["seasonal"]) == len(body["residual"])
    assert body["column"] == "pm25"
    assert 0.0 <= body["strength"]["seasonal"] <= 1.0


@pytest.mark.skipif(
    not decomposition.is_available(),
    reason="statsmodels could not be loaded in this environment",
)
def test_decomposition_reports_how_much_it_invented(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/decompose"), json={"source_ids": [seeded.id]}
    ).json()

    assert "interpolated_points" in body
    assert body["points_analysed"] >= body["points_returned"]


@pytest.mark.skipif(
    not decomposition.is_available(),
    reason="statsmodels could not be loaded in this environment",
)
def test_too_short_a_window_is_a_422_not_a_500(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """A fixable request, so the caller is told it is fixable."""
    response = db_client.post(
        _url(db_settings, "/eda/decompose"),
        json={
            "source_ids": [seeded.id],
            "end": (START + timedelta(hours=30)).isoformat(),
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


def test_an_unknown_decomposition_column_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(
        _url(db_settings, "/eda/decompose"), json={"column": "ozone"}
    )

    assert response.status_code == 422


def test_an_out_of_range_period_is_rejected(
    db_client: TestClient, db_settings: Settings
) -> None:
    response = db_client.post(_url(db_settings, "/eda/decompose"), json={"period": 1})

    assert response.status_code == 422


# --- GET /eda/report (task 4.7) ---------------------------------------------


def test_the_report_endpoint_serves_html(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    response = db_client.get(
        _url(db_settings, "/eda/report"), params={"persist": False}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text.startswith("<!DOCTYPE html>")


def test_the_served_report_is_self_contained(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.get(
        _url(db_settings, "/eda/report"), params={"persist": False}
    ).text

    assert "https://" not in body
    assert "<script" not in body.lower()


def test_the_report_lists_its_sections_in_a_header(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    response = db_client.get(
        _url(db_settings, "/eda/report"),
        params={"persist": False, "include_decomposition": False},
    )

    sections = response.headers["x-report-sections"].split(",")
    assert "correlation" in sections
    assert "decomposition" not in sections


# --- GET /eda/cache ---------------------------------------------------------


def test_cache_statistics_are_exposed(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    db_client.post(_url(db_settings, "/eda/profile"), json={"source_ids": [seeded.id]})

    body = db_client.get(_url(db_settings, "/eda/cache")).json()

    assert body["entries"] >= 1
    assert set(body) == {"entries", "hits", "misses", "max_entries", "ttl_seconds"}


# --- POST /eda/reduce (tasks 6.4, 6.6; AC-6) --------------------------------


def test_the_reduce_endpoint_returns_components_variance_and_loadings(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    response = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={"source_ids": [seeded.id], "persist": False},
    )

    body = response.json()
    assert response.status_code == 200
    first = body["components"][0]
    assert 0 < first["explained_variance_ratio"] <= 1
    assert set(first["loadings"]) == set(MEASUREMENT_COLUMNS)
    assert first["drivers"][0] in MEASUREMENT_COLUMNS


def test_ac6_the_esi_reaches_the_caller_on_a_zero_to_hundred_scale(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={"source_ids": [seeded.id], "persist": False},
    ).json()

    esi = body["esi"]
    assert 0 <= esi["min"] <= esi["max"] <= 100
    assert esi["latest"] is not None


def test_the_reduce_response_says_which_way_pc1_points(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """A component is defined only up to sign, so the response states the
    orientation rather than leaving the reader to guess (design §9)."""
    body = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={"source_ids": [seeded.id], "persist": False},
    ).json()

    assert body["pc1_oriented_by"] == "pm25"
    assert isinstance(body["pc1_sign_flipped"], bool)


def test_the_reduce_response_carries_its_caveats(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={"source_ids": [seeded.id], "persist": False},
    ).json()

    text = " ".join(body["caveats"])
    assert "ETH-1" in text
    assert "clipped" in text


def test_a_component_count_can_be_requested(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={"source_ids": [seeded.id], "n_components": 2, "persist": False},
    ).json()

    assert len(body["components"]) == 2


def test_too_narrow_a_window_for_pca_is_a_422_not_a_500(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    response = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={
            "source_ids": [seeded.id],
            "start": START.isoformat(),
            "end": (START + timedelta(hours=4)).isoformat(),
            "persist": False,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


def test_persistence_can_be_declined(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """The artefact of task 6.2 is opt-out, and the response says which it was.
    (Writing it is covered offline, against a temp directory.)"""
    body = db_client.post(
        _url(db_settings, "/eda/reduce"),
        json={"source_ids": [seeded.id], "persist": False},
    ).json()

    assert body["artifact_path"] is None


# --- POST /eda/tsne (task 6.5) ----------------------------------------------


def test_the_tsne_endpoint_returns_a_two_dimensional_scatter(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    body = db_client.post(
        _url(db_settings, "/eda/tsne"),
        json={"source_ids": [seeded.id], "max_points": 120},
    ).json()

    assert body["points"] <= 120
    assert len(body["x"]) == len(body["y"]) == body["points"]
    assert len(body["timestamps"]) == body["points"]


def test_the_tsne_response_says_the_axes_carry_no_meaning(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """design §9: t-SNE is EDA-only, and the payload says so rather than
    leaving it to whoever writes the UI."""
    body = db_client.post(
        _url(db_settings, "/eda/tsne"),
        json={"source_ids": [seeded.id], "max_points": 120},
    ).json()

    assert "local neighbourhoods" in body["caveat"]
    assert "never feeds a model" in body["caveat"]


def test_an_absurd_point_budget_is_rejected(
    db_client: TestClient, db_settings: Settings, seeded: DataSource
) -> None:
    """t-SNE is quadratic; an unbounded request would be a self-inflicted
    denial of service."""
    response = db_client.post(
        _url(db_settings, "/eda/tsne"),
        json={"source_ids": [seeded.id], "max_points": 500_000},
    )

    assert response.status_code == 422


# --- Contract ---------------------------------------------------------------


def test_every_phase_four_route_is_documented(
    db_client: TestClient, db_settings: Settings
) -> None:
    paths = db_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/eda/profile" in paths
    assert "/api/v1/eda/decompose" in paths
    assert "/api/v1/eda/report" in paths


def test_every_phase_six_route_is_documented(
    db_client: TestClient, db_settings: Settings
) -> None:
    paths = db_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/eda/reduce" in paths
    assert "/api/v1/eda/tsne" in paths
