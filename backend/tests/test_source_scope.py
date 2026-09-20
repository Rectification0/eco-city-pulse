"""Keeping modelled values out of measured statistics (ETH-1).

``observations`` is one wide table across every source, which is what makes the
lag and rolling features well defined -- and what makes a demo row and an AQICN
row indistinguishable once loaded. Before ``resolve_source_ids`` existed, a
request that named no sources got both: 31,680 synthetic points pooled with
whatever live readings had arrived, described in the response as though all of
it had been observed.

The property each test here defends is that one request reads one provenance,
and that the choice is the caller's or the configuration's -- never an accident
of which rows happen to be in the table.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from core.config import AnalyticsScope, Settings, get_settings
from db.models import DataSource, Observation
from main import create_app
from services import datasets, ingestion_service
from services.adapters import (
    DemoAdapter,
    SyntheticTrafficAdapter,
    registered_specs,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _settings(scope: AnalyticsScope) -> Settings:
    return Settings(
        _env_file=None, postgres_password="test-only", analytics_source_scope=scope
    )


# --- Offline: provenance is declared, not inferred --------------------------


def test_only_the_generating_adapters_declare_themselves_synthetic() -> None:
    """The flag has to mean "generates values", not "has no API key".

    Uploads and a keyless AQICN also have ``api_url`` of None or no credential,
    and lumping them in with the generators would exclude real measurements
    from a measured scope.
    """
    synthetic = {spec.name for spec in registered_specs() if spec.synthetic}

    assert synthetic == {DemoAdapter.spec.name, SyntheticTrafficAdapter.spec.name}


def test_an_explicit_request_outranks_the_configured_scope() -> None:
    """`source_ids=[5]` has to keep meaning "source 5".

    A scope that could override the caller would make the parameter advisory,
    and an analyst comparing one source against another could not say what they
    were looking at.
    """
    # No session needed: naming sources answers the question outright, and
    # that early return is the guarantee -- an explicit request must not depend
    # on the database being reachable to be honoured.
    resolved = datasets.resolve_source_ids(
        None, (1, 2), settings=_settings(AnalyticsScope.DEMO)
    )

    assert resolved == (1, 2)


def test_without_settings_no_policy_is_applied() -> None:
    """What keeps unit tests hermetic: a direct call returns the rows the test
    inserted, not whatever the developer's .env says the scope is."""
    assert datasets.resolve_source_ids(None, None) is None


# --- Against the database ---------------------------------------------------


@pytest.mark.db
def test_demo_scope_selects_the_generated_sources_and_live_the_measured_ones(
    db_session: Session,
) -> None:
    """The partition has to be exhaustive and disjoint: every source lands in
    exactly one of the two scopes, or rows go missing from both."""
    ingestion_service.ensure_sources(db_session)

    demo = datasets.resolve_source_ids(
        db_session, None, settings=_settings(AnalyticsScope.DEMO)
    )
    live = datasets.resolve_source_ids(
        db_session, None, settings=_settings(AnalyticsScope.LIVE)
    )
    every = set(db_session.scalars(select(DataSource.id)).all())

    assert set(demo).isdisjoint(live)
    assert set(demo) | set(live) == every
    assert demo, "the demo bundle and traffic fallback must be in the demo scope"


@pytest.mark.db
def test_the_all_scope_applies_no_filter(db_session: Session) -> None:
    """`all` is the pre-existing pooled behaviour, kept for the case where
    someone deliberately wants both provenances."""
    ingestion_service.ensure_sources(db_session)

    assert (
        datasets.resolve_source_ids(
            db_session, None, settings=_settings(AnalyticsScope.ALL)
        )
        is None
    )


@pytest.mark.db
def test_a_scope_matching_no_source_reads_nothing_rather_than_everything(
    db_session: Session,
) -> None:
    """The failure this guards is silent and total.

    An empty tuple and ``None`` both look falsy, so the obvious ``if
    source_ids:`` would treat "no source matched this scope" as "no filter
    wanted" and return the whole table -- the exact blend the scope exists to
    prevent, produced by the code meant to prevent it.
    """
    frame = datasets.load_observations(db_session, source_ids=())

    assert frame.empty


@pytest.mark.db
def test_a_demo_scoped_load_carries_no_measured_rows(db_session: Session) -> None:
    """The end-to-end property, asserted on the frame rather than on the query:
    what a caller receives is what the scope promised."""
    sources = ingestion_service.ensure_sources(db_session)
    demo_source = sources[DemoAdapter.spec.name]
    live_source = sources["AQICN"]

    for index, (source, pm25) in enumerate(((demo_source, 10.0), (live_source, 90.0))):
        db_session.add(
            Observation(
                source_id=source.id,
                timestamp=START,
                lat=28.61 + index,
                lon=77.21,
                pm25=pm25,
            )
        )
    db_session.flush()

    scoped = datasets.resolve_source_ids(
        db_session, None, settings=_settings(AnalyticsScope.DEMO)
    )
    frame = datasets.load_observations(db_session, source_ids=scoped)

    assert live_source.id not in set(frame["source_id"])
    assert demo_source.id in set(frame["source_id"])


@pytest.mark.db
def test_ensure_sources_corrects_a_flag_that_drifted_from_the_registry(
    db_session: Session,
) -> None:
    """The registry is the authority. A row left claiming the demo bundle is
    measured would put generated values into a measured scope, which is worse
    than having no scope at all -- it would be wrong *and* look deliberate."""
    sources = ingestion_service.ensure_sources(db_session)
    demo_source = sources[DemoAdapter.spec.name]
    demo_source.is_synthetic = False
    db_session.flush()

    ingestion_service.ensure_sources(db_session)

    assert demo_source.is_synthetic is True


# --- The dashboard endpoints ------------------------------------------------
#
# These two build their own query rather than going through
# ``load_observations``, so nothing about the resolver reaches them
# automatically. They shipped unscoped once; these tests are why that cannot
# happen quietly a second time.


def _scoped_client(
    session: Session, settings: Settings, scope: AnalyticsScope
) -> TestClient:
    scoped = settings.model_copy(update={"analytics_source_scope": scope})
    app = create_app(scoped)
    app.dependency_overrides[get_settings] = lambda: scoped
    app.dependency_overrides[get_db_session] = lambda: session
    return TestClient(app)


def _two_provenances_at_one_point(session: Session) -> tuple[float, float]:
    """A demo reading and a newer live one at the same coordinate.

    The same coordinate on purpose: OpenWeather and TomTom are fetched *at the
    district centroid*, which is exactly where the demo bundle already sits, so
    collision is the normal case rather than an edge one.
    """
    sources = ingestion_service.ensure_sources(session)
    lat, lon = 28.61, 77.21
    session.add_all(
        [
            Observation(
                source_id=sources[DemoAdapter.spec.name].id,
                timestamp=START,
                lat=lat,
                lon=lon,
                pm25=10.0,
            ),
            Observation(
                source_id=sources["AQICN"].id,
                timestamp=START.replace(hour=6),
                lat=lat,
                lon=lon,
                pm25=90.0,
            ),
        ]
    )
    session.flush()
    return lat, lon


@pytest.mark.db
def test_the_latest_reading_comes_from_the_scoped_provenance(
    db_session: Session, db_settings: Settings
) -> None:
    """DISTINCT ON resolves each coordinate to the newest row, so unscoped it
    silently hands the map whichever provenance wrote last."""
    lat, lon = _two_provenances_at_one_point(db_session)

    with _scoped_client(db_session, db_settings, AnalyticsScope.DEMO) as client:
        body = client.get(_url(db_settings, "/data/observations/latest")).json()

    here = [r for r in body["readings"] if r["lat"] == lat and r["lon"] == lon]

    assert [r["pm25"] for r in here] == [10.0], "the live row outranked the scope"


@pytest.mark.db
def test_a_station_trendline_never_splices_two_provenances(
    db_session: Session, db_settings: Settings
) -> None:
    """The blend a chart cannot show. Two provenances at one coordinate plot as
    one continuous line, and the step where the series changes what it reads is
    indistinguishable from a change in the air."""
    lat, lon = _two_provenances_at_one_point(db_session)

    with _scoped_client(db_session, db_settings, AnalyticsScope.LIVE) as client:
        body = client.get(
            _url(db_settings, "/data/observations/series"),
            params={"lat": lat, "lon": lon, "hours": 48},
        ).json()

    assert [p["pm25"] for p in body["points"]] == [90.0]


def _url(settings: Settings, path: str) -> str:
    return f"{settings.api_v1_prefix}{path}"
