"""Ingestion engine (tasks 2.7-2.9, 2.11; FEAT-01, DR-1, AC-2).

Every test here writes into a transaction that is rolled back afterwards, so
they run against the real schema -- constraints, upsert semantics and all --
without leaving anything behind.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import IngestionMode, Settings
from core.exceptions import DataSourceUnavailableError
from db.models import (
    DataSource,
    IngestionRun,
    Observation,
    Provenance,
    QuarantinedRecord,
    RunStatus,
    SourceStatus,
)
from services import ingestion_service
from services.adapters.base import AdapterSpec, SourceAdapter, SourceDomain
from services.geo_service import Station
from services.harmonizer import MEASUREMENT_FIELDS, SourceReading

pytestmark = pytest.mark.db

IST = timezone(timedelta(hours=5, minutes=30))
UTC_NOON = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
STATION = Station(district_id="test-district", lat=28.61, lon=77.21)


# --- Test doubles -----------------------------------------------------------


class StubAdapter(SourceAdapter):
    """An adapter whose behaviour each test dictates, with no I/O at all."""

    spec = AdapterSpec(
        name="Stub Source",
        domain=SourceDomain.BUNDLE,
        api_url=None,
        requires_credentials=False,
        # Declares every field, because it stands in for a *source* rather than
        # for the traffic fallback. A source that claims nothing may only fill
        # NULLs, so without this the stub could not update its own values on a
        # re-ingest -- which is the property the tests below exist to check.
        authoritative_for=frozenset(MEASUREMENT_FIELDS),
    )

    def __init__(
        self,
        payloads: Sequence[Mapping[str, Any]] = (),
        *,
        fetch_error: Exception | None = None,
    ) -> None:
        self.payloads = list(payloads)
        self.fetch_error = fetch_error

    def fetch(
        self, settings: Settings, *, stations: Sequence[Station]
    ) -> Iterator[Mapping[str, Any]]:
        if self.fetch_error is not None:
            raise self.fetch_error
        yield from self.payloads

    def parse(self, payload: Mapping[str, Any]) -> list[SourceReading]:
        if payload.get("broken"):
            raise ValueError("payload is broken")
        # Every measurement field, not just the two the earliest tests needed:
        # a stub that silently drops `traffic_score` makes a merge test pass or
        # fail for reasons that have nothing to do with the merge.
        return [
            SourceReading(
                timestamp=payload["timestamp"],
                lat=payload.get("lat", STATION.lat),
                lon=payload.get("lon", STATION.lon),
                **{field: payload.get(field) for field in MEASUREMENT_FIELDS},
            )
        ]


class UnconfiguredAdapter(StubAdapter):
    spec = AdapterSpec(
        name="Unconfigured Source",
        domain=SourceDomain.AIR_QUALITY,
        api_url="https://example.invalid/",
        requires_credentials=True,
    )


@pytest.fixture
def stub_source(db_session: Session) -> DataSource:
    """The ``data_sources`` row the stub adapters ingest into."""
    for spec in (StubAdapter.spec, UnconfiguredAdapter.spec):
        db_session.add(
            DataSource(name=spec.name, api_url=spec.api_url, status=SourceStatus.OFFLINE)
        )
    db_session.flush()
    return db_session.scalar(
        select(DataSource).where(DataSource.name == StubAdapter.spec.name)
    )


def _payload(hour: int = 12, **values: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime(2026, 1, 1, hour, tzinfo=timezone.utc),
        **values,
    }


def _observations(session: Session, source: DataSource) -> list[Observation]:
    return list(
        session.scalars(
            select(Observation)
            .where(Observation.source_id == source.id)
            .order_by(Observation.timestamp)
        )
    )


# --- Source registry --------------------------------------------------------


def test_ensure_sources_registers_every_adapter(
    db_session: Session, db_settings: Settings
) -> None:
    ingestion_service.ensure_sources(db_session)

    names = set(db_session.scalars(select(DataSource.name)))
    assert {"AQICN", "OpenWeather", "TomTom Traffic"} <= names


def test_ensure_sources_is_idempotent(db_session: Session) -> None:
    ingestion_service.ensure_sources(db_session)
    before = db_session.scalar(select(func.count()).select_from(DataSource))
    ingestion_service.ensure_sources(db_session)

    assert db_session.scalar(select(func.count()).select_from(DataSource)) == before


def test_ensure_sources_corrects_a_changed_endpoint(db_session: Session) -> None:
    """An api_url fix in code should not need a migration to take effect."""
    ingestion_service.ensure_sources(db_session)
    source = db_session.scalar(select(DataSource).where(DataSource.name == "AQICN"))
    source.api_url = "https://stale.example/"
    db_session.flush()

    ingestion_service.ensure_sources(db_session)
    assert source.api_url != "https://stale.example/"


# --- Happy path -------------------------------------------------------------


def test_a_successful_run_writes_observations_and_logs_it(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    outcome = ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(12, pm25=50.0), _payload(13, pm25=60.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    assert outcome.status is RunStatus.SUCCESS
    assert (outcome.records_fetched, outcome.records_valid) == (2, 2)
    assert outcome.records_written == 2
    assert [o.pm25 for o in _observations(db_session, stub_source)] == [50.0, 60.0]


def test_a_successful_run_marks_the_source_healthy(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """task 2.8: status and last_run are the ingestion-health signal."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=50.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    assert stub_source.status is SourceStatus.HEALTHY
    assert stub_source.last_run is not None


def test_readings_in_the_same_hour_are_averaged_before_writing(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """DR-4 holds end to end, not just inside the harmonizer."""
    payloads = [
        {"timestamp": datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc), "pm25": 10.0},
        {"timestamp": datetime(2026, 1, 1, 12, 50, tzinfo=timezone.utc), "pm25": 20.0},
    ]

    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter(payloads),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    written = _observations(db_session, stub_source)
    assert len(written) == 1
    assert written[0].pm25 == 15.0
    assert written[0].timestamp == UTC_NOON


def test_timestamps_are_stored_as_utc(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """DR-2, verified where it finally matters: on the server."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([{"timestamp": datetime(2026, 1, 1, 17, 30, tzinfo=IST), "pm25": 1.0}]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    assert _observations(db_session, stub_source)[0].timestamp == UTC_NOON


def test_a_reading_with_no_measurements_is_not_written(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """Otherwise the quality engine would later report a phantom row as 100%
    missing."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload()]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    assert _observations(db_session, stub_source) == []


# --- Offline degradation (DR-1, task 2.8) -----------------------------------


@pytest.mark.parametrize(
    "error",
    [
        DataSourceUnavailableError("upstream is down"),
        httpx.ConnectError("name resolution failed"),
        httpx.ReadTimeout("took too long"),
        RuntimeError("a proxy returned HTML"),
    ],
)
def test_a_failed_fetch_degrades_the_source_without_raising(
    db_session: Session, db_settings: Settings, stub_source: DataSource, error: Exception
) -> None:
    """DR-1, the requirement this phase is built around: availability is never
    coupled to a third party. Any upstream failure -- including ones no adapter
    anticipated -- becomes a logged, offline source, not an exception."""
    outcome = ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter(fetch_error=error),
        mode=IngestionMode.SCHEDULED,
        stations=[STATION],
    )

    assert outcome.status is RunStatus.FAILED
    assert stub_source.status is SourceStatus.OFFLINE
    assert outcome.message and "Fetch failed" in outcome.message


def test_a_failed_fetch_leaves_existing_data_untouched(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """"The platform continues on the most recent persisted data" (design §6.3)."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=50.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter(fetch_error=httpx.ConnectError("down")),
        mode=IngestionMode.SCHEDULED,
        stations=[STATION],
    )

    assert [o.pm25 for o in _observations(db_session, stub_source)] == [50.0]


def test_a_failed_fetch_does_not_advance_last_run(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """last_run means "last time we got data", so a failure must not refresh
    it -- otherwise a dead feed looks fresh in the admin view."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter(fetch_error=httpx.ConnectError("down")),
        mode=IngestionMode.SCHEDULED,
        stations=[STATION],
    )

    assert stub_source.last_run is None


def test_an_unconfigured_source_is_skipped_not_failed(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """DR-1: no key means "not switched on", which is not the same as broken."""
    outcome = ingestion_service.ingest_source(
        db_session,
        db_settings,
        UnconfiguredAdapter([_payload(pm25=1.0)]),
        mode=IngestionMode.SCHEDULED,
        stations=[STATION],
    )

    assert outcome.status is RunStatus.SKIPPED
    assert outcome.records_fetched == 0
    assert "no API key" in (outcome.message or "")


# --- Quarantine (task 2.7) --------------------------------------------------


def test_malformed_records_are_quarantined_and_the_rest_still_load(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """One bad record in a batch is not a reason to lose the batch."""
    outcome = ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(12, pm25=50.0), {"broken": True}, _payload(13, pm25=60.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    assert outcome.status is RunStatus.PARTIAL
    assert (outcome.records_valid, outcome.records_quarantined) == (2, 1)
    assert len(_observations(db_session, stub_source)) == 2


def test_a_quarantined_record_keeps_its_payload_and_reason(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """Quarantine exists so a provider's schema change is visible evidence
    rather than an unexplained gap discovered weeks later."""
    outcome = ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([{"broken": True, "marker": "keep-me"}]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    db_session.flush()

    record = db_session.scalar(
        select(QuarantinedRecord).where(QuarantinedRecord.run_id == outcome.run_id)
    )
    assert record is not None
    assert "broken" in record.reason or "ValueError" in record.reason
    assert record.payload["marker"] == "keep-me"


def test_quarantined_records_degrade_the_source(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([{"broken": True}]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    assert stub_source.status is SourceStatus.DEGRADED


def test_quarantine_storage_is_capped(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """A file that is malformed from the first byte must not write one row per
    line; the count is still reported in full."""
    broken = [{"broken": True} for _ in range(ingestion_service.QUARANTINE_LIMIT + 25)]

    outcome = ingestion_service.ingest_source(
        db_session, db_settings, StubAdapter(broken), mode=IngestionMode.UPLOAD,
        stations=[STATION],
    )
    db_session.flush()

    stored = db_session.scalar(
        select(func.count())
        .select_from(QuarantinedRecord)
        .where(QuarantinedRecord.run_id == outcome.run_id)
    )
    assert outcome.records_quarantined == len(broken)
    assert stored == ingestion_service.QUARANTINE_LIMIT


# --- Upsert semantics (task 2.7) --------------------------------------------


def test_re_ingesting_an_hour_updates_it_rather_than_duplicating(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    for value in (50.0, 70.0):
        ingestion_service.ingest_source(
            db_session,
            db_settings,
            StubAdapter([_payload(pm25=value)]),
            mode=IngestionMode.MANUAL,
            stations=[STATION],
        )

    written = _observations(db_session, stub_source)
    assert len(written) == 1
    assert written[0].pm25 == 70.0


def test_a_later_null_never_erases_a_known_value(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """A partial fetch mid-hour must not delete what a complete fetch wrote."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=50.0, temp=20.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=55.0)]),  # temp absent this time
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    written = _observations(db_session, stub_source)[0]
    assert written.pm25 == 55.0
    assert written.temp == 20.0


def test_re_ingestion_does_not_unflag_an_anomaly(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """AC-5: is_anomaly belongs to Phase 3. Ingestion must never clear it."""
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=900.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    observation = _observations(db_session, stub_source)[0]
    observation.is_anomaly = True
    db_session.flush()

    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=905.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    db_session.expire_all()

    assert _observations(db_session, stub_source)[0].is_anomaly is True


def test_different_sources_keep_their_own_rows(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """The unique key includes source_id: provenance is never merged away."""
    ingestion_service.ensure_sources(db_session)
    ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(pm25=50.0)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )

    from services.adapters.synthetic_traffic import SyntheticTrafficAdapter

    class SameHourTraffic(SyntheticTrafficAdapter):
        def fetch(self, settings, *, stations):  # type: ignore[no-untyped-def]
            yield {
                "district_id": STATION.district_id,
                "lat": STATION.lat,
                "lon": STATION.lon,
                "observed_at": UTC_NOON.isoformat(),
                "traffic_score": 42.0,
            }

    ingestion_service.ingest_source(
        db_session, db_settings, SameHourTraffic(), mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    db_session.flush()

    rows = db_session.scalars(
        select(Observation).where(Observation.timestamp == UTC_NOON)
    ).all()
    assert len({row.source_id for row in rows}) == 2


# --- Modes (task 2.9) -------------------------------------------------------


def test_demo_mode_ingests_offline_through_the_normal_pipeline(
    db_session: Session, db_settings: Settings
) -> None:
    """AC-2, stated literally: the exit criterion for this phase."""
    outcomes = ingestion_service.run_demo(
        db_session, db_settings, days=1, end=datetime(2026, 3, 1, tzinfo=timezone.utc)
    )

    assert len(outcomes) == 1
    assert outcomes[0].status is RunStatus.SUCCESS
    assert outcomes[0].records_written > 0
    assert outcomes[0].records_quarantined == 0


def test_demo_mode_makes_no_network_call(
    db_session: Session, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-2 with every live API disabled, enforced rather than assumed."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("demo ingestion must not touch the network")

    monkeypatch.setattr(httpx.Client, "request", _blocked)
    monkeypatch.setattr(httpx.Client, "send", _blocked)

    outcomes = ingestion_service.run_demo(
        db_session, db_settings, days=1, end=datetime(2026, 3, 1, tzinfo=timezone.utc)
    )

    assert outcomes[0].status is RunStatus.SUCCESS


def test_live_mode_skips_keyless_sources_without_contacting_them(
    db_session: Session, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default install has no keys, so a Manual run must be silent."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("a keyless source must not be contacted")

    monkeypatch.setattr(httpx.Client, "request", _blocked)
    monkeypatch.setattr(httpx.Client, "send", _blocked)

    outcomes = ingestion_service.run_live(
        db_session, db_settings, mode=IngestionMode.MANUAL
    )

    by_name = {o.source_name: o for o in outcomes}
    assert by_name["AQICN"].status is RunStatus.SKIPPED
    assert by_name["OpenWeather"].status is RunStatus.SKIPPED


def test_synthetic_traffic_substitutes_for_a_missing_tomtom_key(
    db_session: Session, db_settings: Settings
) -> None:
    """task 2.2: PM2.5 without traffic loses its strongest explanatory
    variable, so a labelled stand-in beats an empty column."""
    adapters = ingestion_service.live_adapters_for(db_settings)
    names = [adapter.name for adapter in adapters]

    assert "TomTom Traffic" not in names
    assert any("Synthetic" in name for name in names)


def test_a_configured_tomtom_key_takes_precedence_over_the_fallback() -> None:
    settings = Settings(_env_file=None, tomtom_api_key="a-key")
    names = [a.name for a in ingestion_service.live_adapters_for(settings)]

    assert "TomTom Traffic" in names
    assert not any("Synthetic" in name for name in names)


def test_upload_mode_records_the_timezone_assumption(
    db_session: Session, db_settings: Settings
) -> None:
    """DR-2: the assumption applied to naive timestamps is part of the record."""
    rows = [{"timestamp": "2026-01-01 17:30:00", "lat": 28.61, "lon": 77.21, "pm25": 9}]

    outcome = ingestion_service.run_upload(
        db_session, db_settings, rows, assume_timezone=IST, filename="analyst.csv"
    )
    db_session.flush()

    assert outcome.status is RunStatus.SUCCESS
    assert "analyst.csv" in (outcome.message or "")
    assert "UTC+05:30" in (outcome.message or "")

    run = db_session.get(IngestionRun, outcome.run_id)
    assert run is not None and "UTC+05:30" in (run.message or "")


def test_upload_mode_applies_the_assumption_to_the_stored_row(
    db_session: Session, db_settings: Settings
) -> None:
    rows = [{"timestamp": "2026-01-01 17:30:00", "lat": 28.61, "lon": 77.21, "pm25": 9}]
    ingestion_service.run_upload(db_session, db_settings, rows, assume_timezone=IST)
    db_session.flush()

    source = db_session.scalar(
        select(DataSource).where(DataSource.name == "Analyst Upload")
    )
    assert _observations(db_session, source)[0].timestamp == UTC_NOON


def test_run_ingestion_dispatches_on_the_configured_mode(
    db_session: Session, db_settings: Settings
) -> None:
    outcomes = ingestion_service.run_ingestion(
        db_session,
        db_settings.model_copy(update={"ingestion_mode": IngestionMode.DEMO}),
        days=1,
        end=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    assert [o.mode for o in outcomes] == ["demo"]


# --- The log (task 2.8) -----------------------------------------------------


def test_every_run_is_logged_with_its_counts(
    db_session: Session, db_settings: Settings, stub_source: DataSource
) -> None:
    """FEAT-01's stated output."""
    outcome = ingestion_service.ingest_source(
        db_session,
        db_settings,
        StubAdapter([_payload(12, pm25=1.0), {"broken": True}]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )
    db_session.flush()

    run = db_session.get(IngestionRun, outcome.run_id)
    assert run is not None
    assert run.mode == "manual"
    assert run.status is RunStatus.PARTIAL
    assert (run.records_fetched, run.records_valid, run.records_quarantined) == (2, 1, 1)
    assert run.finished_at is not None and run.finished_at >= run.started_at


# --- The merge (migration 0005) ---------------------------------------------
#
# Three feeds describe the same station-hour and each fills a different part of
# it. That they converge on one row is the whole reason `provenance` replaced
# `source_id` in the unique key; these tests are what stop it silently coming
# apart again.


def _feed(name: str, owns: frozenset[str]) -> type[StubAdapter]:
    """A stub standing in for one real source, claiming the fields it owns."""

    class Feed(StubAdapter):
        spec = AdapterSpec(
            name=name,
            domain=SourceDomain.BUNDLE,
            api_url=None,
            requires_credentials=False,
            authoritative_for=owns,
        )

    return Feed


AIR = _feed("Air Feed", frozenset({"pm25"}))
WEATHER = _feed("Weather Feed", frozenset({"temp"}))
TRAFFIC = _feed("Traffic Feed", frozenset({"traffic_score"}))


@pytest.fixture
def three_feeds(db_session: Session) -> None:
    for adapter in (AIR, WEATHER, TRAFFIC):
        db_session.add(
            DataSource(name=adapter.spec.name, status=SourceStatus.OFFLINE)
        )
    db_session.flush()


def _run(session: Session, settings: Settings, adapter_type, **values: Any) -> None:
    ingestion_service.ingest_source(
        session,
        settings,
        adapter_type([_payload(**values)]),
        mode=IngestionMode.MANUAL,
        stations=[STATION],
    )


def _row_at(session: Session, hour: int = 12) -> Observation:
    return session.scalar(
        select(Observation).where(
            Observation.timestamp == datetime(2026, 1, 1, hour, tzinfo=timezone.utc),
            Observation.lat == STATION.lat,
            Observation.lon == STATION.lon,
        )
    )


@pytest.mark.db
def test_three_feeds_describing_one_hour_become_one_row(
    db_session: Session, db_settings: Settings, three_feeds: None
) -> None:
    """The row `observations` always claimed to hold.

    Before the key changed, this produced three rows of one column each, so no
    row carried both the target and its strongest predictor and the map's
    DISTINCT ON returned whichever feed happened to write last.
    """
    _run(db_session, db_settings, AIR, pm25=55.0)
    _run(db_session, db_settings, WEATHER, temp=31.0)
    _run(db_session, db_settings, TRAFFIC, traffic_score=42.0)

    rows = list(
        db_session.scalars(
            select(Observation).where(
                Observation.lat == STATION.lat, Observation.lon == STATION.lon
            )
        )
    )
    assert len(rows) == 1
    assert (rows[0].pm25, rows[0].temp, rows[0].traffic_score) == (55.0, 31.0, 42.0)


@pytest.mark.db
@pytest.mark.parametrize("weather_first", [True, False])
def test_the_authority_decides_a_contested_field_whatever_the_order(
    db_session: Session, db_settings: Settings, three_feeds: None, weather_first: bool
) -> None:
    """`temp` arrives from the air-quality feed and the weather feed at once,
    and they disagree. Without an authority the winner was decided by the order
    of a tuple in the adapter registry -- deterministic, but for no reason
    anyone could point at, and it changed if the tuple was ever reordered."""
    order = [(WEATHER, 31.0), (AIR, 20.0)] if weather_first else [(AIR, 20.0), (WEATHER, 31.0)]
    for adapter_type, temp in order:
        _run(db_session, db_settings, adapter_type, temp=temp)

    assert _row_at(db_session).temp == 31.0


@pytest.mark.db
def test_a_feed_may_fill_a_gap_in_a_field_it_does_not_own(
    db_session: Session, db_settings: Settings, three_feeds: None
) -> None:
    """Deferring is not the same as being ignored: a value nobody authoritative
    has supplied is better than a NULL, which is why the fallback exists."""
    _run(db_session, db_settings, AIR, pm25=55.0, temp=20.0)

    assert _row_at(db_session).temp == 20.0

    _run(db_session, db_settings, WEATHER, temp=31.0)

    assert _row_at(db_session).temp == 31.0


@pytest.mark.db
def test_a_generated_row_never_merges_into_a_measured_one(
    db_session: Session, db_settings: Settings, three_feeds: None
) -> None:
    """The line the merge must not cross (ETH-1).

    The demo bundle describes the same centroids and the same hours as the live
    feeds. Merging on location and time alone would fold generated values into
    a row presented as measurement, which is the one blend no caveat can repair
    after the fact.
    """
    synthetic = _feed("Generated Feed", frozenset({"pm25"}))
    synthetic.spec = replace(synthetic.spec, synthetic=True)
    db_session.add(DataSource(name=synthetic.spec.name, is_synthetic=True))
    db_session.flush()

    _run(db_session, db_settings, AIR, pm25=55.0)
    _run(db_session, db_settings, synthetic, pm25=12.0)

    rows = {
        row.provenance: row.pm25
        for row in db_session.scalars(
            select(Observation).where(
                Observation.lat == STATION.lat, Observation.lon == STATION.lon
            )
        )
    }
    assert rows == {Provenance.MEASURED: 55.0, Provenance.SYNTHETIC: 12.0}
