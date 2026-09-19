"""Profile cache and dataset fingerprint (task 4.6, design §11).

design §11's requirement is specifically that results "never go stale
silently", so the tests that matter are the ones proving a cached entry
becomes unreachable the moment its data changes -- in each of the four ways
the data can change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from core.config import Settings
from db.models import DataSource, Observation, SourceStatus
from services.eda import cache
from services.eda.cache import DatasetVersion, ProfileCache

START = datetime(2026, 6, 1, tzinfo=timezone.utc)


# --- The cache itself (no database) -----------------------------------------


def _version(**overrides: object) -> DatasetVersion:
    base: dict[str, object] = {
        "rows": 100,
        "max_id": 500,
        "latest_timestamp": START,
        "flagged_rows": 3,
    }
    base.update(overrides)
    return DatasetVersion(**base)  # type: ignore[arg-type]


def test_a_stored_value_comes_back() -> None:
    store = ProfileCache()
    key = store.key(_version(), kind="profile")

    store.set(key, "result")

    assert store.get(key) == "result"


def test_a_miss_returns_none_and_is_counted() -> None:
    store = ProfileCache()

    assert store.get("absent") is None
    assert store.stats()["misses"] == 1


def test_hits_and_misses_are_tracked() -> None:
    store = ProfileCache()
    key = store.key(_version(), kind="profile")
    store.set(key, 1)

    store.get(key)
    store.get(key)
    store.get("absent")

    assert store.stats()["hits"] == 2
    assert store.stats()["misses"] == 1


def test_entries_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clock is driven rather than waited on.

    Sleeping past a TTL is flaky on Windows, where ``time.monotonic`` has a
    ~15 ms tick: a 10 ms sleep can still measure as zero elapsed. Stepping a
    fake clock tests the same branch in microseconds and never flakes.
    """
    now = [1_000.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])

    store = ProfileCache(ttl_seconds=60)
    key = store.key(_version(), kind="profile")
    store.set(key, "result")

    now[0] += 59
    assert store.get(key) == "result"

    now[0] += 2  # now 61s old, past the TTL
    assert store.get(key) is None


def test_the_least_recently_used_entry_is_evicted_first() -> None:
    store = ProfileCache(max_entries=2)
    store.set("a", 1)
    store.set("b", 2)
    store.get("a")  # touch a, so b becomes least recent
    store.set("c", 3)

    assert store.get("a") == 1
    assert store.get("b") is None
    assert store.get("c") == 3


def test_clearing_resets_everything() -> None:
    store = ProfileCache()
    store.set("a", 1)
    store.get("a")

    store.clear()

    assert store.stats() == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "max_entries": store.max_entries,
        "ttl_seconds": store.ttl_seconds,
    }


# --- Keys -------------------------------------------------------------------


def test_the_same_request_on_the_same_data_shares_a_key() -> None:
    store = ProfileCache()

    assert store.key(_version(), kind="profile", columns=["pm25"]) == store.key(
        _version(), kind="profile", columns=["pm25"]
    )


def test_a_different_request_on_the_same_data_gets_a_different_key() -> None:
    """The same data profiled over different columns is a different answer."""
    store = ProfileCache()

    assert store.key(_version(), kind="profile", columns=["pm25"]) != store.key(
        _version(), kind="profile", columns=["pm10"]
    )


def test_key_order_does_not_matter() -> None:
    store = ProfileCache()

    assert store.key(_version(), a=1, b=2) == store.key(_version(), b=2, a=1)


@pytest.mark.parametrize(
    "change",
    [
        {"rows": 101},
        {"max_id": 501},
        {"latest_timestamp": START + timedelta(hours=1)},
        {"flagged_rows": 4},
    ],
)
def test_any_change_to_the_data_changes_the_key(change: dict[str, object]) -> None:
    """design §11, the whole point: four ways the data can move, and none of
    them may leave a cached answer reachable.

    ``rows`` catches inserts and deletes, ``max_id`` catches an insert that
    replaced a delete, ``latest_timestamp`` catches a backfill landing earlier
    than the newest row, and ``flagged_rows`` catches a quality run that
    changed nothing but the judgement.
    """
    store = ProfileCache()
    original = store.key(_version(), kind="profile")

    assert store.key(_version(**change), kind="profile") != original


def test_the_fingerprint_is_short_and_stable() -> None:
    first = _version().fingerprint

    assert first == _version().fingerprint
    assert len(first) == 16


# --- The fingerprint against a live database --------------------------------


@pytest.mark.db
def test_the_fingerprint_reflects_the_database(
    db_session: Session, db_settings: Settings
) -> None:
    source = DataSource(
        name=f"cache-test-{datetime.now(timezone.utc).timestamp()}",
        status=SourceStatus.HEALTHY,
    )
    db_session.add(source)
    db_session.flush()

    before = cache.dataset_version(db_session, source_ids=(source.id,))
    assert before.rows == 0

    db_session.add(
        Observation(
            source_id=source.id, timestamp=START, lat=28.61, lon=77.21, pm25=50.0
        )
    )
    db_session.flush()

    after = cache.dataset_version(db_session, source_ids=(source.id,))

    assert after.rows == 1
    assert after.fingerprint != before.fingerprint


@pytest.mark.db
def test_flagging_a_row_moves_the_fingerprint(
    db_session: Session, db_settings: Settings
) -> None:
    """A quality run changes no measurement, so only the flag count catches it."""
    source = DataSource(
        name=f"cache-flag-{datetime.now(timezone.utc).timestamp()}",
        status=SourceStatus.HEALTHY,
    )
    db_session.add(source)
    db_session.flush()
    observation = Observation(
        source_id=source.id, timestamp=START, lat=28.61, lon=77.21, pm25=50.0
    )
    db_session.add(observation)
    db_session.flush()

    before = cache.dataset_version(db_session, source_ids=(source.id,))
    observation.is_anomaly = True
    db_session.flush()
    after = cache.dataset_version(db_session, source_ids=(source.id,))

    assert before.rows == after.rows
    assert after.flagged_rows == 1
    assert after.fingerprint != before.fingerprint


@pytest.mark.db
def test_the_fingerprint_respects_the_window(
    db_session: Session, db_settings: Settings
) -> None:
    source = DataSource(
        name=f"cache-window-{datetime.now(timezone.utc).timestamp()}",
        status=SourceStatus.HEALTHY,
    )
    db_session.add(source)
    db_session.flush()
    for hour in range(5):
        db_session.add(
            Observation(
                source_id=source.id,
                timestamp=START + timedelta(hours=hour),
                lat=28.61,
                lon=77.21,
                pm25=50.0 + hour,
            )
        )
    db_session.flush()

    whole = cache.dataset_version(db_session, source_ids=(source.id,))
    part = cache.dataset_version(
        db_session, source_ids=(source.id,), end=START + timedelta(hours=2)
    )

    assert whole.rows == 5
    assert part.rows == 3
    assert whole.fingerprint != part.fingerprint


@pytest.mark.db
def test_the_second_identical_profile_is_served_from_cache(
    db_session: Session, db_settings: Settings
) -> None:
    from services.eda import service

    cache.PROFILE_CACHE.clear()

    first = service.build_profile(db_session, db_settings)
    second = service.build_profile(db_session, db_settings)

    assert first.cached is False
    assert second.cached is True
    assert second.version.fingerprint == first.version.fingerprint


@pytest.mark.db
def test_the_cache_can_be_bypassed(
    db_session: Session, db_settings: Settings
) -> None:
    from services.eda import service

    cache.PROFILE_CACHE.clear()
    service.build_profile(db_session, db_settings)

    assert service.build_profile(db_session, db_settings, use_cache=False).cached is False


@pytest.mark.db
def test_new_data_is_never_served_from_a_stale_entry(
    db_session: Session, db_settings: Settings
) -> None:
    """The requirement in one test: write a row, and the previous profile must
    become unreachable rather than merely unlikely."""
    from services.eda import service

    cache.PROFILE_CACHE.clear()
    source = DataSource(
        name=f"cache-stale-{datetime.now(timezone.utc).timestamp()}",
        status=SourceStatus.HEALTHY,
    )
    db_session.add(source)
    db_session.flush()
    db_session.add(
        Observation(
            source_id=source.id, timestamp=START, lat=28.61, lon=77.21, pm25=10.0
        )
    )
    db_session.flush()

    first = service.build_profile(db_session, db_settings, source_ids=(source.id,))
    assert first.profile.rows == 1

    db_session.add(
        Observation(
            source_id=source.id,
            timestamp=START + timedelta(hours=1),
            lat=28.61,
            lon=77.21,
            pm25=90.0,
        )
    )
    db_session.flush()

    second = service.build_profile(db_session, db_settings, source_ids=(source.id,))

    assert second.cached is False
    assert second.profile.rows == 2
