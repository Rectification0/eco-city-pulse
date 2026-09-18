"""Load the Demo-mode dataset into PostgreSQL (task 1.9, DR-1, AC-2).

    python -m scripts.seed_demo --days 120

Runs with no network access: every value comes from ``services.demo_data``.
The script is idempotent -- observations are written with ON CONFLICT DO
NOTHING against the ``(source_id, timestamp, lat, lon)`` unique constraint, so
re-running extends the history rather than duplicating it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from db.models import DataSource, Observation, SourceStatus
from db.session import session_scope
from services import demo_data

# Rows per INSERT. Large enough that the round trips disappear, small enough
# that a batch's parameters stay well inside the driver's limits.
BATCH_SIZE = 2_000

CSV_COLUMNS = (
    "timestamp",
    "district_id",
    "lat",
    "lon",
    "pm25",
    "pm10",
    "temp",
    "humidity",
    "traffic_score",
)


def ensure_sources(session: Session) -> DataSource:
    """Register the demo bundle and the three live adapters; return the bundle.

    The live sources are recorded as ``offline`` rather than omitted. With no
    API key that is their true state, and the distinction is what task 2.10
    reports as ingestion health.
    """
    for name, api_url in demo_data.LIVE_SOURCES:
        existing = session.scalar(select(DataSource).where(DataSource.name == name))
        if existing is None:
            session.add(
                DataSource(name=name, api_url=api_url, status=SourceStatus.OFFLINE)
            )

    bundle = session.scalar(
        select(DataSource).where(DataSource.name == demo_data.DEMO_SOURCE_NAME)
    )
    if bundle is None:
        bundle = DataSource(
            name=demo_data.DEMO_SOURCE_NAME,
            api_url=None,  # no endpoint: that is the point of demo mode
            status=SourceStatus.HEALTHY,
        )
        session.add(bundle)

    session.flush()  # assign bundle.id before observations reference it
    return bundle


def _batched(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _as_mapping(record: demo_data.DemoObservation, source_id: int) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "timestamp": record.timestamp,
        "lat": record.lat,
        "lon": record.lon,
        "pm25": record.pm25,
        "pm10": record.pm10,
        "temp": record.temp,
        "humidity": record.humidity,
        "traffic_score": record.traffic_score,
        # Anomaly detection is Phase 3's job; the seed must not pre-judge it.
        "is_anomaly": False,
    }


def _count_for_source(session: Session, source_id: int) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(Observation)
            .where(Observation.source_id == source_id)
        )
        or 0
    )


def insert_observations(
    session: Session, records: Iterable[demo_data.DemoObservation], source_id: int
) -> int:
    """Bulk-insert, skipping rows already present. Returns the number written.

    The count comes from a before/after difference rather than ``rowcount``:
    with executemany plus ON CONFLICT, SQLAlchemy returns a result object that
    does not carry a meaningful affected-row count.
    """
    is_postgres = session.get_bind().dialect.name == "postgresql"
    before = _count_for_source(session, source_id)

    for batch in _batched((_as_mapping(r, source_id) for r in records), BATCH_SIZE):
        if is_postgres:
            statement = pg_insert(Observation).on_conflict_do_nothing(
                constraint="uq_observations_source_timestamp_location"
            )
        else:
            statement = Observation.__table__.insert()
        session.execute(statement, batch)

    session.flush()
    return _count_for_source(session, source_id) - before


def reset_demo_observations(session: Session, source_id: int) -> int:
    """Delete this source's observations. Other sources are left untouched."""
    deleted = session.query(Observation).filter(Observation.source_id == source_id).delete(
        synchronize_session=False
    )
    return int(deleted)


def write_csv(path: Path, records: Iterable[demo_data.DemoObservation]) -> int:
    """Dump the same dataset to the raw landing zone (design §5).

    Gives the Phase 2 upload path (task 2.3) a real file to parse, and makes the
    demo data inspectable without a database client.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for record in records:
            writer.writerow(
                [
                    record.timestamp.isoformat(),
                    record.district_id,
                    record.lat,
                    record.lon,
                    "" if record.pm25 is None else record.pm25,
                    "" if record.pm10 is None else record.pm10,
                    "" if record.temp is None else record.temp,
                    "" if record.humidity is None else record.humidity,
                    "" if record.traffic_score is None else record.traffic_score,
                ]
            )
            count += 1
    return count


def seed(
    *,
    days: int = demo_data.DEFAULT_DAYS,
    seed_value: int = demo_data.DEFAULT_SEED,
    end: datetime | None = None,
    reset: bool = False,
    csv_path: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Seed the database and report what happened."""
    settings = settings or get_settings()
    end = demo_data.floor_to_hour(end or datetime.now(timezone.utc))

    stations = demo_data.stations_from_districts(settings)
    summary: dict[str, Any] = {
        "days": days,
        "end": end.isoformat(),
        "stations": len(stations),
        "seed": seed_value,
    }

    with session_scope(settings) as session:
        bundle = ensure_sources(session)
        assert bundle.id is not None

        summary["deleted"] = (
            reset_demo_observations(session, bundle.id) if reset else 0
        )

        records = demo_data.generate_observations(
            days=days, end=end, stations=stations, seed=seed_value, settings=settings
        )
        summary["inserted"] = insert_observations(session, records, bundle.id)

        # last_run is the ingestion-health signal /data/sources reports.
        bundle.last_run = end
        bundle.status = SourceStatus.HEALTHY

        summary["total_rows"] = session.scalar(
            select(func.count()).select_from(Observation)
        )

    if csv_path is not None:
        summary["csv_path"] = str(csv_path)
        summary["csv_rows"] = write_csv(
            csv_path,
            demo_data.generate_observations(
                days=days, end=end, stations=stations, seed=seed_value, settings=settings
            ),
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.seed_demo",
        description="Load the offline Demo-mode dataset (DR-1, AC-2).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=demo_data.DEFAULT_DAYS,
        help=f"Hours of history to generate, in days (default: {demo_data.DEFAULT_DAYS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=demo_data.DEFAULT_SEED,
        help="Generator seed; the same seed always yields the same values.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the demo bundle's existing observations first.",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also write the dataset to CSV (default: data/raw/demo_observations.csv).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    csv_path: Path | None = None
    if args.csv is not None:
        csv_path = (
            Path(args.csv)
            if args.csv
            else settings.data_raw_path / "demo_observations.csv"
        )

    summary = seed(
        days=args.days,
        seed_value=args.seed,
        reset=args.reset,
        csv_path=csv_path,
        settings=settings,
    )

    print("Demo seed complete (offline, no API keys used):")
    for key, value in summary.items():
        print(f"  {key:<12} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
