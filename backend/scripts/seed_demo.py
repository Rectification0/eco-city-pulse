"""Load the Demo-mode dataset (task 1.9, DR-1, AC-2).

    python -m scripts.seed_demo --days 120

Runs with no network access: every value comes from ``services.demo_data``.

Since Phase 2 this script owns no database logic of its own. It parses
arguments and calls ``ingestion_service.run_demo``, so the demo data enters
through exactly the same fetch → validate → harmonize → upsert → log pipeline
as a live API (task 2.7's "single write path"). Re-running is idempotent: the
upsert updates the hour in place rather than duplicating it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from core.config import Settings, get_settings
from db.models import Observation
from db.session import session_scope
from services import demo_data, ingestion_service

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


def write_csv(path: Path, records: Iterable[demo_data.DemoObservation]) -> int:
    """Dump the dataset to the raw landing zone (design §5).

    Gives the upload path (task 2.3) a real file to parse, and makes the demo
    data inspectable without a database client.
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
    csv_path: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Seed the database through the ingestion engine and report the outcome."""
    settings = settings or get_settings()
    end = demo_data.floor_to_hour(end or datetime.now(timezone.utc))

    summary: dict[str, Any] = {"days": days, "end": end.isoformat(), "seed": seed_value}

    with session_scope(settings) as session:
        outcomes = ingestion_service.run_demo(
            session, settings, days=days, end=end, seed=seed_value
        )
        outcome = outcomes[0]
        summary.update(
            status=outcome.status.value,
            fetched=outcome.records_fetched,
            valid=outcome.records_valid,
            quarantined=outcome.records_quarantined,
            written=outcome.records_written,
            run_id=outcome.run_id,
            message=outcome.message,
        )
        summary["total_rows"] = session.scalar(
            select(func.count()).select_from(Observation)
        )

    if csv_path is not None:
        summary["csv_path"] = str(csv_path)
        summary["csv_rows"] = write_csv(
            csv_path,
            demo_data.generate_observations(
                days=days, end=end, seed=seed_value, settings=settings
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
        days=args.days, seed_value=args.seed, csv_path=csv_path, settings=settings
    )

    print("Demo seed complete (offline, no API keys used):")
    for key, value in summary.items():
        print(f"  {key:<12} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
