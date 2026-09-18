"""Run the data quality pipeline (Phase 3).

    python -m scripts.run_quality --max-gap-hours 3

The same work as ``POST /api/v1/data/quality``, for a cron job or a container
shell. Repairs values, flags anomalies, and writes the cleaned frame plus its
report to ``data/processed`` -- and never removes a record (AC-5).
"""

from __future__ import annotations

import argparse
import sys

from core.config import get_settings
from db.session import session_scope
from services.quality import imputation, outliers, pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_quality",
        description="Impute missing values and flag anomalies (AC-4, AC-5).",
    )
    parser.add_argument(
        "--max-gap-hours",
        type=int,
        default=imputation.DEFAULT_MAX_GAP_HOURS,
        help="Longest gap filled locally before MICE takes over.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=outliers.DEFAULT_MIN_VOTES,
        help="Detectors that must agree before a row is flagged.",
    )
    parser.add_argument(
        "--no-backward-fill",
        action="store_true",
        help="Forward fill only. Use when preparing a training window (AC-8).",
    )
    parser.add_argument(
        "--no-persist", action="store_true", help="Skip writing to data/processed."
    )
    parser.add_argument(
        "--no-flags", action="store_true", help="Do not write is_anomaly back."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with session_scope(settings) as session:
        report = pipeline.run(
            session,
            settings,
            max_gap_hours=args.max_gap_hours,
            min_votes=args.min_votes,
            backward_fill=not args.no_backward_fill,
            persist=not args.no_persist,
            write_flags=not args.no_flags,
        )

    print("Data quality run complete:")
    print(f"  rows            {report.rows_in} -> {report.rows_out}")
    print(f"  rows preserved  {report.rows_preserved}   (AC-5)")
    print(f"  nulls remaining {report.imputation.remaining_nulls}")
    print(f"  feature set complete {report.feature_set_is_complete}   (AC-4)")
    print(f"  anomalies       {report.anomalies.flagged} ({report.anomalies.flagged_pct:.2f}%)")
    print(f"  flags written   {report.flags_written}")
    print()
    print("  missingness mechanisms:")
    for column in report.missingness.columns:
        print(
            f"    {column.column:<14} {column.missing:>6} missing "
            f"({column.missing_pct:5.2f}%)  -> {column.mechanism.value}"
        )
    if report.cleaned_path:
        print()
        print(f"  cleaned  {report.cleaned_path}")
        print(f"  report   {report.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
