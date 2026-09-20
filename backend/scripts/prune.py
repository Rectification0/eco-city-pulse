"""Bound what grows without bound (retention).

    python -m scripts.prune --dry-run
    python -m scripts.prune

Sweeps superseded and orphaned model artifacts, ingestion runs past their
window, and forecasts whose hour passed without ever being scored.

**It never deletes an observation and never deletes a ``models`` row.** The
measurements are the training data and the quality engine's rule is flag, never
delete (AC-4, AC-5); the model rows carry the metrics and are what
``predictions`` hangs off with ``ON DELETE CASCADE``. What this reclaims is the
artifact directory, which grows by roughly 46 MB per training run and is the
only part of the system that grows fast enough to matter.

``--dry-run`` first is the habit worth having: it reports exactly what would go,
and deletes nothing.
"""

from __future__ import annotations

import argparse
import sys

from core.config import get_settings
from db.session import session_scope
from services import retention


def _megabytes(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} MB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.prune",
        description="Delete superseded artifacts and expired log rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted, delete nothing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List every file and rule, not just the totals.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with session_scope(settings) as session:
        report = retention.run(session, settings, dry_run=args.dry_run)

    print("Dry run — nothing deleted:" if args.dry_run else "Retention sweep:")
    for sweep in report.sweeps:
        line = f"  {sweep.name:<22s} removed={sweep.removed:<5d}"
        if sweep.bytes_freed:
            line += f" freed={_megabytes(sweep.bytes_freed):>10s}"
        if sweep.name == "artifacts":
            line += f"  kept={sweep.kept}"
        print(line)
        if args.verbose:
            for item in sweep.detail:
                print(f"      {item}")

    print(f"  {'total':<22s} removed={report.removed:<5d} "
          f"freed={_megabytes(report.bytes_freed)}")
    print("\n  observations: untouched by design (AC-4, AC-5).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
