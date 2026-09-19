"""Build the engineered feature set (Phase 5).

    python -m scripts.build_features --max-gap-hours 3

Reads ``observations``, repairs short gaps locally, applies the feature spec,
and writes ``features.csv`` plus the transformer and manifest that describe it
to ``data/processed`` (task 5.6).

``--end`` matters more than it looks: fitting the log-transform decision on data
the model will later be tested on is a leak, so a training run should stop the
window where the training period stops (AC-8).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from core.config import get_settings
from db.session import session_scope
from services.features import service


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.build_features",
        description="Engineer temporal, lag, rolling and log features (Phase 5).",
    )
    parser.add_argument(
        "--max-gap-hours",
        type=int,
        default=service.DEFAULT_MAX_GAP_HOURS,
        help="Longest gap repaired by forward fill before features go NaN.",
    )
    parser.add_argument(
        "--start", type=_timestamp, default=None, help="ISO timestamp, inclusive."
    )
    parser.add_argument(
        "--end",
        type=_timestamp,
        default=None,
        help="ISO timestamp, inclusive. Stop at the end of the training period.",
    )
    parser.add_argument(
        "--source-id",
        type=int,
        action="append",
        dest="source_ids",
        help="Restrict to one source. Repeat for several.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Compute without writing to data/processed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with session_scope(settings) as session:
        result = service.build_features(
            session,
            settings,
            source_ids=tuple(args.source_ids) if args.source_ids else None,
            start=args.start,
            end=args.end,
            max_gap_hours=args.max_gap_hours,
            persist=not args.no_persist,
        )

    spec = result.transformer.spec
    print("Feature build complete:")
    print(f"  rows            {result.rows}")
    print(f"  complete rows   {result.complete_rows} ({result.complete_pct}%)")
    print(f"  stations        {result.window.stations}")
    print(f"  history needed  {spec.history_hours}h per row")
    print(f"  spec            {result.transformer.fingerprint}")
    print()
    print(f"  features ({len(spec.feature_names)}):")
    for name in spec.feature_names:
        nulls = int(result.frame[name].isna().sum()) if name in result.frame else 0
        print(f"    {name:<32} {nulls:>6} null")
    print()
    print(f"  short gaps forward-filled: {result.repaired}")
    if spec.log_columns:
        print(f"  log transform applied to:  {list(spec.log_columns)}")
    else:
        print("  log transform applied to:  none (no pollutant skewed enough)")
    if result.stored:
        print()
        print(f"  features {result.stored.features_path}")
        print(f"  spec     {result.stored.spec_path}")
        print(f"  manifest {result.stored.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
