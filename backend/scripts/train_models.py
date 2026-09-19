"""Train and register the model ladder (Phase 7).

    python -m scripts.train_models                    # all three horizons
    python -m scripts.train_models --horizon 1 --no-classical    # fast

Prints the comparison that matters: every model's MAE beside the persistence
baseline's, because an MAE in µg/m³ means nothing without the floor it is
supposed to clear (AC-7).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from core.config import get_settings
from db.session import session_scope
from services.ml import targets, training


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.train_models",
        description="Train Naive/Ridge/RF/XGBoost per horizon and register them.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        dest="horizons",
        help=f"Hours ahead. Repeat for several. Default: {list(targets.DEFAULT_HORIZONS)}.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Share of the period held out, chronologically (AC-8).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=training.DEFAULT_CV_SPLITS,
        help="Expanding-window folds over the training portion.",
    )
    parser.add_argument("--start", type=_timestamp, default=None, help="ISO timestamp.")
    parser.add_argument("--end", type=_timestamp, default=None, help="ISO timestamp.")
    parser.add_argument(
        "--no-classical",
        action="store_true",
        help="Skip the ARIMA/Prophet baselines, which dominate the runtime.",
    )
    parser.add_argument(
        "--no-prophet", action="store_true", help="Run ARIMA but not Prophet."
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Score without writing artifacts or registry rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with session_scope(settings) as session:
        report = training.run(
            session,
            settings,
            horizons=tuple(args.horizons) if args.horizons else targets.DEFAULT_HORIZONS,
            start=args.start,
            end=args.end,
            test_fraction=args.test_fraction,
            cv_splits=args.cv_splits,
            include_classical=not args.no_classical,
            include_prophet=not args.no_prophet,
            register=not args.no_register,
        )

        print("Training complete.")
        print(f"  window        {report.window.rows} rows, {report.window.stations} stations")
        print(f"  feature spec  {report.transformer.fingerprint}")
        print(f"  leakage clean {report.leakage_clean}   (AC-8)")
        print(f"  beats baseline {report.beats_baseline}   (AC-7, 1h horizon)")

        for horizon in report.horizons:
            print()
            print(f"  --- {horizon.target} ({horizon.horizon}h ahead) ---")
            print(
                f"  train ends {horizon.leakage_audit['train_end']} -> "
                f"test starts {horizon.leakage_audit['test_start']} "
                f"(gap {horizon.leakage_audit['gap_hours']}h)"
            )
            print(
                f"  features: {horizon.selection['kept_count']} kept, "
                f"{horizon.selection['dropped_count']} dropped"
            )
            print(f"  {'model':<16}{'MAE':>9}{'RMSE':>9}{'R2':>8}{'skill':>9}")
            for model in horizon.models:
                if model.unavailable:
                    print(f"  {model.name:<16}{'unavailable':>35}")
                    continue
                skill = (
                    f"{model.skill_vs_baseline:+.1%}"
                    if model.skill_vs_baseline is not None
                    else "-"
                )
                print(
                    f"  {model.name:<16}{model.metrics.mae:>9.3f}"
                    f"{model.metrics.rmse:>9.3f}{model.metrics.r2:>8.3f}{skill:>9}"
                )
            for baseline in horizon.classical:
                if baseline.metrics:
                    print(
                        f"  {baseline.name + ' (uni)':<16}"
                        f"{baseline.metrics.mae:>9.3f}{baseline.metrics.rmse:>9.3f}"
                        f"{baseline.metrics.r2:>8.3f}{'-':>9}"
                    )
                else:
                    print(f"  {baseline.name + ' (uni)':<16}{baseline.note:>35}")

        registered = [
            model.registered
            for horizon in report.horizons
            for model in horizon.models
            if model.registered
        ]
        if registered:
            print()
            print(f"  registered {len(registered)} model(s):")
            for entry in registered:
                print(f"    #{entry.model_id} {entry.target:<10} {entry.name:<14} {entry.artifact_path}")

    return 0 if report.leakage_clean else 1


if __name__ == "__main__":
    sys.exit(main())
