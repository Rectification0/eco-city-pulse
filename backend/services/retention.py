"""Retention: bounding what grows without bound.

Three things in this system grow forever, and they are not the ones you would
guess. Measured on a real deployment:

* ``observations`` -- ~950 bytes a row including indexes, 264 rows a day for
  eleven stations. **Under 100 MB a year**, and it is the training data. There
  is deliberately no policy for it here; see ``core.config``.
* the artifact directory -- **~46 MB per training run**, because a random
  forest pickles to roughly 22 MB and the ladder trains one per horizon.
  Retrained daily that is 17 GB a year. This is the one that matters.
* ``ingestion_runs`` and ``predictions`` -- small per row, but unbounded, and
  one of them is written on every forecast.

Two rules shape everything below.

**Retention deletes artifacts, never ``models`` rows.** The row holds the
metrics, the feature list and the date: the record of what was trained and how
well it did, which is the part worth keeping and the part that costs nothing.
The file is the weights. ``predictions.model_id`` is ``ON DELETE CASCADE``, so
deleting rows would take the drift dataset (design §6.1) with them silently --
an audit trail destroyed as a side effect of tidying up is the worst possible
version of this feature. ``registry.load_artifact`` already answers a missing
file with ``ModelNotFoundError``, so a pruned artifact has a defined failure
mode rather than a stack trace.

**Nothing here deletes a measurement.** The quality engine's rule is flag,
never delete (AC-4, AC-5); this module holds to the same line for the same
reason. The only observations it would ever touch are none.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from db.models import IngestionRun, MLModel, Prediction

ARTIFACT_SUFFIX = ".joblib"


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What one sweep removed, and what it would have removed on a dry run."""

    name: str
    removed: int = 0
    bytes_freed: int = 0
    kept: int = 0
    detail: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "removed": self.removed,
            "bytes_freed": self.bytes_freed,
            "kept": self.kept,
            "detail": list(self.detail),
        }


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """Every sweep from one pass, and whether anything was actually deleted."""

    dry_run: bool
    sweeps: tuple[SweepResult, ...] = field(default_factory=tuple)

    @property
    def bytes_freed(self) -> int:
        return sum(sweep.bytes_freed for sweep in self.sweeps)

    @property
    def removed(self) -> int:
        return sum(sweep.removed for sweep in self.sweeps)

    def as_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "removed": self.removed,
            "bytes_freed": self.bytes_freed,
            "sweeps": [sweep.as_dict() for sweep in self.sweeps],
        }


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:  # pragma: no cover - racing another process
        return 0


def artifacts_to_keep(session: Session, settings: Settings) -> set[Path]:
    """The artifact files retention must not touch.

    The newest ``artifact_keep_per_model`` per ``(target, name)``. Grouping by
    both, rather than by target alone, is what guarantees the production model
    survives: ``serving.load_latest`` resolves a target to its newest row of
    *any* name, and that row is necessarily the newest of its own name too, so
    keeping at least one per group keeps it.

    Returned as resolved paths because the same file reached by two different
    spellings -- a relative row written before a deployment moved, an absolute
    one after -- must compare equal, or a sweep deletes a file it meant to
    keep.
    """
    keep_count = max(1, settings.artifact_keep_per_model)

    grouped: dict[tuple[str, str], list[MLModel]] = defaultdict(list)
    for row in session.scalars(select(MLModel).order_by(MLModel.created_at.desc())):
        grouped[(row.target, row.name)].append(row)

    keep: set[Path] = set()
    for rows in grouped.values():
        for row in rows[:keep_count]:
            keep.add(Path(row.artifact_path).resolve())
    return keep


def sweep_artifacts(
    session: Session, settings: Settings | None = None, *, dry_run: bool = False
) -> SweepResult:
    """Delete superseded and orphaned artifacts from the artifact directory.

    Two kinds of file go:

    * **superseded** -- referenced by a ``models`` row that is no longer among
      the newest few for its target and name. The row stays.
    * **orphaned** -- in the directory and referenced by no row at all. These
      accumulate from runs whose transaction rolled back, which includes every
      ``db``-marked test that trains a ladder.

    The directory is only ever scanned for ``*.joblib``, so nothing else that
    happens to share the folder is at risk.
    """
    settings = settings or get_settings()
    directory = settings.model_artifact_path
    if not directory.exists():
        return SweepResult(name="artifacts")

    keep = artifacts_to_keep(session, settings)
    removed = 0
    freed = 0
    detail: list[str] = []

    for path in sorted(directory.glob(f"*{ARTIFACT_SUFFIX}")):
        if path.resolve() in keep:
            continue
        size = _size(path)
        detail.append(path.name)
        removed += 1
        freed += size
        if not dry_run:
            path.unlink(missing_ok=True)

    return SweepResult(
        name="artifacts",
        removed=removed,
        bytes_freed=freed,
        kept=len(keep),
        detail=tuple(detail),
    )


def sweep_run_log(
    session: Session,
    settings: Settings | None = None,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> SweepResult:
    """Drop ingestion runs older than the retention window.

    ``quarantined_records`` hangs off ``ingestion_runs`` with ``ON DELETE
    CASCADE``, so the quarantined payloads of a dropped run go with it. That is
    intended: a payload is evidence for the run that rejected it, and keeping
    it after the run is gone leaves a rejection nobody can trace.
    """
    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=settings.run_log_retention_days)

    doomed = list(
        session.scalars(select(IngestionRun.id).where(IngestionRun.started_at < cutoff))
    )
    if doomed and not dry_run:
        session.execute(delete(IngestionRun).where(IngestionRun.id.in_(doomed)))

    kept = session.scalar(
        select(IngestionRun.id).where(IngestionRun.started_at >= cutoff).limit(1)
    )
    return SweepResult(
        name="ingestion_runs",
        removed=len(doomed),
        kept=0 if kept is None else 1,
        detail=(f"older than {cutoff.isoformat()}",),
    )


def sweep_unscored_predictions(
    session: Session,
    settings: Settings | None = None,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> SweepResult:
    """Drop forecasts whose hour passed long ago without ever being scored.

    A prediction earns its place by being compared against what happened. One
    still missing an ``actual_value`` a month after its target hour is not
    waiting for anything -- the observation it needed never arrived -- and it
    will sit in the pending scan of every future backfill.

    **Scored rows are never removed.** They are the drift-monitoring dataset,
    the reason design §6.1 asks for the table at all, and they are the only
    record of how a deployed model behaved on data it never trained on.
    """
    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=settings.unscored_prediction_retention_days)

    condition = (Prediction.actual_value.is_(None)) & (Prediction.target_time < cutoff)
    doomed = list(session.scalars(select(Prediction.id).where(condition)))
    if doomed and not dry_run:
        session.execute(delete(Prediction).where(Prediction.id.in_(doomed)))

    scored = (
        session.scalar(
            select(Prediction.id).where(Prediction.actual_value.is_not(None)).limit(1)
        )
        is not None
    )
    return SweepResult(
        name="unscored_predictions",
        removed=len(doomed),
        kept=1 if scored else 0,
        detail=(f"unscored and older than {cutoff.isoformat()}",),
    )


def run(
    session: Session,
    settings: Settings | None = None,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    """Every sweep, in one pass. Nothing here touches ``observations``."""
    settings = settings or get_settings()
    return RetentionReport(
        dry_run=dry_run,
        sweeps=(
            sweep_artifacts(session, settings, dry_run=dry_run),
            sweep_run_log(session, settings, now=now, dry_run=dry_run),
            sweep_unscored_predictions(session, settings, now=now, dry_run=dry_run),
        ),
    )


__all__ = [
    "ARTIFACT_SUFFIX",
    "RetentionReport",
    "SweepResult",
    "artifacts_to_keep",
    "run",
    "sweep_artifacts",
    "sweep_run_log",
    "sweep_unscored_predictions",
]
