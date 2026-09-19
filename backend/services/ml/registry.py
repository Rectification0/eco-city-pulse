"""Model registry (task 7.12, FEAT-05, specs §9).

A trained model that exists only in the process that trained it is not a
deliverable. Registration is two writes that must agree:

1. **The artifact** -- the fitted estimator, joblib-pickled to
   ``settings.model_artifact_path``, together with everything needed to rebuild
   its input: the feature spec, the selected columns in order, the horizon, and
   the metrics it earned. Phase 9 loads exactly this and needs nothing else.
2. **The row** in ``models`` -- name, target, ``features_used``, MAE/RMSE/R²
   and ``artifact_path``, which is what the Model Lab reads (task 10.12).

**The artifact carries its own feature contract.** A model file and a feature
spec that have drifted apart produce confident nonsense: the same column names
in a different order, or a log transform applied at training and not at serving.
So the Phase 5 transformer's fingerprint is stored in the artifact, and serving
compares it before predicting rather than trusting that nothing changed.

**Registration is append-only.** A retrain writes a new row and a new file;
``ix_models_target_created_at`` makes "the current model for this target" the
newest row. Overwriting would destroy the record of what was deployed when,
which is the only thing that makes a drift investigation possible later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from core.exceptions import ModelNotFoundError
from db.models import MLModel

ARTIFACT_SUFFIX = ".joblib"

# A 200-tree forest fitted on 18k rows pickles to about 52 MB uncompressed and
# 17 MB at this level, for well under a second of CPU either way. Registration
# is append-only, so every retrain adds another copy -- the ratio matters more
# here than the milliseconds.
COMPRESSION_LEVEL = 3

# Anything outside this becomes an underscore before it reaches a path.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """What a registration produced."""

    model_id: int
    name: str
    target: str
    artifact_path: str
    mae: float | None
    rmse: float | None
    r2: float | None
    created_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "target": self.target,
            "artifact_path": self.artifact_path,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "created_at": self.created_at.isoformat(),
        }


def artifact_filename(name: str, target: str, created_at: datetime) -> str:
    """A filename that sorts by time and says what it holds.

    The name is sanitised because it reaches the filesystem: a model name is
    developer-supplied, and a path separator inside one would write outside the
    artifact directory.
    """
    stamp = created_at.strftime("%Y%m%dT%H%M%S")
    safe_name = _SAFE_NAME.sub("_", name)
    safe_target = _SAFE_NAME.sub("_", target)
    return f"{safe_target}__{safe_name}__{stamp}{ARTIFACT_SUFFIX}"


def save_artifact(
    payload: dict[str, Any],
    *,
    name: str,
    target: str,
    created_at: datetime,
    settings: Settings | None = None,
) -> Path:
    """Write the estimator and its contract to the artifact directory."""
    settings = settings or get_settings()
    directory = settings.model_artifact_path
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / artifact_filename(name, target, created_at)
    joblib.dump(payload, path, compress=COMPRESSION_LEVEL)
    return path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Read an artifact back (what Phase 9 calls).

    A missing file is a domain 404 rather than an OSError: the usual cause is a
    registry row pointing at an artifact directory that was not mounted, and
    that deserves a clear answer rather than a stack trace.
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Model artifact not found at {path}.", details={"artifact_path": str(path)}
        )
    return joblib.load(path)


def register(
    session: Session,
    *,
    name: str,
    target: str,
    features_used: list[str],
    artifact: dict[str, Any],
    mae: float | None = None,
    rmse: float | None = None,
    r2: float | None = None,
    settings: Settings | None = None,
    created_at: datetime | None = None,
) -> RegisteredModel:
    """Persist artifact then row, so a row never points at a missing file."""
    settings = settings or get_settings()
    created_at = created_at or datetime.now(timezone.utc)

    path = save_artifact(
        artifact, name=name, target=target, created_at=created_at, settings=settings
    )

    row = MLModel(
        name=name,
        target=target,
        features_used=list(features_used),
        mae=mae,
        rmse=rmse,
        r2=r2,
        artifact_path=str(path),
        created_at=created_at,
    )
    session.add(row)
    session.flush()

    return RegisteredModel(
        model_id=row.id,
        name=row.name,
        target=row.target,
        artifact_path=row.artifact_path,
        mae=row.mae,
        rmse=row.rmse,
        r2=row.r2,
        created_at=row.created_at,
    )


def latest_for_target(
    session: Session, target: str, *, name: str | None = None
) -> MLModel | None:
    """The current model for a target: the newest row, per design §6.1.

    ``name`` narrows it to one rung of the ladder. Serving passes the production
    model's name rather than taking whatever was registered last: a training run
    registers the whole ladder so the Model Lab can compare it, and "newest row"
    alone would make the choice of served model an accident of iteration order.
    """
    statement = select(MLModel).where(MLModel.target == target)
    if name:
        statement = statement.where(MLModel.name == name)
    return session.scalars(
        statement.order_by(MLModel.created_at.desc(), MLModel.id.desc()).limit(1)
    ).first()


def list_models(
    session: Session, *, target: str | None = None, limit: int = 100
) -> list[MLModel]:
    """The registry, newest first -- what the Model Lab table renders."""
    statement = select(MLModel).order_by(MLModel.created_at.desc(), MLModel.id.desc())
    if target:
        statement = statement.where(MLModel.target == target)
    return list(session.scalars(statement.limit(limit)))


__all__ = [
    "ARTIFACT_SUFFIX",
    "RegisteredModel",
    "artifact_filename",
    "latest_for_target",
    "list_models",
    "load_artifact",
    "register",
    "save_artifact",
]
