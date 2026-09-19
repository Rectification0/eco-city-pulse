"""The feature store (task 5.6, design §2).

design §2 lists a "Feature store" in the intelligence layer. Here it is three
files in ``data/processed``, and the modesty is deliberate -- at this scale a
dedicated store (Feast, a warehouse table) would add an service to operate for
no capability the platform uses.

    features.csv            the engineered frame
    feature_spec.json       the fitted transformer that produced it
    feature_manifest.json   what it covers, and how complete it is

**The spec travels with the data.** A feature file without the definition that
built it is a set of unlabelled numbers: ``pm25_rolling_mean_24h`` means
something different if the window coverage rule changed. Writing them together
means a reader can always reconstruct the transform, and Phase 7 can assert that
the transformer it is about to serve with is the one the file was built from
(``fingerprint``).

**CSV, like Phase 3.** Parquet would be smaller and would carry dtypes, at the
cost of a pyarrow dependency for a dataset measured in megabytes. The cleaned
frame of task 3.8 is CSV for the same reason, and a consistent processed
directory is worth more here than a faster read.

**Not a cache.** Nothing reads these files to answer an API request; they are an
artefact for inspection, for the notebook-shaped work the syllabus asks about,
and for handing a reproducible matrix to the next phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings, get_settings
from core.exceptions import DatasetNotFoundError
from services.datasets import DatasetWindow
from services.features.spec import FeatureSpec
from services.features.transformer import FeatureTransformer, complete_mask

FEATURES_FILENAME = "features.csv"
SPEC_FILENAME = "feature_spec.json"
MANIFEST_FILENAME = "feature_manifest.json"


@dataclass(frozen=True, slots=True)
class StoredFeatures:
    """Where a build landed, and what it contains."""

    features_path: str
    spec_path: str
    manifest_path: str
    rows: int
    complete_rows: int
    generated_at: datetime

    @property
    def complete_pct(self) -> float:
        """Share of rows with every feature present.

        Below 100% is normal and not a fault: the first ``history_hours`` of
        every station can never have a 24-hour lag. It is reported because a
        *sharp* drop means something else -- a gap the local repair could not
        bridge -- and that is worth seeing.
        """
        return round(100.0 * self.complete_rows / self.rows, 2) if self.rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": self.features_path,
            "spec": self.spec_path,
            "manifest": self.manifest_path,
            "rows": self.rows,
            "complete_rows": self.complete_rows,
            "complete_pct": self.complete_pct,
            "generated_at": self.generated_at.isoformat(),
        }


def paths(settings: Settings | None = None) -> tuple[Path, Path, Path]:
    settings = settings or get_settings()
    directory = settings.data_processed_path
    return (
        directory / FEATURES_FILENAME,
        directory / SPEC_FILENAME,
        directory / MANIFEST_FILENAME,
    )


def write(
    frame: pd.DataFrame,
    transformer: FeatureTransformer,
    *,
    window: DatasetWindow,
    settings: Settings | None = None,
    dataset_version: dict[str, Any] | None = None,
) -> StoredFeatures:
    """Persist the frame, the transformer and the manifest together."""
    settings = settings or get_settings()
    features_path, spec_path, manifest_path = paths(settings)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    complete = int(complete_mask(frame, transformer.spec).sum())

    frame.to_csv(features_path, index=False)
    transformer.save(spec_path)

    manifest = {
        "generated_at": generated_at.isoformat(),
        "rows": len(frame),
        "complete_rows": complete,
        "window": window.as_dict(),
        "dataset_version": dataset_version,
        "fingerprint": transformer.fingerprint,
        "features": list(transformer.feature_names),
        # Per-feature null counts: the quickest read on whether a window is
        # thin. A lag column with far more nulls than the warm-up explains is
        # a gap, not an artefact of the series start.
        "nulls": {
            name: int(frame[name].isna().sum())
            for name in transformer.feature_names
            if name in frame.columns
        },
        "files": {
            "features": FEATURES_FILENAME,
            "spec": SPEC_FILENAME,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    return StoredFeatures(
        features_path=str(features_path),
        spec_path=str(spec_path),
        manifest_path=str(manifest_path),
        rows=len(frame),
        complete_rows=complete,
        generated_at=generated_at,
    )


def load_transformer(settings: Settings | None = None) -> FeatureTransformer:
    """The transformer that built the stored feature set (task 5.5).

    Phase 7 trains from the stored frame and Phase 9 serves from this object, so
    both sides of the reproducibility guarantee come from one file.
    """
    _, spec_path, _ = paths(settings)
    if not spec_path.exists():
        raise DatasetNotFoundError(
            "No feature spec has been written yet. Run: python -m scripts.build_features",
            details={"expected": str(spec_path)},
        )
    return FeatureTransformer.load(spec_path)


def read(settings: Settings | None = None) -> tuple[pd.DataFrame, FeatureTransformer]:
    """Read the stored feature set back, with dtypes restored.

    CSV loses the timezone-aware dtype and the float/int distinction, so both
    are put back explicitly here rather than left for the caller to discover.
    """
    features_path, _, _ = paths(settings)
    if not features_path.exists():
        raise DatasetNotFoundError(
            "No feature set has been written yet. Run: python -m scripts.build_features",
            details={"expected": str(features_path)},
        )

    transformer = load_transformer(settings)
    frame = pd.read_csv(features_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    for name in transformer.feature_names:
        if name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    return frame, transformer


def read_manifest(settings: Settings | None = None) -> dict[str, Any]:
    _, _, manifest_path = paths(settings)
    if not manifest_path.exists():
        raise DatasetNotFoundError(
            "No feature manifest has been written yet.",
            details={"expected": str(manifest_path)},
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def stored_spec(settings: Settings | None = None) -> FeatureSpec:
    return load_transformer(settings).spec


__all__ = [
    "FEATURES_FILENAME",
    "MANIFEST_FILENAME",
    "SPEC_FILENAME",
    "StoredFeatures",
    "load_transformer",
    "paths",
    "read",
    "read_manifest",
    "stored_spec",
    "write",
]
