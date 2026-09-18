"""The cleaning pipeline (tasks 3.7, 3.8; AC-4, AC-5).

Ties the three stages together and is the only thing that writes back:

    load → analyse missingness → impute → detect anomalies → flag → persist

**Ownership of ``observations``.** Phase 2 established ingestion as the single
write path for the measurement columns, and it deliberately never touches
``is_anomaly``. This module owns exactly that one column and nothing else, so
the two never fight: ingestion records what arrived, the quality engine records
what it thinks of it.

**Row counts are invariant.** Rows in equals rows out, at every stage. AC-5 is
not "mostly keep the outliers" -- it is that the pipeline never removes a
record, and the report carries the counts to prove it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import update
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from db.models import Observation
from services import datasets
from services.datasets import MEASUREMENT_COLUMNS, DatasetWindow
from services.quality import imputation, missingness, outliers

# Rows per UPDATE when writing flags back.
FLAG_BATCH_SIZE = 5_000

CLEANED_FILENAME = "observations_cleaned.csv"
REPORT_FILENAME = "quality_report.json"


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Everything the run found, in one serialisable object."""

    window: DatasetWindow
    missingness: missingness.MissingnessReport
    imputation: imputation.ImputationSummary
    anomalies: outliers.AnomalySummary
    rows_in: int
    rows_out: int
    flags_written: int
    cleaned_path: str | None
    report_path: str | None
    generated_at: datetime

    @property
    def rows_preserved(self) -> bool:
        """AC-5: the pipeline flags, it never deletes."""
        return self.rows_in == self.rows_out

    @property
    def feature_set_is_complete(self) -> bool:
        """AC-4: no nulls remain in the modelled feature set."""
        return self.imputation.is_complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "window": self.window.as_dict(),
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_preserved": self.rows_preserved,
            "feature_set_is_complete": self.feature_set_is_complete,
            "flags_written": self.flags_written,
            "missingness": self.missingness.as_dict(),
            "imputation": self.imputation.as_dict(),
            "anomalies": self.anomalies.as_dict(),
            "outputs": {"cleaned": self.cleaned_path, "report": self.report_path},
        }


def observed_mask(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Where the original reading was present, before any imputation.

    Captured before the repair stages so the detectors can be prevented from
    flagging a value this pipeline invented.
    """
    return frame[list(columns)].notna()


def write_anomaly_flags(
    session: Session, frame: pd.DataFrame, flags: pd.Series
) -> int:
    """Persist ``is_anomaly`` (task 3.7). The only column this engine writes.

    Both directions are written: a row that no longer meets the threshold is
    unflagged, so the column reflects the current detectors rather than the
    union of every run ever made. The *row* is never touched -- unflagging
    changes a judgement, not the data.
    """
    if frame.empty:
        return 0

    current = frame["is_anomaly"].astype(bool).to_numpy()
    target = flags.reindex(frame.index).fillna(False).astype(bool).to_numpy()
    changed = current != target

    if not changed.any():
        return 0

    ids = frame.loc[changed, "id"].tolist()
    targets = target[changed]

    written = 0
    for value in (True, False):
        subset = [row_id for row_id, flag in zip(ids, targets, strict=True) if flag == value]
        for start in range(0, len(subset), FLAG_BATCH_SIZE):
            batch = subset[start : start + FLAG_BATCH_SIZE]
            session.execute(
                update(Observation)
                .where(Observation.id.in_(batch))
                .values(is_anomaly=value)
            )
            written += len(batch)

    return written


def persist_outputs(
    frame: pd.DataFrame, report: dict[str, Any], *, settings: Settings
) -> tuple[str, str]:
    """Write the cleaned frame and the report to ``data/processed`` (task 3.8).

    CSV rather than Parquet: the dataset is small, the file stays readable
    without a toolchain, and it adds no dependency. The JSON sidecar carries the
    report so a cleaned file is never separated from the account of how it was
    produced.
    """
    directory = settings.data_processed_path
    directory.mkdir(parents=True, exist_ok=True)

    cleaned_path = directory / CLEANED_FILENAME
    report_path = directory / REPORT_FILENAME

    frame.to_csv(cleaned_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return str(cleaned_path), str(report_path)


def run(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    max_gap_hours: int = imputation.DEFAULT_MAX_GAP_HOURS,
    backward_fill: bool = True,
    min_votes: int = outliers.DEFAULT_MIN_VOTES,
    write_flags: bool = True,
    persist: bool = True,
    declared_mnar: dict[str, str] | None = None,
) -> QualityReport:
    """Run the full cleaning pipeline over a slice of ``observations``."""
    settings = settings or get_settings()

    raw = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    window = datasets.describe_window(raw, tuple(source_ids or ()))
    rows_in = len(raw)

    # --- 3.1 ---------------------------------------------------------------
    missing_report = missingness.analyse(raw, declared_mnar=declared_mnar)

    # Captured now: after imputation there is nothing left to distinguish an
    # observed value from a filled one.
    mask = observed_mask(raw, MEASUREMENT_COLUMNS)

    # --- 3.2, 3.3 ----------------------------------------------------------
    cleaned, impute_summary = imputation.impute(
        raw, max_gap_hours=max_gap_hours, backward=backward_fill
    )

    # --- 3.4 - 3.7 ---------------------------------------------------------
    flags, anomaly_summary = outliers.detect(
        cleaned, observed_mask=mask, min_votes=min_votes
    )
    cleaned = cleaned.copy()
    cleaned["is_anomaly"] = flags.reindex(cleaned.index).fillna(False).astype(bool)

    flags_written = write_anomaly_flags(session, raw, flags) if write_flags else 0

    report = QualityReport(
        window=window,
        missingness=missing_report,
        imputation=impute_summary,
        anomalies=anomaly_summary,
        rows_in=rows_in,
        rows_out=len(cleaned),
        flags_written=flags_written,
        cleaned_path=None,
        report_path=None,
        generated_at=datetime.now(timezone.utc),
    )

    # --- 3.8 ---------------------------------------------------------------
    if persist and not cleaned.empty:
        cleaned_path, report_path = persist_outputs(
            cleaned, report.as_dict(), settings=settings
        )
        report = QualityReport(
            **{
                **{
                    field: getattr(report, field)
                    for field in (
                        "window",
                        "missingness",
                        "imputation",
                        "anomalies",
                        "rows_in",
                        "rows_out",
                        "flags_written",
                        "generated_at",
                    )
                },
                "cleaned_path": cleaned_path,
                "report_path": report_path,
            }
        )

    return report


__all__ = [
    "CLEANED_FILENAME",
    "REPORT_FILENAME",
    "QualityReport",
    "observed_mask",
    "persist_outputs",
    "run",
    "write_anomaly_flags",
]
