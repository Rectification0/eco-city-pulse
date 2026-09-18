"""Data Quality Engine — Phase 3, BACSE301 Module 2 (specs §5.3, design §7).

Split into modules rather than one ``quality_service.py`` because the three
concerns are independently testable and each carries real statistical
reasoning:

- ``missingness`` — counts, gap lengths, and the MCAR/MAR judgement, bounded by
  what the data can actually support.
- ``imputation`` — short-gap forward/backward fill, then MICE.
- ``outliers`` — IQR, Z-score and Isolation Forest, voting into one flag.
- ``pipeline`` — the orchestration, the write-back, and the persisted output.

The engine's guarantee, in one line: **it repairs values and flags records; it
never removes one** (AC-5).
"""

from services.quality import imputation, missingness, outliers, pipeline
from services.quality.imputation import (
    ImputationSummary,
    fill_short_gaps,
    impute,
    impute_mice,
)
from services.quality.missingness import (
    ColumnMissingness,
    Mechanism,
    MissingnessReport,
    analyse,
)
from services.quality.outliers import (
    AnomalySummary,
    detect,
    flag_iqr,
    flag_isolation_forest,
    flag_zscore,
    iqr_bounds,
)
from services.quality.pipeline import QualityReport, run

__all__ = [
    "AnomalySummary",
    "imputation",
    "missingness",
    "outliers",
    "pipeline",
    "ColumnMissingness",
    "ImputationSummary",
    "Mechanism",
    "MissingnessReport",
    "QualityReport",
    "analyse",
    "detect",
    "fill_short_gaps",
    "flag_iqr",
    "flag_isolation_forest",
    "flag_zscore",
    "impute",
    "impute_mice",
    "iqr_bounds",
    "run",
]
