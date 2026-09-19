"""Log transformation of skewed pollutants (task 5.4, specs §6.1).

specs §6.1 asks for "log transformation on highly skewed pollutants". Which
pollutants those are is a property of the data, and Phase 4 already answers it:
``eda.profile.assess_distribution`` applies ``log1p``, recomputes the skew, and
recommends the transform only when it materially improves (task 4.4). This
module reuses that verdict rather than restating the rule, so the number shown
in the EDA Studio and the decision taken by the feature pipeline can never
disagree.

**The decision is fitted; the arithmetic is not.** Which columns get a log is
chosen once, on the training slice, and frozen into the ``FeatureSpec``. At
inference time the spec is replayed -- no skew is recomputed, because a serving
window of 48 hours would give a different answer from a year of training data
and the two paths would silently drift apart (task 5.5, AC-8).

**``log1p``, not ``log``.** Zero is a legitimate concentration, and
``log(0)`` is not a number. ``log1p`` is defined there, and on the µg/m³ scale
the ``+1`` is well inside measurement error.

**The raw column is kept.** The transform adds ``pm25_log`` beside ``pm25``
instead of replacing it: tree models are invariant to monotone transforms and
read better on the original scale, while the linear models of Phase 7 want the
symmetric one. Phase 7's feature selection (task 7.4) drops whichever of the
pair a given model does not need -- keeping both here means neither model is
forced onto the other's scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.eda import profile
from services.features.spec import POLLUTANT_COLUMNS

LOG_SUFFIX = "_log"


def log_feature_name(column: str) -> str:
    return f"{column}{LOG_SUFFIX}"


def choose_log_columns(
    frame: pd.DataFrame, candidates: tuple[str, ...] = POLLUTANT_COLUMNS
) -> tuple[tuple[str, ...], dict[str, dict[str, float | None]]]:
    """Which candidates are skewed enough for the transform to earn its place.

    Returns the chosen columns and the evidence behind each verdict, so the
    stored spec can say *why* a column was transformed rather than only that it
    was.
    """
    chosen: list[str] = []
    evidence: dict[str, dict[str, float | None]] = {}

    for column in candidates:
        if column not in frame.columns:
            continue
        assessment = profile.assess_distribution(frame, column)
        evidence[column] = {
            "skewness": assessment.skewness,
            "log_skewness": assessment.log_skewness,
            "recommended": assessment.recommend_log_transform,
            "rationale": assessment.rationale,
        }
        if assessment.recommend_log_transform:
            chosen.append(column)

    return tuple(chosen), evidence


def add_log_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Add ``<column>_log`` for each named column.

    A negative reading yields NaN rather than an exception. Negative µg/m³ is
    not a measurement, and one bad row should leave that row's feature unknown
    instead of failing a build of thirty thousand.
    """
    frame = frame.copy()

    for column in columns:
        name = log_feature_name(column)
        if frame.empty:
            frame[name] = pd.Series(dtype="float64")
            continue
        values = frame[column].astype("float64")
        frame[name] = np.log1p(values.where(values >= 0))

    return frame


__all__ = [
    "LOG_SUFFIX",
    "add_log_features",
    "choose_log_columns",
    "log_feature_name",
]
