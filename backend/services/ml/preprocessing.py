"""The model matrix and its scaling (task 7.3, AC-8, design §10.1).

Two jobs, kept apart on purpose because only one of them can leak.

**Encoding is stateless.** ``season`` is a nominal code (Phase 5), so it is
one-hot encoded -- but against the *fixed* category list in the feature spec,
never against the categories present in the data. A fitted encoder would learn
"this window contains winter and summer" and then meet monsoon at serving time
with no column for it. Because the categories come from the spec, the encoding
is a pure function and can safely run before the split.

**Scaling is fitted, so it lives inside the model.** design §10.1: "``fit`` on
train, ``transform`` on test. Fitting on the full set leaks future
distribution." The structural guarantee is better than the discipline: the
scaler is a step *inside* each estimator's pipeline, and the pipeline is only
ever handed training rows, so there is no call site at which it could see test
data. The audit in task 7.13 then confirms it from the fitted object.

**What goes in the matrix.** The engineered features of Phase 5 *and* the raw
measurements of the same hour. Both are known when a forecast is made -- the
reading at *t* is the single most informative input to a forecast for *t + 1h*,
and the naive baseline is built from exactly that value. Excluding it would
handicap every model against the baseline it must beat (AC-7).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.datasets import MEASUREMENT_COLUMNS
from services.features.spec import NOMINAL_FEATURES, SEASONS, FeatureSpec

SEASON_COLUMN = "season"


def season_columns() -> tuple[str, ...]:
    return tuple(f"season_{label}" for label in SEASONS)


def model_columns(spec: FeatureSpec) -> tuple[str, ...]:
    """Every column a model may read, in a fixed order.

    Order is derived from the spec rather than from the frame, for the same
    reason Phase 5 derives it: a matrix handed to a fitted model must carry its
    columns in the order the model learned them.
    """
    engineered = tuple(name for name in spec.feature_names if name != SEASON_COLUMN)
    seasons = season_columns() if SEASON_COLUMN in spec.feature_names else ()
    return (*MEASUREMENT_COLUMNS, *engineered, *seasons)


def encode(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build the model matrix: measurements + engineered features, season one-hot.

    Stateless, and therefore safe to run over the whole frame before splitting.
    Columns the frame does not carry are an error rather than a silent zero: a
    missing feature means the frame was not built by the Phase 5 transformer,
    and a model quietly fitted on fewer columns than it claims is worse than a
    failure.
    """
    missing = [
        name
        for name in (*MEASUREMENT_COLUMNS, *spec.feature_names)
        if name not in frame.columns
    ]
    if missing:
        raise ValueError(f"frame is missing model columns: {missing}")

    matrix = frame[list(MEASUREMENT_COLUMNS)].copy()

    for name in spec.feature_names:
        if name == SEASON_COLUMN:
            continue
        matrix[name] = frame[name]

    if SEASON_COLUMN in spec.feature_names:
        codes = frame[SEASON_COLUMN]
        for index, label in enumerate(SEASONS):
            # Comparison rather than get_dummies: the column set is the spec's,
            # not the window's, so a season absent from this slice still gets
            # its (all-zero) column and the matrix width never changes.
            matrix[f"season_{label}"] = (codes == index).astype("float64")

    return matrix[list(model_columns(spec))].astype("float64")


def nominal_columns_encoded() -> tuple[str, ...]:
    """The columns the one-hot produced, for the report."""
    return season_columns() if SEASON_COLUMN in NOMINAL_FEATURES else ()


def with_scaler(estimator: Any, *, scale: bool) -> Pipeline | Any:
    """Wrap an estimator so its scaler can only ever see training rows.

    Trees are invariant to monotone rescaling, so they are handed the matrix
    unwrapped -- scaling them would cost a transform and buy nothing. Ridge is
    not invariant: without it, a coefficient on PM10 in the hundreds and one on
    a 0/1 season flag would be penalised on wildly different scales.
    """
    if not scale:
        return estimator
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])


def fitted_scaler(model: Any) -> StandardScaler | None:
    """The scaler inside a fitted pipeline, if it has one (task 7.13)."""
    if isinstance(model, Pipeline) and "scaler" in model.named_steps:
        return model.named_steps["scaler"]
    return None


__all__ = [
    "SEASON_COLUMN",
    "encode",
    "fitted_scaler",
    "model_columns",
    "nominal_columns_encoded",
    "season_columns",
    "with_scaler",
]
