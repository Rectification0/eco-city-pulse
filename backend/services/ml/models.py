"""The model ladder (tasks 7.5-7.8, design §10).

design §10 calls the ladder deliberate, and the order is the argument:

| Model | Purpose |
|-------|---------|
| **Naive Lag-1** | Honest baseline. "Next hour looks like this hour" is a strong forecast, and anything that cannot beat it adds nothing. |
| **Ridge** | Linear reference, regularized against collinear pollutants. |
| **Random Forest** | Non-linear, low-tuning benchmark. |
| **XGBoost** | Primary production model (FEAT-05). |

**The baseline is a real model, not a formality.** It is the reason AC-7 exists:
a 1-hour PM2.5 forecast that merely repeats the current reading achieves an MAE
most published models would envy, because pollution is strongly autocorrelated.
Reporting R² without that comparison would make an unremarkable model look
excellent. So persistence is fitted, scored and registered exactly like the
others, and the trainer refuses to call a run successful if XGBoost cannot beat
it at the 1-hour horizon.

**Every estimator is seeded and single-recipe.** No hyperparameter search: the
phase's deliverable is a sound pipeline, and a search inside a time-series
pipeline is one more place for the validation folds to leak. The chosen values
are conservative defaults, recorded with each registered model so the Model Lab
can show them (task 10.12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from core.exceptions import EcoCityPulseError

RANDOM_STATE = 42

# The column persistence reads. Selection protects it for this reason.
PERSISTENCE_COLUMN = "pm25"


class ModelUnavailableError(EcoCityPulseError):
    """An optional estimator's library could not be loaded.

    A 503 rather than a 500, for the same reason STL raises one: the rest of the
    pipeline is unaffected, and the cause is the environment rather than the
    request.
    """

    status_code = 503
    code = "model_unavailable"


class NaiveLag1(BaseEstimator, RegressorMixin):
    """Persistence: the forecast is the most recent observation (task 7.5).

    Fitting stores nothing but the column name -- which is the point. It has no
    parameters to overfit with, so its score is a property of the data rather
    than of any modelling choice, and that is what makes it the honest floor.

    Note what it is *not*: it is not "the value one row above", which across a
    missing hour would be some older reading. It is the PM2.5 of the row being
    predicted from, which the Phase 5 matrix carries directly.
    """

    def __init__(self, column: str = PERSISTENCE_COLUMN) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: Any = None) -> NaiveLag1:  # noqa: N803
        if self.column not in X.columns:
            raise ValueError(
                f"the persistence baseline needs the {self.column!r} column; "
                f"got {list(X.columns)[:8]}..."
            )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return X[self.column].to_numpy(dtype="float64")


def _xgboost_regressor(**overrides: Any) -> Any:
    """Import XGBoost on first use.

    Lazy for the same reason STL is: xgboost ships a compiled extension, and on
    a machine where it cannot load, a module-level import would take the whole
    application down at startup over one estimator. This way the rest of the
    ladder still trains and only XGBoost reports itself unavailable.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelUnavailableError(
            "XGBoost is unavailable: the library could not be loaded.",
            details={"error": str(exc)},
        ) from exc

    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        # Row and column subsampling: the cheapest regularisation there is, and
        # this data has more correlated columns than independent ones.
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **overrides,
    }
    return XGBRegressor(**params)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One rung of the ladder: how to build it, and whether it wants scaling."""

    name: str
    build: Any
    scale: bool = False
    is_baseline: bool = False
    notes: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scaled": self.scale,
            "is_baseline": self.is_baseline,
            "notes": self.notes,
            "hyperparameters": self.hyperparameters,
        }


# The ladder, in the order design §10 presents it. Baseline first, so a report
# read top to bottom sets the bar before it shows anything clearing it.
def build_ladder() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            name="naive_lag1",
            build=NaiveLag1,
            is_baseline=True,
            notes="Persistence: the forecast is the current reading (AC-7 floor).",
        ),
        ModelSpec(
            name="ridge",
            build=lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
            scale=True,
            notes="Linear reference, regularized against collinear pollutants.",
            hyperparameters={"alpha": 1.0},
        ),
        ModelSpec(
            name="random_forest",
            build=lambda: RandomForestRegressor(
                n_estimators=200,
                # Unbounded depth on 30k rows memorises the training window;
                # a leaf floor costs a little fit and buys a lot of stability.
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            notes="Non-linear, low-tuning benchmark.",
            hyperparameters={
                "n_estimators": 200,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
            },
        ),
        ModelSpec(
            name="xgboost",
            build=_xgboost_regressor,
            notes="Primary production model (FEAT-05).",
            hyperparameters={
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
            },
        ),
    )


PRODUCTION_MODEL = "xgboost"
BASELINE_MODEL = "naive_lag1"


def ladder_by_name() -> dict[str, ModelSpec]:
    return {spec.name: spec for spec in build_ladder()}


__all__ = [
    "BASELINE_MODEL",
    "PERSISTENCE_COLUMN",
    "PRODUCTION_MODEL",
    "RANDOM_STATE",
    "ModelSpec",
    "ModelUnavailableError",
    "NaiveLag1",
    "build_ladder",
    "ladder_by_name",
]
