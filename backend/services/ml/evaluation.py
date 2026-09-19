"""Metrics (task 7.11, FEAT-05).

MAE, RMSE and R², per model per horizon, because each answers a question the
others do not:

* **MAE** -- the typical error, in µg/m³. The number to quote to a person.
* **RMSE** -- the same error with large misses weighted more heavily. Read
  beside MAE it says whether the error is evenly spread or concentrated in a
  few bad hours, which for a pollution forecast is the difference between a
  model that is slightly vague and one that misses every spike.
* **R²** -- the share of variance explained. Included because FEAT-05 names it,
  and reported with the caveat it needs: on a strongly autocorrelated series R²
  is flattering, and persistence alone scores well above 0.9 at one hour.

**Skill against the baseline is reported with every model.** An MAE of 6 µg/m³
means nothing on its own; an MAE 20% below persistence means something. That
comparison is what AC-7 turns on, so it is computed here rather than left to the
reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

R2_CAVEAT = (
    "R² is flattering on an autocorrelated series: persistence alone scores "
    "well above 0.9 at the 1-hour horizon. Read the skill score against the "
    "baseline, not R² on its own."
)


@dataclass(frozen=True, slots=True)
class Metrics:
    """One model's error on one set of rows."""

    mae: float
    rmse: float
    r2: float
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "rows": self.rows,
        }


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """MAE, RMSE and R² over aligned arrays."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"predictions and targets differ in length: {len(y_pred)} vs {len(y_true)}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot score an empty set")

    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        # Computed from MSE rather than with squared=False, whose spelling has
        # moved between scikit-learn versions.
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        rows=len(y_true),
    )


def skill(model: Metrics, baseline: Metrics) -> float | None:
    """Fractional MAE improvement over the baseline.

    0.2 means "20% less error than persistence". Negative means the model is
    worse than doing nothing, which is a result worth seeing plainly rather than
    hiding behind a respectable R².
    """
    if baseline.mae <= 0:
        return None
    return round(1.0 - (model.mae / baseline.mae), 4)


def summarise_folds(folds: list[Metrics]) -> dict[str, Any]:
    """Mean and spread across cross-validation folds (task 7.10).

    The spread matters as much as the mean: a model whose MAE doubles between
    the first expanding-window fold and the last is not stable over time, which
    a single averaged number would hide.
    """
    if not folds:
        return {"folds": 0, "mae_mean": None, "mae_std": None, "mae_per_fold": []}

    maes = [fold.mae for fold in folds]
    return {
        "folds": len(folds),
        "mae_mean": round(float(np.mean(maes)), 4),
        "mae_std": round(float(np.std(maes)), 4),
        "rmse_mean": round(float(np.mean([fold.rmse for fold in folds])), 4),
        "r2_mean": round(float(np.mean([fold.r2 for fold in folds])), 4),
        "mae_per_fold": [round(value, 4) for value in maes],
    }


__all__ = ["R2_CAVEAT", "Metrics", "score", "skill", "summarise_folds"]
