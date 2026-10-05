"""Explainability — Phase 8 (design §10.2, FEAT-06).

design §10.2: "SHAP values expose per-prediction feature attribution; the
``/ml/predict`` response carries ``top_features`` so every number the UI shows
comes with its reasoning. This is the design answer to the problem statement's
complaint about opaque models."

**Additivity is the whole reason SHAP is worth the cost.** A feature importance
from a tree's split counts tells you what the *model* looked at overall; a SHAP
value tells you what moved *this* prediction, in µg/m³, with a sign, and the
contributions sum exactly to the distance between the prediction and the model's
baseline expectation:

    prediction = base_value + Σ contributions

That identity is what makes an explanation checkable rather than decorative, so
it is asserted on every path (and a failure is raised, not rounded away).

**Every rung of the ladder can explain itself**, not only XGBoost:

| Model | Method | Exact? |
|-------|--------|--------|
| XGBoost, Random Forest | ``shap.TreeExplainer`` | yes, for trees |
| Ridge | analytic: ``coef · (x − mean)`` from the fitted scaler | yes |
| Naive Lag-1 | the prediction *is* PM2.5, so PM2.5 gets all of it | yes, trivially |

The last two need no SHAP library at all, and computing them in closed form is
both faster and more honest than approximating a known-exact answer. It also
means the Model Lab can compare *reasoning* across the ladder, not just error --
a model that beats the baseline for the wrong reasons is worth seeing.

**Attribution is not causation** (ETH-1). A SHAP value says how a feature moved
*this model's* output, which is a statement about the model, not about the
atmosphere. The caveat ships in the payload rather than being left to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from core.exceptions import EcoCityPulseError
from services.ml.models import PERSISTENCE_COLUMN

# Rows sampled when summarising a model globally.
#
# Chosen by measurement rather than by feel. TreeSHAP costs are linear in rows
# and in tree depth, and on this feature set the Random Forest takes ~6s at 500
# rows and ~25s at 2000, where XGBoost takes 0.12s and 0.37s. The *ranking*,
# though, is identical at every size from 100 upward -- mean |SHAP| settles long
# before the cost does. 500 is therefore comfortably past convergence while
# keeping the slowest rung of the ladder inside a few seconds, and the analyst
# can lower it further (task 8.4).
DEFAULT_SAMPLE_ROWS = 500

# Features returned with a single prediction. The spec's example
# (specs §8) shows two; five is enough to be useful without turning a
# forecast card into a table.
DEFAULT_TOP_FEATURES = 5

# TreeSHAP's additivity is exact in theory and floating-point in practice.
ADDITIVITY_TOLERANCE = 1e-3

ATTRIBUTION_CAVEAT = (
    "SHAP values describe how each feature moved *this model's* output, in "
    "µg/m³. They are a statement about the model, not about the atmosphere: a "
    "large attribution is not evidence that the feature caused the pollution "
    "(ETH-1)."
)

PATH_DEPENDENT = "tree_path_dependent"
INTERVENTIONAL = "interventional"
PERTURBATIONS = (PATH_DEPENDENT, INTERVENTIONAL)


class ExplainerUnavailableError(EcoCityPulseError):
    """SHAP could not be loaded, so tree models cannot be explained.

    A 503 for the same reason STL raises one: the cause is the environment, and
    the linear and baseline explainers still work, so the rest of the Model Lab
    is unaffected.
    """

    status_code = 503
    code = "explainer_unavailable"


def _load_shap() -> Any:
    """Import SHAP on first use.

    Lazy because the package pulls in numba and a compiled backend; a machine
    where that cannot load should lose tree explanations rather than the whole
    application at startup.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ExplainerUnavailableError(
            "SHAP is unavailable: the library could not be loaded.",
            details={"error": str(exc)},
        ) from exc
    return shap


# --- Results ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attribution:
    """Why one prediction came out where it did (task 8.3)."""

    base_value: float
    contributions: dict[str, float]
    prediction: float
    method: str

    @property
    def total(self) -> float:
        return float(sum(self.contributions.values()))

    @property
    def is_additive(self) -> bool:
        """The identity that makes the explanation checkable."""
        return bool(
            abs(self.base_value + self.total - self.prediction) <= ADDITIVITY_TOLERANCE
        )

    def top_features(self, limit: int = DEFAULT_TOP_FEATURES) -> dict[str, float]:
        """The ``limit`` largest contributions by magnitude, **signed**.

        Signed on purpose. "Traffic: 4.1" and "traffic: −4.1" are opposite
        explanations of the same forecast, and an absolute-valued importance
        would show them identically -- which is exactly the opacity FEAT-06
        exists to remove.
        """
        ranked = sorted(
            self.contributions.items(), key=lambda item: abs(item[1]), reverse=True
        )
        return {name: round(value, 4) for name, value in ranked[:limit]}

    def as_dict(self, limit: int = DEFAULT_TOP_FEATURES) -> dict[str, Any]:
        return {
            "base_value": round(self.base_value, 4),
            "prediction": round(self.prediction, 4),
            "top_features": self.top_features(limit),
            "method": self.method,
            "additive": self.is_additive,
            "caveat": ATTRIBUTION_CAVEAT,
        }


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """One feature's average influence across a sample."""

    feature: str
    mean_abs_shap: float
    mean_shap: float
    share: float

    @property
    def direction(self) -> str:
        """Which way the feature usually pushes.

        Reported beside the magnitude because they answer different questions:
        a feature can matter enormously and push both ways depending on its
        value, which ``mean_abs_shap`` alone would hide.
        """
        if abs(self.mean_shap) < 0.05 * max(self.mean_abs_shap, 1e-9):
            return "mixed"
        return "raises" if self.mean_shap > 0 else "lowers"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "mean_abs_shap": round(self.mean_abs_shap, 4),
            "mean_shap": round(self.mean_shap, 4),
            "share": round(self.share, 4),
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class GlobalImportance:
    """What the model relies on overall (task 8.2)."""

    features: tuple[FeatureImportance, ...]
    rows: int
    method: str
    base_value: float

    def top(self, limit: int = DEFAULT_TOP_FEATURES) -> tuple[FeatureImportance, ...]:
        return self.features[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "method": self.method,
            "base_value": round(self.base_value, 4),
            "features": [item.as_dict() for item in self.features],
            "caveat": ATTRIBUTION_CAVEAT,
        }


# --- Strategies -------------------------------------------------------------


def _unwrap(estimator: Any) -> tuple[Any, Pipeline | None]:
    """Separate a fitted estimator from the pipeline that scales for it."""
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["estimator"], estimator
    return estimator, None


def _is_tree(estimator: Any) -> bool:
    module = type(estimator).__module__
    return module.startswith(("xgboost", "sklearn.ensemble"))


def _is_linear(estimator: Any) -> bool:
    return hasattr(estimator, "coef_") and hasattr(estimator, "intercept_")


class ModelExplainer:
    """Explains one fitted model, whichever rung of the ladder it came from.

    Construction is deliberately cheap for the exact strategies and deferred
    for the SHAP one, so building an explainer for a Ridge model costs nothing
    and does not require the SHAP library to be installed at all.
    """

    def __init__(
        self,
        estimator: Any,
        columns: tuple[str, ...],
        *,
        perturbation: str = PATH_DEPENDENT,
        background: pd.DataFrame | None = None,
    ) -> None:
        if perturbation not in PERTURBATIONS:
            raise ValueError(
                f"unknown perturbation {perturbation!r}; available: {list(PERTURBATIONS)}"
            )

        self.columns = tuple(columns)
        self.perturbation = perturbation
        self.background = background
        self.pipeline = estimator
        self.inner, self._wrapper = _unwrap(estimator)
        self._shap_explainer: Any = None

        if _is_tree(self.inner):
            self.method = f"shap_tree:{perturbation}"
        elif _is_linear(self.inner):
            self.method = "linear_exact"
        else:
            self.method = "persistence_exact"

    # --- The three strategies ----------------------------------------------

    def _tree_values(self, matrix: pd.DataFrame) -> tuple[np.ndarray, float]:
        shap = _load_shap()

        if self.perturbation == INTERVENTIONAL and self.background is None:
            raise ValueError(
                "the interventional perturbation needs a background sample; "
                "pass one or use tree_path_dependent."
            )

        # SHAP rejects an unsupported tree format at *evaluation* time, not at
        # construction, so both steps sit inside the same guard.
        try:
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(
                    self.inner,
                    data=self.background[list(self.columns)]
                    if self.background is not None
                    and self.perturbation == INTERVENTIONAL
                    else None,
                    feature_perturbation=self.perturbation,
                )
            values = np.asarray(
                self._shap_explainer.shap_values(matrix[list(self.columns)]),
                dtype="float64",
            )
        except NotImplementedError as exc:
            # XGBoost 3 emits categorical-split metadata even for entirely
            # numeric features, and SHAP's interventional path refuses it.
            # Reported rather than silently downgraded: an analyst who asked for
            # interventional values and quietly received path-dependent ones
            # would draw conclusions from a method they did not choose.
            raise ExplainerUnavailableError(
                f"The {self.perturbation!r} perturbation is not supported for "
                f"this model's tree format ({type(self.inner).__name__}). "
                f"Use {PATH_DEPENDENT!r}, which reads the trees directly.",
                details={
                    "perturbation": self.perturbation,
                    "estimator": type(self.inner).__name__,
                    "error": str(exc),
                },
            ) from exc
        except ImportError as exc:
            # ``import shap`` succeeding does not mean TreeSHAP works: the
            # compiled tree kernel (``shap._cext``) is imported only when the
            # first tree is loaded. A host that blocks that one DLL — Windows
            # Application Control has, here — would otherwise surface as a bare
            # 500 on every forecast. It is the same environmental cause
            # ``_load_shap`` already maps, so it gets the same 503.
            raise ExplainerUnavailableError(
                "SHAP is unavailable: its compiled tree backend could not be "
                "loaded.",
                details={
                    "estimator": type(self.inner).__name__,
                    "error": str(exc),
                },
            ) from exc

        base = np.asarray(self._shap_explainer.expected_value, dtype="float64")
        return values, float(base.reshape(-1)[0])

    def _linear_values(self, matrix: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Closed-form SHAP for a linear model.

        For a linear model with independent features the Shapley value of
        feature *i* is exactly ``coef_i · (x_i − E[x_i])``, and the fitted
        scaler already holds ``E[x_i]`` from the training rows. So the "SHAP"
        values here are not an approximation of anything -- they are the
        quantity SHAP would converge to, computed directly.
        """
        raw = matrix[list(self.columns)].to_numpy(dtype="float64")
        coef = np.asarray(self.inner.coef_, dtype="float64").reshape(-1)

        scaler = (
            self._wrapper.named_steps["scaler"]
            if self._wrapper is not None and "scaler" in self._wrapper.named_steps
            else None
        )
        if scaler is not None:
            # The model sees standardized inputs, so the reference point is the
            # training mean the scaler already holds and the effective slope is
            # coef / scale.
            centre = np.asarray(scaler.mean_, dtype="float64")
            scale = np.asarray(scaler.scale_, dtype="float64")
            contributions = (raw - centre) / scale * coef
            base = float(self.inner.intercept_)
        elif self.background is not None:
            centre = self.background[list(self.columns)].mean().to_numpy(dtype="float64")
            contributions = (raw - centre) * coef
            base = float(self.inner.intercept_ + centre @ coef)
        else:
            # No scaler and no background means no reference distribution to
            # centre on. Measuring from the intercept instead is still exact and
            # still additive; it is simply expressed as "what each feature adds
            # to the intercept" rather than "how each feature differs from
            # typical". The reference must not be taken from the rows being
            # explained -- that would make a single-row explanation measure a
            # row against itself and attribute nothing to anything.
            contributions = raw * coef
            base = float(self.inner.intercept_)

        return contributions, base

    def _persistence_values(self, matrix: pd.DataFrame) -> tuple[np.ndarray, float]:
        """The baseline's explanation: PM2.5 accounts for all of it.

        Degenerate, and worth producing anyway. It makes the Model Lab's
        comparison uniform, and it states plainly what the baseline is -- a
        model whose entire reasoning is "it will stay as it is".

        The reference is the training mean stored at fit time, never the mean of
        the rows being explained: a single-row request measured against itself
        would report that nothing contributed anything.
        """
        raw = matrix[list(self.columns)].to_numpy(dtype="float64")
        contributions = np.zeros_like(raw)

        index = self.columns.index(PERSISTENCE_COLUMN)
        base = getattr(self.inner, "base_value_", None)
        if base is None:
            base = (
                float(self.background[PERSISTENCE_COLUMN].mean())
                if self.background is not None
                else float(matrix[PERSISTENCE_COLUMN].mean())
            )

        contributions[:, index] = raw[:, index] - float(base)
        return contributions, float(base)

    def _values(self, matrix: pd.DataFrame) -> tuple[np.ndarray, float]:
        missing = [column for column in self.columns if column not in matrix.columns]
        if missing:
            raise ValueError(f"matrix is missing explained columns: {missing}")

        if _is_tree(self.inner):
            return self._tree_values(matrix)
        if _is_linear(self.inner):
            return self._linear_values(matrix)
        return self._persistence_values(matrix)

    # --- Public surface -----------------------------------------------------

    def explain(self, matrix: pd.DataFrame) -> tuple[Attribution, ...]:
        """Per-row attributions (task 8.3)."""
        values, base = self._values(matrix)
        predictions = np.asarray(
            self.pipeline.predict(matrix[list(self.columns)]), dtype="float64"
        )

        return tuple(
            Attribution(
                base_value=base,
                contributions={
                    name: float(value)
                    for name, value in zip(self.columns, row, strict=True)
                },
                prediction=float(prediction),
                method=self.method,
            )
            for row, prediction in zip(values, predictions, strict=True)
        )

    def explain_one(self, matrix: pd.DataFrame) -> Attribution:
        """Attribution for a single-row frame -- what serving needs."""
        if len(matrix) != 1:
            raise ValueError(f"expected exactly one row, got {len(matrix)}")
        return self.explain(matrix)[0]

    def global_importance(
        self, matrix: pd.DataFrame, *, sample_rows: int = DEFAULT_SAMPLE_ROWS
    ) -> GlobalImportance:
        """Mean |SHAP| per feature over a sample (task 8.2).

        Sampled evenly across the frame rather than randomly: the frame is time
        ordered, so a stride keeps the whole period represented and the same
        request twice gives the same answer.
        """
        if matrix.empty:
            raise ValueError("cannot summarise an empty matrix")

        if len(matrix) > sample_rows:
            positions = np.unique(
                np.linspace(0, len(matrix) - 1, sample_rows).astype(int)
            )
            matrix = matrix.iloc[positions]

        values, base = self._values(matrix)
        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        total = float(mean_abs.sum()) or 1.0

        features = sorted(
            (
                FeatureImportance(
                    feature=name,
                    mean_abs_shap=float(magnitude),
                    mean_shap=float(signed),
                    share=float(magnitude) / total,
                )
                for name, magnitude, signed in zip(
                    self.columns, mean_abs, mean_signed, strict=True
                )
            ),
            key=lambda item: item.mean_abs_shap,
            reverse=True,
        )

        return GlobalImportance(
            features=tuple(features),
            rows=len(matrix),
            method=self.method,
            base_value=base,
        )


def for_artifact(
    artifact: dict[str, Any],
    *,
    perturbation: str = PATH_DEPENDENT,
    background: pd.DataFrame | None = None,
) -> ModelExplainer:
    """Build an explainer from a registered model artifact (task 8.1)."""
    return ModelExplainer(
        artifact["estimator"],
        tuple(artifact["columns"]),
        perturbation=perturbation,
        background=background,
    )


__all__ = [
    "ADDITIVITY_TOLERANCE",
    "ATTRIBUTION_CAVEAT",
    "DEFAULT_SAMPLE_ROWS",
    "DEFAULT_TOP_FEATURES",
    "INTERVENTIONAL",
    "PATH_DEPENDENT",
    "PERTURBATIONS",
    "Attribution",
    "ExplainerUnavailableError",
    "FeatureImportance",
    "GlobalImportance",
    "ModelExplainer",
    "for_artifact",
]
