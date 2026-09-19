"""Confidence intervals for a forecast (task 9.3, specs §8).

specs §8 requires every prediction to carry a ``confidence_interval``. The
question is where the number comes from, and the honest answers are narrower
than they look.

**Not from the model's own uncertainty.** XGBoost, the production model, has no
notion of predictive variance; a gradient-boosted point forecast is a point and
nothing more. A Random Forest can report the spread across its trees, but that
measures disagreement between trees, not error against reality -- a forest can
be unanimously wrong.

**So the interval is calibrated from residuals.** Split conformal prediction:
take the errors the model actually made on data it was not fitted on, and read
the quantiles off them. If 90% of held-out errors fell between −11 and +14
µg/m³, then ``prediction − 11`` to ``prediction + 14`` is a 90% interval. It
requires no distributional assumption and no extra model.

**The quantiles are signed, not absolute.** A symmetric ``± q`` would hide two
real properties of pollution error: models under-predict spikes more than they
over-predict calm hours (skew), and a model can be biased at one horizon and
not another. Signed quantiles carry both, so the interval sits where the errors
actually are rather than centred on the point forecast by assumption.

**The guarantee is approximate here, and that is stated rather than implied.**
Conformal coverage holds under *exchangeability*. A time series is not
exchangeable -- tomorrow's errors are correlated with today's, and a regime the
calibration window never saw (a stubble-burning week, a new sensor) will break
it. The reported coverage is what the calibration window produced; it is not a
promise about the future, and the payload says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Coverage levels the calibration grid stores. Dense enough that any request
# between 50% and 99% is read off directly rather than extrapolated.
QUANTILE_GRID: tuple[float, ...] = tuple(
    round(value, 3) for value in np.arange(0.005, 0.9951, 0.005)
)

DEFAULT_COVERAGE = 0.80

# Below this a "confidence interval" is a point, above it the tails are being
# estimated from a handful of observations.
MIN_COVERAGE = 0.50
MAX_COVERAGE = 0.99

# Fewer calibration residuals than this and the quantiles are noise.
MIN_CALIBRATION_ROWS = 50

# z-scores for the fallback, keyed by coverage. Used only when an artifact
# predates calibration and all that survives is an RMSE.
_Z = {0.5: 0.674, 0.68: 1.0, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}

CONFORMAL = "conformal_residual_quantiles"
NORMAL = "rmse_normal_approximation"

INTERVAL_CAVEAT = (
    "Calibrated on the errors this model made on held-out data. Conformal "
    "coverage assumes exchangeability, which a time series does not satisfy: a "
    "pollution regime absent from the calibration window can fall outside the "
    "interval more often than the stated rate."
)


@dataclass(frozen=True, slots=True)
class Calibration:
    """Signed residual quantiles from a held-out window.

    ``residual = actual − predicted``, so a positive quantile means the model
    under-predicted. Stored with the model artifact, because an interval is
    only meaningful beside the model whose errors produced it.
    """

    quantiles: dict[float, float]
    rows: int
    rmse: float
    window: str = ""

    @property
    def median_bias(self) -> float:
        """The middle residual. Non-zero means the model is systematically off."""
        return self.lookup(0.5)

    def lookup(self, level: float) -> float:
        """The stored quantile nearest the requested level."""
        if not self.quantiles:
            return 0.0
        nearest = min(self.quantiles, key=lambda stored: abs(stored - level))
        return self.quantiles[nearest]

    def interval(self, prediction: float, coverage: float) -> tuple[float, float]:
        """The prediction's interval at this coverage.

        The lower bound is clamped at zero: a negative PM2.5 concentration is
        not a measurement, and an interval that reaches below zero would be
        claiming something impossible rather than something uncertain.
        """
        tail = (1.0 - coverage) / 2.0
        low = prediction + self.lookup(tail)
        high = prediction + self.lookup(1.0 - tail)
        return max(0.0, round(low, 4)), round(max(high, low), 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": CONFORMAL,
            "rows": self.rows,
            "rmse": round(self.rmse, 4),
            "median_bias": round(self.median_bias, 4),
            "window": self.window,
            "quantiles": {str(level): round(value, 4) for level, value in self.quantiles.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Calibration:
        return cls(
            quantiles={
                float(level): float(value)
                for level, value in payload.get("quantiles", {}).items()
            },
            rows=int(payload.get("rows", 0)),
            rmse=float(payload.get("rmse", 0.0)),
            window=str(payload.get("window", "")),
        )


def calibrate(
    actual: np.ndarray, predicted: np.ndarray, *, window: str = ""
) -> Calibration | None:
    """Read the residual quantiles off a held-out window (task 9.3).

    Returns ``None`` when there are too few residuals to estimate a tail from;
    the caller then falls back to the RMSE approximation and says which it used.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")

    residuals = actual - predicted
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < MIN_CALIBRATION_ROWS:
        return None

    return Calibration(
        quantiles={
            level: float(value)
            for level, value in zip(
                QUANTILE_GRID,
                np.quantile(residuals, QUANTILE_GRID),
                strict=True,
            )
        },
        rows=len(residuals),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        window=window,
    )


def normal_interval(
    prediction: float, rmse: float, coverage: float
) -> tuple[float, float]:
    """Fallback for an artifact registered before calibration existed.

    A symmetric Gaussian interval around the point forecast. Weaker than the
    conformal one -- it assumes the errors are normal and unbiased, which
    pollution residuals are not -- so the response labels which method produced
    the number rather than presenting the two as equivalent.
    """
    nearest = min(_Z, key=lambda level: abs(level - coverage))
    half = _Z[nearest] * max(rmse, 0.0)
    return max(0.0, round(prediction - half, 4)), round(prediction + half, 4)


def empirical_coverage(
    actual: np.ndarray, predicted: np.ndarray, calibration: Calibration, coverage: float
) -> float:
    """Share of held-out points the interval actually contained.

    Computed on the calibration window itself, so it is a consistency check --
    "do these quantiles do what they claim on the data they came from" -- not
    evidence about future coverage.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")

    tail = (1.0 - coverage) / 2.0
    low = predicted + calibration.lookup(tail)
    high = predicted + calibration.lookup(1.0 - tail)
    inside = (actual >= low) & (actual <= high)
    return float(np.mean(inside)) if len(actual) else 0.0


def validate_coverage(coverage: float) -> float:
    if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
        raise ValueError(
            f"coverage must be between {MIN_COVERAGE} and {MAX_COVERAGE}, got {coverage}"
        )
    return coverage


__all__ = [
    "CONFORMAL",
    "DEFAULT_COVERAGE",
    "INTERVAL_CAVEAT",
    "MAX_COVERAGE",
    "MIN_CALIBRATION_ROWS",
    "MIN_COVERAGE",
    "NORMAL",
    "QUANTILE_GRID",
    "Calibration",
    "calibrate",
    "empirical_coverage",
    "normal_interval",
    "validate_coverage",
]
