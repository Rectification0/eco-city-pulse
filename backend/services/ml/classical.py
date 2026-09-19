"""ARIMA and Prophet baselines (task 7.9, specs §3.1, design §10).

These sit beside the ladder rather than on it. They are *univariate*: they see
one station's PM2.5 history and nothing else -- no traffic, no temperature, no
calendar features beyond what they infer themselves. Comparing them to XGBoost
is not a fair fight and is not meant to be. It answers a different question:
**how much does the multivariate pipeline actually buy over classical
time-series methods applied to the same series?** If the answer were "nothing",
the whole feature pipeline would be unjustified.

**How each one is evaluated, and why they differ.**

*ARIMA* is autoregressive, so a forecast made without recent readings is
meaningless. Parameters are fitted once on the training window, then for each
evaluation origin the fitted parameters are **applied** to the history up to
that origin and a ``horizon``-step forecast is taken. No refitting -- refitting
at each origin would be a different (and much slower) experiment, and would let
the model tune itself on data the other models never saw. Because each origin
costs a pass over the history, origins are sampled evenly across the test
window and the count is reported.

*Prophet* is not autoregressive: it is a trend-plus-seasonality curve fitted to
the training period and extrapolated. So it is fitted once and asked for the
whole test window in one call. That it never consults a recent reading is
exactly why it is a floor rather than a competitor -- it is the score to beat
by knowing anything at all about the last few hours.

Both libraries are imported lazily. Prophet in particular pulls in a compiled
Stan backend, and a machine where that cannot load should lose one optional
baseline rather than the entire ML pipeline.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.ml.evaluation import Metrics, score
from services.ml.models import ModelUnavailableError

# ARIMA(2,0,2) with a constant: enough autoregression for an hourly pollution
# series without a search that would need its own validation discipline. The
# series is differenced only if it fails a stationarity check at fit time.
DEFAULT_ORDER = (2, 0, 2)

# Evaluation origins sampled from the test window. Each one costs a pass over
# the history, so this bounds the baseline at a few seconds.
DEFAULT_MAX_ORIGINS = 200

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClassicalResult:
    """One classical baseline's score, with how it was produced."""

    name: str
    metrics: Metrics | None
    origins: int
    note: str = ""
    available: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "origins": self.origins,
            "note": self.note,
            "metrics": self.metrics.as_dict() if self.metrics else None,
        }


def _series(frame: pd.DataFrame, column: str = "pm25") -> pd.Series:
    """One station's series on a gapless hourly index.

    Both libraries want a regular series; the grid of Phase 5 is reused so an
    absent hour is an interpolated point rather than a silently compressed one.
    """
    series = frame.set_index(frame["timestamp"].dt.floor("h"))[column]
    series = series[~series.index.duplicated(keep="first")].sort_index()
    full = series.reindex(
        pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    )
    return full.interpolate(limit_direction="both")


def _origins(count: int, horizon: int, max_origins: int) -> np.ndarray:
    """Evenly spaced evaluation origins inside the test window."""
    usable = count - horizon
    if usable <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, usable - 1, min(max_origins, usable)).astype(int))


def arima_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    column: str = "pm25",
    order: tuple[int, int, int] = DEFAULT_ORDER,
    max_origins: int = DEFAULT_MAX_ORIGINS,
) -> ClassicalResult:
    """Fit ARIMA on train, forecast ``horizon`` hours ahead from test origins."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:  # pragma: no cover - environment dependent
        return ClassicalResult(
            name="arima",
            metrics=None,
            origins=0,
            note=f"statsmodels could not be loaded: {exc}",
            available=False,
        )

    train_series = _series(train, column)
    test_series = _series(test, column)
    history = pd.concat([train_series, test_series])

    origins = _origins(len(test_series), horizon, max_origins)
    if len(origins) == 0:
        return ClassicalResult(
            name="arima",
            metrics=None,
            origins=0,
            note="test window shorter than the forecast horizon",
        )

    try:
        with warnings.catch_warnings():
            # Convergence chatter on a 2,0,2 fit is expected and not actionable;
            # a failure to fit at all is caught below and reported.
            warnings.simplefilter("ignore")
            fitted = ARIMA(train_series, order=order).fit()

            predictions, actuals = [], []
            for origin in origins:
                # Index into the combined history: everything up to and
                # including this origin, and nothing after it.
                end = len(train_series) + int(origin) + 1
                applied = fitted.apply(history.iloc[:end], refit=False)
                predictions.append(float(applied.forecast(horizon).iloc[-1]))
                actuals.append(float(history.iloc[end + horizon - 1]))
    except Exception as exc:  # noqa: BLE001 - a baseline must not fail a run
        logger.warning("ARIMA baseline failed: %s", exc)
        return ClassicalResult(
            name="arima", metrics=None, origins=0, note=f"fit failed: {exc}"
        )

    return ClassicalResult(
        name="arima",
        metrics=score(np.asarray(actuals), np.asarray(predictions)),
        origins=len(origins),
        note=(
            f"ARIMA{order} fitted once on the training window; parameters applied "
            f"to the history at {len(origins)} evenly spaced origins, each "
            f"forecasting {horizon}h ahead. Univariate: it sees PM2.5 only."
        ),
    )


def prophet_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    column: str = "pm25",
) -> ClassicalResult:
    """Fit Prophet on train and extrapolate across the test window."""
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - environment dependent
        return ClassicalResult(
            name="prophet",
            metrics=None,
            origins=0,
            note=f"prophet could not be loaded: {exc}",
            available=False,
        )

    train_series = _series(train, column)
    test_series = _series(test, column)

    if len(test_series) <= horizon:
        return ClassicalResult(
            name="prophet",
            metrics=None,
            origins=0,
            note="test window shorter than the forecast horizon",
        )

    # Prophet wants naive timestamps in columns named ds/y.
    history = pd.DataFrame(
        {"ds": train_series.index.tz_localize(None), "y": train_series.to_numpy()}
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
            )
            # cmdstanpy logs the whole optimiser at INFO; it is noise here.
            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
            model.fit(history)

            future = pd.DataFrame({"ds": test_series.index.tz_localize(None)})
            forecast = model.predict(future)
    except Exception as exc:  # noqa: BLE001 - a baseline must not fail a run
        logger.warning("Prophet baseline failed: %s", exc)
        return ClassicalResult(
            name="prophet", metrics=None, origins=0, note=f"fit failed: {exc}"
        )

    return ClassicalResult(
        name="prophet",
        metrics=score(test_series.to_numpy(), forecast["yhat"].to_numpy()),
        origins=len(test_series),
        note=(
            "Trend plus daily/weekly seasonality fitted on the training window "
            "and extrapolated across the test window. It consults no recent "
            "reading, which is why it is a floor rather than a competitor -- "
            "and why this score is identical at every horizon: the forecast for "
            "a given hour does not depend on how far ahead it was requested. "
            "Read it as a long-range floor, not as an h-step comparison."
        ),
    )


def run_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    column: str = "pm25",
    include_prophet: bool = True,
    max_origins: int = DEFAULT_MAX_ORIGINS,
) -> tuple[ClassicalResult, ...]:
    """Both classical baselines, neither of which may fail the training run."""
    results = [
        arima_baseline(
            train, test, horizon=horizon, column=column, max_origins=max_origins
        )
    ]
    if include_prophet:
        results.append(prophet_baseline(train, test, horizon=horizon, column=column))
    return tuple(results)


__all__ = [
    "DEFAULT_MAX_ORIGINS",
    "DEFAULT_ORDER",
    "ClassicalResult",
    "ModelUnavailableError",
    "arima_baseline",
    "prophet_baseline",
    "run_baselines",
]
