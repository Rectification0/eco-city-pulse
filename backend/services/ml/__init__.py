"""ML Pipeline — Phase 7 (FEAT-05, specs §6.3, design §10).

Split by pipeline stage, because each stage is where a specific failure lives
and each is worth reading on its own:

- ``targets``       — the forward shift that makes it a forecast (7.1)
- ``splitting``     — chronological split, embargo, expanding-window CV (7.2, 7.10)
- ``preprocessing`` — the model matrix; the scaler that only sees train (7.3)
- ``selection``     — which columns survive, fitted on train (7.4)
- ``models``        — the ladder: persistence, Ridge, Random Forest, XGBoost (7.5-7.8)
- ``classical``     — ARIMA and Prophet, univariate baselines (7.9)
- ``evaluation``    — MAE, RMSE, R², and skill against the baseline (7.11)
- ``registry``      — artifact plus the ``models`` row (7.12)
- ``training``      — the orchestration
- ``explain``       — SHAP attribution, exact for Ridge and the baseline (8.1-8.4)
- ``serving``       — load a registered model and replay its feature contract
- ``intervals``     — conformal prediction intervals from held-out residuals (9.3)
- ``prediction``    — the forecast, its interval, its reasoning, its row (9.1-9.6)

Two commitments run through all of it:

**Nothing is fitted on data that is later scored.** Not the scaler, not the
feature selection, not the Phase 5 log-transform decision. Each is fitted inside
the training window and the run reports the timestamps that prove it (AC-8).

**The baseline is taken seriously.** Persistence is a strong forecast for
hourly PM2.5, so it is fitted, scored and registered like any other model, and
a run that cannot beat it at one hour says so rather than quoting an R² that
would hide it (AC-7).
"""

from services.ml import (
    classical,
    evaluation,
    explain,
    intervals,
    models,
    prediction,
    preprocessing,
    registry,
    selection,
    serving,
    splitting,
    targets,
    training,
)
from services.ml.evaluation import Metrics
from services.ml.explain import Attribution, GlobalImportance, ModelExplainer
from services.ml.models import NaiveLag1, build_ladder
from services.ml.prediction import PredictionResult, backfill_actuals
from services.ml.registry import RegisteredModel
from services.ml.splitting import TimeSplit
from services.ml.targets import DEFAULT_HORIZONS, target_name
from services.ml.training import HorizonReport, ModelReport, TrainingReport, run

__all__ = [
    "DEFAULT_HORIZONS",
    "Attribution",
    "GlobalImportance",
    "HorizonReport",
    "Metrics",
    "ModelExplainer",
    "ModelReport",
    "NaiveLag1",
    "PredictionResult",
    "RegisteredModel",
    "TimeSplit",
    "TrainingReport",
    "build_ladder",
    "backfill_actuals",
    "classical",
    "evaluation",
    "explain",
    "intervals",
    "models",
    "prediction",
    "preprocessing",
    "registry",
    "run",
    "selection",
    "serving",
    "splitting",
    "target_name",
    "targets",
    "training",
]
