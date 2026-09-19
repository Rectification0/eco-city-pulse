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
    models,
    preprocessing,
    registry,
    selection,
    splitting,
    targets,
    training,
)
from services.ml.evaluation import Metrics
from services.ml.models import NaiveLag1, build_ladder
from services.ml.registry import RegisteredModel
from services.ml.splitting import TimeSplit
from services.ml.targets import DEFAULT_HORIZONS, target_name
from services.ml.training import HorizonReport, ModelReport, TrainingReport, run

__all__ = [
    "DEFAULT_HORIZONS",
    "HorizonReport",
    "Metrics",
    "ModelReport",
    "NaiveLag1",
    "RegisteredModel",
    "TimeSplit",
    "TrainingReport",
    "build_ladder",
    "classical",
    "evaluation",
    "models",
    "preprocessing",
    "registry",
    "run",
    "selection",
    "splitting",
    "target_name",
    "targets",
    "training",
]
