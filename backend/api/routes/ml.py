"""ML routes — FEAT-05, explainability and FEAT-06 (Phases 7-9).

specs §8 names only ``POST /ml/predict``, which is Phase 9. The endpoints here
exist because the dependency order in ``tasks.md`` has the frontend reading
"Phase 7 endpoints": the Model Lab (tasks 10.12, 10.13) renders the trained
models with their hyperparameters, their MAE/RMSE/R² and their feature
importances, and something has to trigger a run.

**Training is synchronous, and that is a deliberate limit rather than an
oversight.** A full run over the demo dataset takes tens of seconds, and adding
a job queue would mean a fourth container for an operation an analyst performs
occasionally (specs §12 fixes the deployment at three). The endpoint is
therefore narrowable -- one horizon, a subset of the ladder, classical baselines
off -- so an interactive call can be made small, and the CLI
(``python -m scripts.train_models``) exists for the full run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, model_validator

from api.dependencies import SessionDep, SettingsDep
from services.ml import (
    explain,
    intervals,
    prediction,
    registry,
    serving,
    splitting,
    targets,
    training,
)
from services.ml import models as model_zoo

router = APIRouter(prefix="/ml", tags=["ml"])

LADDER_NAMES = tuple(spec.name for spec in model_zoo.build_ladder())


# --- Requests (SEC-1) -------------------------------------------------------


class TrainingRequest(BaseModel):
    """What to train, over which slice."""

    source_ids: list[int] | None = Field(
        default=None, description="Restrict to these data sources. Omit for all."
    )
    start: datetime | None = Field(default=None, description="Inclusive lower bound.")
    end: datetime | None = Field(default=None, description="Inclusive upper bound.")
    horizons: list[int] | None = Field(
        default=None,
        description=f"Hours ahead. Defaults to {list(targets.DEFAULT_HORIZONS)}.",
    )
    test_fraction: float = Field(
        default=splitting.DEFAULT_TEST_FRACTION,
        gt=0.05,
        lt=0.9,
        description="Share of the period held out, chronologically (AC-8).",
    )
    cv_splits: int = Field(
        default=training.DEFAULT_CV_SPLITS,
        ge=2,
        le=10,
        description="Expanding-window folds over the training portion.",
    )
    models: list[str] | None = Field(
        default=None,
        description=(
            f"Subset of {list(LADDER_NAMES)}. The baseline is always included: "
            "a score without it cannot be judged (AC-7)."
        ),
    )
    include_classical: bool = Field(
        default=False,
        description="Add the ARIMA/Prophet baselines. They dominate the runtime.",
    )
    include_prophet: bool = Field(
        default=True, description="Only consulted when include_classical is set."
    )
    persist: bool = Field(
        default=True,
        description=(
            "Write artifacts and registry rows (task 7.12). Named for symmetry "
            "with the other endpoints -- and because `register` shadows an "
            "attribute of pydantic's BaseModel."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> TrainingRequest:
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        if self.horizons is not None:
            if not self.horizons:
                raise ValueError("horizons must not be empty")
            if any(horizon < 1 or horizon > 168 for horizon in self.horizons):
                raise ValueError("each horizon must be between 1 and 168 hours")
        if self.models:
            unknown = sorted(set(self.models) - set(LADDER_NAMES))
            if unknown:
                raise ValueError(f"unknown models {unknown}; available: {list(LADDER_NAMES)}")
        return self


# --- Responses (SEC-1) ------------------------------------------------------


class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    r2: float
    rows: int


class ModelResultResponse(BaseModel):
    name: str
    is_baseline: bool
    metrics: MetricsResponse
    cross_validation: dict[str, Any]
    skill_vs_baseline: float | None = Field(
        default=None,
        description=(
            "Fractional MAE improvement over persistence. 0.2 is 20% less error; "
            "negative means worse than doing nothing."
        ),
    )
    hyperparameters: dict[str, Any]
    notes: str
    fit_seconds: float
    scaler_audit: dict[str, Any] | None = Field(
        default=None, description="Evidence the scaler saw training rows only (AC-8)."
    )
    registered: dict[str, Any] | None = None
    unavailable: str | None = None


class ClassicalResponse(BaseModel):
    name: str
    available: bool
    origins: int
    note: str
    metrics: MetricsResponse | None


class HorizonResponse(BaseModel):
    horizon_hours: int
    target: str
    rows_modelled: int
    rows_dropped: int
    split: dict[str, Any]
    leakage_audit: dict[str, Any] = Field(
        description=(
            "Computed, not asserted: last training timestamp, first test "
            "timestamp, and the embargo between them (task 7.13)."
        )
    )
    feature_selection: dict[str, Any]
    models: list[ModelResultResponse]
    classical_baselines: list[ClassicalResponse]
    beats_baseline: bool = Field(
        description=(
            "AC-7: the production model's MAE is below the persistence "
            "baseline's. False also when the production model was not part of "
            "the run -- check the model list to tell an unjudged comparison "
            "from a lost one."
        )
    )


class TrainingResponse(BaseModel):
    """AC-7 and AC-8 in one payload."""

    generated_at: datetime
    window: dict[str, Any]
    feature_spec: dict[str, Any]
    leakage_clean: bool = Field(
        description="Every horizon's split passed every audit check (AC-8)."
    )
    beats_baseline: bool = Field(
        description="AC-7, judged at the 1-hour horizon the criterion names."
    )
    horizons: list[HorizonResponse]
    caveats: list[str]


class RegisteredModelResponse(BaseModel):
    id: int
    name: str
    target: str
    features_used: list[str] | dict[str, Any]
    mae: float | None
    rmse: float | None
    r2: float | None
    created_at: datetime
    artifact_path: str


# --- Explainability — Phase 8 (design §10.2) --------------------------------


class FeatureImportanceResponse(BaseModel):
    feature: str
    mean_abs_shap: float = Field(
        description="Average magnitude of this feature's effect, in µg/m³."
    )
    mean_shap: float = Field(
        description="Average signed effect. Near zero with a large magnitude "
        "means the feature pushes both ways depending on its value."
    )
    share: float = Field(description="Fraction of total attributed magnitude.")
    direction: str = Field(description="raises · lowers · mixed")


class ImportanceResponse(BaseModel):
    """Task 8.2: what the Model Lab's feature-importance chart renders."""

    model_id: int
    name: str
    target: str
    horizon_hours: int
    trained_at: str | None
    method: str = Field(
        description=(
            "How the attribution was computed. Tree models use SHAP; Ridge and "
            "the persistence baseline are solved exactly in closed form."
        )
    )
    base_value: float = Field(
        description="The model's expected output before any feature moves it."
    )
    rows: int
    features: list[FeatureImportanceResponse]
    cached: bool
    caveat: str = Field(description="ETH-1: attribution describes the model, not the air.")


# --- Routes -----------------------------------------------------------------


@router.post(
    "/train",
    response_model=TrainingResponse,
    summary="Train the model ladder and register the results",
)
async def train(
    session: SessionDep, settings: SettingsDep, request: TrainingRequest | None = None
) -> TrainingResponse:
    """Train Naive/Ridge/RF/XGBoost at each horizon (FEAT-05, AC-7, AC-8).

    Returns 422 when the window holds too little history to split into a
    training and a test period -- a request problem, and one a wider window
    fixes.
    """
    request = request or TrainingRequest()

    report = training.run(
        session,
        settings,
        source_ids=tuple(request.source_ids) if request.source_ids else None,
        start=request.start,
        end=request.end,
        horizons=tuple(request.horizons) if request.horizons else targets.DEFAULT_HORIZONS,
        test_fraction=request.test_fraction,
        cv_splits=request.cv_splits,
        model_names=tuple(request.models) if request.models else None,
        include_classical=request.include_classical,
        include_prophet=request.include_prophet,
        register=request.persist,
    )
    return TrainingResponse.model_validate(report.as_dict())


@router.get(
    "/models",
    response_model=list[RegisteredModelResponse],
    summary="The model registry, newest first",
)
async def list_models(
    session: SessionDep,
    target: Annotated[
        str | None, Query(description="Filter to one target, e.g. pm25_h1.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RegisteredModelResponse]:
    """What the Model Lab table renders (task 10.12).

    Every rung of the ladder is listed, not just the production model: the
    comparison *is* the point, and a table showing only the winner would hide
    the baseline it has to beat.
    """
    rows = registry.list_models(session, target=target, limit=limit)
    return [
        RegisteredModelResponse(
            id=row.id,
            name=row.name,
            target=row.target,
            features_used=row.features_used,
            mae=row.mae,
            rmse=row.rmse,
            r2=row.r2,
            created_at=row.created_at,
            artifact_path=row.artifact_path,
        )
        for row in rows
    ]


# --- Prediction — FEAT-06 (Phase 9) ----------------------------------------


class PredictionRequest(BaseModel):
    """Location, time and horizon (design §11)."""

    lat: float = Field(ge=-90, le=90, description="Decimal degrees (DR-3).")
    lon: float = Field(ge=-180, le=180, description="Decimal degrees (DR-3).")
    horizon: int = Field(
        default=1,
        description=(
            f"Hours ahead. A model must be registered for it; "
            f"{list(targets.DEFAULT_HORIZONS)} are trained by default."
        ),
    )
    at: datetime | None = Field(
        default=None,
        description=(
            "Origin of the forecast. Defaults to the newest observation, which "
            "is what 'now' means to this platform — in demo mode the wall clock "
            "can run ahead of the seeded data."
        ),
    )
    coverage: float = Field(
        default=intervals.DEFAULT_COVERAGE,
        ge=intervals.MIN_COVERAGE,
        le=intervals.MAX_COVERAGE,
        description="Interval coverage, e.g. 0.8 for an 80% interval.",
    )
    top_features: int = Field(
        default=explain.DEFAULT_TOP_FEATURES,
        ge=1,
        le=20,
        description="How many attributions to return with the forecast.",
    )
    model_name: str | None = Field(
        default=None,
        description=(
            f"Which rung to serve from. Defaults to {model_zoo.PRODUCTION_MODEL}; "
            "the registry holds the whole ladder so the others can be compared."
        ),
    )
    persist: bool = Field(
        default=True, description="Write the forecast to `predictions` (task 9.5)."
    )

    @model_validator(mode="after")
    def _check(self) -> PredictionRequest:
        if self.horizon < 1 or self.horizon > 168:
            raise ValueError("horizon must be between 1 and 168 hours")
        if self.model_name and self.model_name not in LADDER_NAMES:
            raise ValueError(
                f"unknown model {self.model_name!r}; available: {list(LADDER_NAMES)}"
            )
        return self


class PredictionResponse(BaseModel):
    """AC-9: the four fields specs §8 specifies, plus provenance.

    The first four keys are the spec's contract, spelled exactly as the example
    writes them — including ``"ug/m3"`` in ASCII and the interval as a
    two-element array. Everything after them is additional context a caller may
    ignore; nothing the spec names has been renamed, reshaped or nested.
    """

    prediction: float
    unit: str
    confidence_interval: list[float] = Field(
        min_length=2, max_length=2, description="[lower, upper]"
    )
    top_features: dict[str, float] = Field(
        description=(
            "Signed SHAP contributions in µg/m³, largest first. Positive pushed "
            "the forecast up, negative pulled it down."
        )
    )

    target_time: datetime
    origin_time: datetime
    horizon_hours: int
    lat: float
    lon: float
    coverage: float
    interval_method: str = Field(
        description=(
            "conformal_residual_quantiles (calibrated on held-out errors) or "
            "rmse_normal_approximation (fallback for a model registered before "
            "calibration existed)."
        )
    )
    base_value: float
    model: dict[str, Any]
    prediction_id: int | None
    caveats: list[str]


class BackfillResponse(BaseModel):
    matched: int
    scanned: int
    still_pending: int


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="PM2.5 forecast with its interval and its reasoning",
)
async def predict(
    session: SessionDep, settings: SettingsDep, request: PredictionRequest
) -> PredictionResponse:
    """The prediction endpoint (FEAT-06, AC-9).

    Returns 409 when no model is registered for the horizon, and 422 when the
    station has too little history to build every feature — the difference
    matters, because the first is fixed by training and the second by waiting.
    """
    result = prediction.predict(
        session,
        settings,
        lat=request.lat,
        lon=request.lon,
        horizon=request.horizon,
        at=request.at,
        coverage=request.coverage,
        model_name=request.model_name,
        persist=request.persist,
        top_features=request.top_features,
    )
    return PredictionResponse.model_validate(result.as_dict(request.top_features))


@router.post(
    "/predictions/backfill",
    response_model=BackfillResponse,
    summary="Fill in the outcome of forecasts whose hour has passed",
)
async def backfill(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=50_000)] = 5_000,
) -> BackfillResponse:
    """Task 9.6 — what turns `predictions` into the drift-monitoring dataset.

    Matching is on station *and* hour: pairing a forecast for one district with
    a reading from another would record the difference as model error.
    """
    return BackfillResponse.model_validate(
        prediction.backfill_actuals(session, limit=limit).as_dict()
    )


@router.get(
    "/models/{model_id}/importance",
    response_model=ImportanceResponse,
    summary="Global feature importance for one registered model",
)
async def model_importance(
    session: SessionDep,
    settings: SettingsDep,
    model_id: int,
    sample_rows: Annotated[
        int,
        Query(
            ge=50,
            le=5000,
            description=(
                "Rows sampled for the summary (task 8.4). The ranking settles "
                "by ~100; cost is linear in rows and in tree depth."
            ),
        ),
    ] = explain.DEFAULT_SAMPLE_ROWS,
    perturbation: Annotated[
        str,
        Query(
            description=(
                "tree_path_dependent follows the trees' own splits and needs no "
                "reference data; interventional integrates over a background "
                "sample, which is truer to the data and slower."
            )
        ),
    ] = explain.PATH_DEPENDENT,
    use_cache: Annotated[bool, Query()] = True,
) -> ImportanceResponse:
    """What the model relies on overall (tasks 8.1, 8.2, 8.4).

    Returns 404 when the model or its artifact is missing, and 503 when SHAP
    cannot be loaded — the difference matters, because only the first is
    fixable by changing the request.
    """
    if perturbation not in explain.PERTURBATIONS:
        raise RequestValidationError(
            [
                {
                    "loc": ("query", "perturbation"),
                    "msg": f"must be one of {list(explain.PERTURBATIONS)}",
                    "type": "value_error",
                }
            ]
        )

    loaded, importance, cached = serving.global_importance(
        session,
        settings,
        model_id=model_id,
        sample_rows=sample_rows,
        perturbation=perturbation,
        use_cache=use_cache,
    )

    return ImportanceResponse.model_validate(
        {**loaded.as_dict(), **importance.as_dict(), "cached": cached}
    )


__all__ = [
    "LADDER_NAMES",
    "ImportanceResponse",
    "PredictionRequest",
    "PredictionResponse",
    "RegisteredModelResponse",
    "TrainingRequest",
    "TrainingResponse",
    "router",
]
