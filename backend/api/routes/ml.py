"""ML routes — FEAT-05 (Phase 7).

specs §8 names only ``POST /ml/predict``, which is Phase 9. The two endpoints
here exist because the dependency order in ``tasks.md`` has the frontend
reading "Phase 7 endpoints": the Model Lab (task 10.12) renders the trained
models with their hyperparameters and MAE/RMSE/R², and something has to trigger
a run.

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
from pydantic import BaseModel, Field, model_validator

from api.dependencies import SessionDep, SettingsDep
from services.ml import models as model_zoo
from services.ml import registry, splitting, targets, training

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


__all__ = [
    "LADDER_NAMES",
    "RegisteredModelResponse",
    "TrainingRequest",
    "TrainingResponse",
    "router",
]
