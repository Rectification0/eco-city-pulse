"""The training pipeline (Phase 7, FEAT-05, design §10).

    Raw Data
       ↓ Impute / Clean          (window-local, Phase 3/5)
       ↓ Time-Aware Split        (chronological, embargoed)
       ↓ Feature Selection       (fitted on train only)
       ↓ Scale / Encode          (scaler inside the model, train only)
       ↓ Train Models            (Naive Lag-1, Ridge, Random Forest, XGBoost)
       ↓ Cross-Validation        (expanding window, never K-Fold)
       ↓ Model Registry          (artifact + metrics → models)

**The order of the first two steps is the whole argument of this module.** The
Phase 5 transformer has one fitted parameter -- which pollutants to log -- and
fitting it on the full dataset would let the test period influence a decision
applied to training. So the split timestamp is computed *first*, from the raw
frame, and the transformer is fitted on the training side of it alone. It is
then applied to everything, which is safe precisely because every feature it
builds looks backward (Phase 5's guarantee, and what makes that guarantee worth
having).

**Three horizons, three independent problems** (specs §6.3). Each gets its own
split embargo, its own feature selection, its own models and its own registry
rows, because the 1-hour problem is nearly persistence and the 24-hour problem
is nearly climatology.

**A run reports its own leakage audit.** The trainer does not merely avoid
leakage; it computes the evidence -- last training timestamp, first test
timestamp, the gap between them, and the scaler's fitted mean against the
training mean -- and carries it in the report. A claim of no leakage that cannot
be inspected is a claim to be taken on faith, and AC-8 is not a matter of faith.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from core.exceptions import InsufficientDataError
from services import datasets
from services.datasets import DatasetWindow
from services.features import service as feature_service
from services.features.spec import FeatureSpec
from services.features.transformer import FeatureTransformer
from services.ml import (
    classical,
    evaluation,
    intervals,
    preprocessing,
    registry,
    selection,
    splitting,
    targets,
)
from services.ml import (
    models as model_zoo,
)
from services.ml.evaluation import Metrics

DEFAULT_CV_SPLITS = 3


@dataclass(frozen=True, slots=True)
class ModelReport:
    """One model at one horizon: what it scored, and how it was built."""

    name: str
    metrics: Metrics
    cv: dict[str, Any]
    skill_vs_baseline: float | None
    is_baseline: bool
    hyperparameters: dict[str, Any]
    notes: str
    fit_seconds: float
    scaler_audit: dict[str, Any] | None = None
    # Residual quantiles from the held-out window, frozen for Phase 9's
    # prediction intervals (task 9.3). None when the window was too short.
    calibration: intervals.Calibration | None = None
    registered: registry.RegisteredModel | None = None
    unavailable: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_baseline": self.is_baseline,
            "metrics": self.metrics.as_dict(),
            "cross_validation": self.cv,
            "skill_vs_baseline": self.skill_vs_baseline,
            "hyperparameters": self.hyperparameters,
            "notes": self.notes,
            "fit_seconds": round(self.fit_seconds, 2),
            "scaler_audit": self.scaler_audit,
            "calibration": self.calibration.as_dict() if self.calibration else None,
            "registered": self.registered.as_dict() if self.registered else None,
            "unavailable": self.unavailable,
        }


@dataclass(frozen=True, slots=True)
class HorizonReport:
    """Everything one forecast horizon produced."""

    horizon: int
    target: str
    rows_modelled: int
    rows_dropped: int
    split: dict[str, Any]
    leakage_audit: dict[str, Any]
    selection: dict[str, Any]
    models: tuple[ModelReport, ...]
    classical: tuple[classical.ClassicalResult, ...]

    @property
    def baseline(self) -> ModelReport | None:
        return next((report for report in self.models if report.is_baseline), None)

    @property
    def production(self) -> ModelReport | None:
        return next(
            (
                report
                for report in self.models
                if report.name == model_zoo.PRODUCTION_MODEL
            ),
            None,
        )

    @property
    def beats_baseline(self) -> bool:
        """AC-7 at this horizon: does the production model beat persistence?

        False when the production model was not part of the run at all -- a
        narrowed request, or XGBoost unavailable. That is an *unjudged*
        comparison rather than a lost one, and the model list distinguishes the
        two: a missing rung is absent, a beaten one is present with its score.
        """
        baseline, production = self.baseline, self.production
        if baseline is None or production is None:
            return False
        if production.unavailable is not None:
            return False
        return production.metrics.mae < baseline.metrics.mae

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_hours": self.horizon,
            "target": self.target,
            "rows_modelled": self.rows_modelled,
            "rows_dropped": self.rows_dropped,
            "split": self.split,
            "leakage_audit": self.leakage_audit,
            "feature_selection": self.selection,
            "models": [report.as_dict() for report in self.models],
            "classical_baselines": [item.as_dict() for item in self.classical],
            "beats_baseline": self.beats_baseline,
        }


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """One training run, across every horizon."""

    window: DatasetWindow
    transformer: FeatureTransformer
    horizons: tuple[HorizonReport, ...]
    generated_at: datetime
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def leakage_clean(self) -> bool:
        """Every horizon's split passes every audit check (AC-8)."""
        return all(
            all(
                report.leakage_audit[check]
                for check in (
                    "train_precedes_test",
                    "embargo_respected",
                    "no_overlapping_rows",
                    "no_shared_timestamps",
                )
            )
            for report in self.horizons
        )

    @property
    def beats_baseline(self) -> bool:
        """AC-7, judged at the 1-hour horizon the criterion names."""
        one_hour = next(
            (report for report in self.horizons if report.horizon == 1), None
        )
        return one_hour.beats_baseline if one_hour else False

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "window": self.window.as_dict(),
            "feature_spec": {
                "fingerprint": self.transformer.fingerprint,
                "features": list(self.transformer.feature_names),
                "log_columns": list(self.transformer.spec.log_columns),
                "fitted_on": self.transformer.fit_window.as_dict()
                if self.transformer.fit_window
                else None,
            },
            "leakage_clean": self.leakage_clean,
            "beats_baseline": self.beats_baseline,
            "horizons": [report.as_dict() for report in self.horizons],
            "caveats": list(self.caveats),
        }


# --- Preparation ------------------------------------------------------------


def training_cutoff(
    frame: pd.DataFrame, *, test_fraction: float, max_horizon: int
) -> datetime:
    """The timestamp the feature transformer may be fitted up to.

    The split point less the longest embargo, so the transformer is fitted on
    data that is training data at *every* horizon rather than at some of them.
    """
    moments = np.sort(frame["timestamp"].unique())
    if len(moments) < 2:
        raise InsufficientDataError(
            "Training needs at least two distinct timestamps.",
            details={"timestamps": len(moments)},
        )

    cut = int(len(moments) * (1.0 - test_fraction))
    cut = min(max(cut, 1), len(moments) - 1)
    return pd.Timestamp(moments[cut]).to_pydatetime() - timedelta(hours=max_horizon)


def prepare_matrix(
    frame: pd.DataFrame,
    transformer: FeatureTransformer,
    *,
    horizons: tuple[int, ...],
    max_gap_hours: int = feature_service.DEFAULT_MAX_GAP_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[targets.TargetSummary, ...]]:
    """Features, targets and the encoded model matrix, aligned row for row."""
    featured, _ = feature_service.build(frame, transformer, max_gap_hours=max_gap_hours)
    featured, summaries = targets.add_targets(featured, horizons)
    matrix = preprocessing.encode(featured, transformer.spec)
    return featured, matrix, summaries


# --- One horizon ------------------------------------------------------------


def _fit_and_score(
    spec: model_zoo.ModelSpec,
    matrix: pd.DataFrame,
    target: pd.Series,
    split: splitting.TimeSplit,
    *,
    folds: list[tuple[np.ndarray, np.ndarray]],
    horizon: int,
) -> tuple[ModelReport, Any]:
    """Fit one model on train, cross-validate on train, score on test.

    Returns the report and the fitted estimator; the estimator is ``None`` when
    the model's library could not be loaded, which is a missing rung rather
    than a failed run.
    """
    train_x = matrix.iloc[split.train_index]
    train_y = target.iloc[split.train_index]
    test_x = matrix.iloc[split.test_index]
    test_y = target.iloc[split.test_index]

    started = time.perf_counter()

    try:
        model = preprocessing.with_scaler(spec.build(), scale=spec.scale)
        model.fit(train_x, train_y)
    except model_zoo.ModelUnavailableError as exc:
        return (
            ModelReport(
                name=spec.name,
                metrics=Metrics(
                    mae=float("nan"), rmse=float("nan"), r2=float("nan"), rows=0
                ),
                cv={"folds": 0},
                skill_vs_baseline=None,
                is_baseline=spec.is_baseline,
                hyperparameters=spec.hyperparameters,
                notes=spec.notes,
                fit_seconds=0.0,
                unavailable=exc.message,
            ),
            None,
        )

    test_predictions = model.predict(test_x)
    metrics = evaluation.score(test_y.to_numpy(), test_predictions)

    # --- 9.3: calibrate the prediction interval on the held-out window ------
    # The residuals a model made on data it was not fitted on are the only
    # honest basis for an interval, and this is the one place they exist. They
    # are read here and frozen into the artifact, because an interval is
    # meaningful only beside the model whose errors produced it.
    calibration = intervals.calibrate(
        test_y.to_numpy(),
        test_predictions,
        window=f"{split.test_start.isoformat()}..{split.test_end.isoformat()}",
    )

    # --- Cross-validation on the training portion only (task 7.10) ----------
    # The folds are computed once per horizon and shared by every model, so the
    # comparison between models is over identical splits rather than over
    # independently regenerated ones.
    fold_scores: list[Metrics] = []
    for fold_train, fold_test in folds:
        fold_model = preprocessing.with_scaler(spec.build(), scale=spec.scale)
        fold_model.fit(train_x.iloc[fold_train], train_y.iloc[fold_train])
        fold_scores.append(
            evaluation.score(
                train_y.iloc[fold_test].to_numpy(),
                fold_model.predict(train_x.iloc[fold_test]),
            )
        )

    # --- The scaler audit (task 7.13) ---------------------------------------
    scaler = preprocessing.fitted_scaler(model)
    scaler_audit = None
    if scaler is not None:
        train_means = train_x.mean().to_numpy()
        scaler_audit = {
            "fitted_on_train_only": bool(
                np.allclose(scaler.mean_, train_means, rtol=1e-9, atol=1e-9)
            ),
            "differs_from_full_set_mean": bool(
                not np.allclose(scaler.mean_, matrix.mean().to_numpy(), rtol=1e-9)
            ),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "train_rows": len(train_x),
        }

    return (
        ModelReport(
            name=spec.name,
            metrics=metrics,
            cv=evaluation.summarise_folds(fold_scores),
            skill_vs_baseline=None,
            is_baseline=spec.is_baseline,
            hyperparameters=spec.hyperparameters,
            notes=spec.notes,
            fit_seconds=time.perf_counter() - started,
            scaler_audit=scaler_audit,
            calibration=calibration,
        ),
        model,
    )


def train_horizon(
    session: Session,
    featured: pd.DataFrame,
    matrix: pd.DataFrame,
    transformer: FeatureTransformer,
    *,
    horizon: int,
    test_fraction: float,
    cv_splits: int,
    ladder: tuple[model_zoo.ModelSpec, ...],
    include_classical: bool,
    include_prophet: bool,
    register: bool,
    settings: Settings,
) -> HorizonReport:
    """Train, evaluate and register every model for one horizon."""
    target_column = targets.target_name(horizon)

    usable = matrix.notna().all(axis=1) & featured[target_column].notna()
    rows_dropped = int((~usable).sum())

    frame = featured.loc[usable].reset_index(drop=True)
    model_matrix = matrix.loc[usable].reset_index(drop=True)
    target = frame[target_column]

    # --- 7.2: chronological split with a horizon-length embargo -------------
    split = splitting.chronological_split(
        frame, test_fraction=test_fraction, embargo_hours=horizon
    )
    audit = splitting.audit(
        frame, split.train_index, split.test_index, embargo_hours=horizon
    )

    # --- 7.4: feature selection, fitted on the training rows alone ----------
    chosen = selection.select(
        model_matrix.iloc[split.train_index], target.iloc[split.train_index]
    )
    model_matrix = selection.apply(model_matrix, chosen)

    # --- 7.10: the expanding-window folds, built once for every model -------
    train_frame = frame.iloc[split.train_index].reset_index(drop=True)
    try:
        folds = list(
            splitting.expanding_window_splits(
                train_frame, n_splits=cv_splits, embargo_hours=horizon
            )
        )
    except InsufficientDataError:
        # Too short a training window to fold. The held-out test score still
        # stands; reporting zero folds beats reporting folds never run.
        folds = []

    # --- 7.5-7.8, 7.11 ------------------------------------------------------
    reports: list[ModelReport] = []
    fitted: dict[str, Any] = {}

    for spec in ladder:
        report, model = _fit_and_score(
            spec,
            model_matrix,
            target,
            split,
            folds=folds,
            horizon=horizon,
        )
        reports.append(report)
        if report.unavailable is None:
            fitted[spec.name] = model

    # Skill is relative, so it is filled once the baseline has been scored.
    baseline = next((report for report in reports if report.is_baseline), None)
    if baseline is not None:
        reports = [
            replace(
                report,
                skill_vs_baseline=evaluation.skill(report.metrics, baseline.metrics)
                if report.unavailable is None
                else None,
            )
            for report in reports
        ]

    # --- 7.9: classical baselines, which may never fail the run -------------
    classical_results: tuple[classical.ClassicalResult, ...] = ()
    if include_classical:
        classical_results = classical.run_baselines(
            frame.iloc[split.train_index],
            frame.iloc[split.test_index],
            horizon=horizon,
            include_prophet=include_prophet,
        )

    # --- 7.12: registry -----------------------------------------------------
    if register:
        registered = []
        for report in reports:
            if report.unavailable is not None:
                registered.append(report)
                continue
            entry = registry.register(
                session,
                name=report.name,
                target=target_column,
                features_used=list(chosen.kept),
                artifact=_artifact(
                    fitted[report.name],
                    report=report,
                    transformer=transformer,
                    columns=tuple(chosen.kept),
                    horizon=horizon,
                    target_column=target_column,
                ),
                mae=report.metrics.mae,
                rmse=report.metrics.rmse,
                r2=report.metrics.r2,
                settings=settings,
            )
            registered.append(replace(report, registered=entry))
        reports = registered

    return HorizonReport(
        horizon=horizon,
        target=target_column,
        rows_modelled=len(frame),
        rows_dropped=rows_dropped,
        split=split.as_dict(),
        leakage_audit=audit,
        selection=chosen.as_dict(),
        models=tuple(reports),
        classical=classical_results,
    )


def _artifact(
    model: Any,
    *,
    report: ModelReport,
    transformer: FeatureTransformer,
    columns: tuple[str, ...],
    horizon: int,
    target_column: str,
) -> dict[str, Any]:
    """Everything Phase 9 needs to rebuild this model's input and use it."""
    return {
        "estimator": model,
        "model_name": report.name,
        "target": target_column,
        "horizon_hours": horizon,
        "columns": list(columns),
        "feature_spec": transformer.spec.as_dict(),
        "feature_fingerprint": transformer.fingerprint,
        "hyperparameters": report.hyperparameters,
        "metrics": report.metrics.as_dict(),
        # Task 9.3: the interval travels with the model, because residual
        # quantiles only describe the model that produced them.
        "calibration": report.calibration.as_dict() if report.calibration else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


# --- The run ----------------------------------------------------------------


def run(
    session: Session,
    settings: Settings | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    horizons: tuple[int, ...] = targets.DEFAULT_HORIZONS,
    test_fraction: float = splitting.DEFAULT_TEST_FRACTION,
    cv_splits: int = DEFAULT_CV_SPLITS,
    model_names: tuple[str, ...] | None = None,
    include_classical: bool = True,
    include_prophet: bool = True,
    register: bool = True,
    spec: FeatureSpec | None = None,
) -> TrainingReport:
    """Train the ladder at every horizon and register the results (FEAT-05)."""
    settings = settings or get_settings()
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )

    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end
    )
    window = datasets.describe_window(frame, tuple(source_ids or ()))

    if len(frame) < splitting.MIN_TRAIN_ROWS + splitting.MIN_TEST_ROWS:
        raise InsufficientDataError(
            "Not enough observations to train: "
            f"{len(frame)} rows, need at least "
            f"{splitting.MIN_TRAIN_ROWS + splitting.MIN_TEST_ROWS}.",
            details={"rows": len(frame)},
        )

    # --- AC-8: the transformer is fitted on training data only --------------
    cutoff = training_cutoff(
        frame, test_fraction=test_fraction, max_horizon=max(horizons)
    )
    train_only = frame[frame["timestamp"] < cutoff]
    repaired, _ = feature_service.prepare(train_only)
    transformer = FeatureTransformer.fit(repaired, base_spec=spec or FeatureSpec())

    featured, matrix, _ = prepare_matrix(frame, transformer, horizons=horizons)

    ladder = model_zoo.build_ladder()
    if model_names:
        chosen = set(model_names) | {model_zoo.BASELINE_MODEL}
        ladder = tuple(item for item in ladder if item.name in chosen)

    reports = tuple(
        train_horizon(
            session,
            featured,
            matrix,
            transformer,
            horizon=horizon,
            test_fraction=test_fraction,
            cv_splits=cv_splits,
            ladder=ladder,
            include_classical=include_classical,
            include_prophet=include_prophet,
            register=register,
            settings=settings,
        )
        for horizon in horizons
    )

    return TrainingReport(
        window=window,
        transformer=transformer,
        horizons=reports,
        generated_at=datetime.now(timezone.utc),
        caveats=(
            evaluation.R2_CAVEAT,
            "Scores describe this city over this window. They are not a claim "
            "about any other city, period, or sensor network.",
        ),
    )


__all__ = [
    "DEFAULT_CV_SPLITS",
    "HorizonReport",
    "ModelReport",
    "TrainingReport",
    "prepare_matrix",
    "run",
    "train_horizon",
    "training_cutoff",
]
