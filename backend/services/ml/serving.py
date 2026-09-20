"""Loading a registered model and rebuilding its input (Phase 8; feeds Phase 9).

Explaining a model requires the same thing predicting with it does: the
artifact, and a feature matrix built exactly the way the model was trained.
That shared need is this module, so Phase 9's ``/ml/predict`` extends it rather
than growing a second, slightly-different replay path.

**The feature contract is replayed, not reconstructed.** The artifact carries
the Phase 5 ``FeatureSpec`` it was trained with, including the frozen
log-transform decision, and the matrix is built from *that* spec rather than
from whatever the current default happens to be. A model trained before a spec
change therefore keeps working and keeps meaning the same thing, and the
fingerprint check catches the case where it cannot.

**A mismatch is refused, loudly.** A model file and a feature spec that have
drifted apart do not fail -- they produce confident nonsense, which is worse.
So the columns the artifact names must all be present, and the fingerprint the
artifact recorded must match the spec it carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from core.exceptions import ModelNotFoundError, ModelNotTrainedError
from db.models import MLModel
from services import datasets
from services.eda import cache
from services.features import service as feature_service
from services.features.spec import FeatureSpec
from services.features.transformer import FeatureTransformer
from services.ml import explain, preprocessing, registry
from services.ml.models import PRODUCTION_MODEL


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A registry row and the artifact it points at, checked against each other."""

    row: MLModel
    artifact: dict[str, Any]

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec.from_dict(self.artifact["feature_spec"])

    @property
    def transformer(self) -> FeatureTransformer:
        """The Phase 5 transformer, rebuilt from the frozen spec.

        Unfitted in the sense that ``fit`` was never called on it here -- and it
        does not need to be. The only thing fitting decides is which pollutants
        to log, and that decision is already inside the spec.
        """
        return FeatureTransformer(spec=self.spec)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.artifact["columns"])

    @property
    def horizon(self) -> int:
        return int(self.artifact["horizon_hours"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.row.id,
            "name": self.row.name,
            "target": self.row.target,
            "horizon_hours": self.horizon,
            "trained_at": self.artifact.get("trained_at"),
            "feature_fingerprint": self.artifact.get("feature_fingerprint"),
            "metrics": self.artifact.get("metrics"),
            "hyperparameters": self.artifact.get("hyperparameters"),
        }


def load(session: Session, model_id: int) -> LoadedModel:
    """Load one registered model by id."""
    row = session.get(MLModel, model_id)
    if row is None:
        raise ModelNotFoundError(
            f"No model registered with id {model_id}.", details={"model_id": model_id}
        )
    return _checked(row)


def load_latest(
    session: Session, target: str, *, name: str | None = PRODUCTION_MODEL
) -> LoadedModel:
    """The current model for a target (design §6.1).

    Defaults to the production model by name rather than to the newest row: a
    training run registers the whole ladder, so "newest" alone would make the
    served model an accident of iteration order.
    """
    row = registry.latest_for_target(session, target, name=name)
    if row is None:
        raise ModelNotTrainedError(
            f"No model has been registered for {target}."
            + (f" (looking for {name!r})" if name else ""),
            details={"target": target, "model": name},
        )
    return _checked(row)


def _checked(row: MLModel) -> LoadedModel:
    """Load the artifact and verify it agrees with itself."""
    artifact = registry.load_artifact(row.artifact_path)

    spec = FeatureSpec.from_dict(artifact["feature_spec"])
    transformer = FeatureTransformer(spec=spec)
    recorded = artifact.get("feature_fingerprint")

    if recorded and recorded != transformer.fingerprint:
        raise ModelNotFoundError(
            "The stored feature spec does not match the fingerprint recorded "
            "with this model; it cannot be replayed safely. Retrain it.",
            details={
                "model_id": row.id,
                "recorded": recorded,
                "computed": transformer.fingerprint,
            },
        )

    return LoadedModel(row=row, artifact=artifact)


def build_matrix(
    session: Session,
    loaded: LoadedModel,
    *,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild this model's input over a slice of ``observations``.

    Returns the featured frame (for context -- timestamps, stations) and the
    model matrix (for the model). The path is the training path: repair short
    gaps, apply the transformer, encode, then take the artifact's columns in
    the artifact's order.
    """
    frame = datasets.load_observations(
        session, source_ids=source_ids, start=start, end=end, limit=limit
    )
    datasets.require_rows(frame, minimum=1, what="model explanation")

    featured, _ = feature_service.build(frame, loaded.transformer)
    encoded = preprocessing.encode(featured, loaded.spec)

    missing = [column for column in loaded.columns if column not in encoded.columns]
    if missing:
        raise ModelNotFoundError(
            "The current feature pipeline cannot produce every column this "
            "model was trained on; it cannot be replayed safely.",
            details={"model_id": loaded.row.id, "missing": missing},
        )

    matrix = encoded[list(loaded.columns)]
    usable = matrix.notna().all(axis=1)
    return featured.loc[usable], matrix.loc[usable]


def explain_at(
    session: Session,
    loaded: LoadedModel,
    *,
    at: datetime,
    lat: float | None = None,
    lon: float | None = None,
    station: str | None = None,
    source_ids: tuple[int, ...] | None = None,
    perturbation: str = explain.PATH_DEPENDENT,
) -> tuple[float, explain.Attribution]:
    """One forecast and the reasoning behind it (task 8.3).

    The features come from the Phase 5 inference path -- the same transformer,
    over only the history that hour needs -- so the row explained here is the
    row the model was trained to read. Phase 9's ``/ml/predict`` calls this and
    puts ``attribution.top_features()`` in its response; it is separated out
    because the reasoning is a property of the model, not of the endpoint.
    """
    row = feature_service.features_at(
        session,
        loaded.transformer,
        at=at,
        lat=lat,
        lon=lon,
        station=station,
        source_ids=source_ids,
    )
    matrix = preprocessing.encode(row, loaded.spec)[list(loaded.columns)]

    explainer = explain.for_artifact(loaded.artifact, perturbation=perturbation)
    attribution = explainer.explain_one(matrix)

    return attribution.prediction, attribution


def global_importance(
    session: Session,
    settings: Settings | None = None,
    *,
    model_id: int,
    source_ids: tuple[int, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    sample_rows: int = explain.DEFAULT_SAMPLE_ROWS,
    perturbation: str = explain.PATH_DEPENDENT,
    use_cache: bool = True,
) -> tuple[LoadedModel, explain.GlobalImportance, bool]:
    """What the Model Lab renders (task 8.2).

    Cached on the dataset fingerprint *and* the model id: the artifact is
    immutable once registered, so a cached summary can only go stale when the
    data underneath the sample changes -- which the fingerprint catches.
    """
    settings = settings or get_settings()
    # One provenance, chosen before anything is loaded, fingerprinted or
    # cached: the scope has to reach the cache key too, or two scopes share
    # one entry and each is served the other's answer.
    source_ids = datasets.resolve_source_ids(
        session, source_ids, settings=settings
    )
    loaded = load(session, model_id)

    version = cache.dataset_version(
        session, source_ids=source_ids, start=start, end=end
    )
    key = cache.PROFILE_CACHE.key(
        version,
        kind="shap",
        model_id=model_id,
        sample_rows=sample_rows,
        perturbation=perturbation,
        source_ids=list(source_ids or ()),
        start=start,
        end=end,
    )

    if use_cache:
        hit = cache.PROFILE_CACHE.get(key)
        if hit is not None:
            return loaded, hit, True

    background = None
    if perturbation == explain.INTERVENTIONAL:
        # The interventional perturbation integrates over a reference sample, so
        # it needs one; the path-dependent variant reads the trees themselves.
        _, background = build_matrix(
            session, loaded, source_ids=source_ids, start=start, end=end
        )
        background = background.iloc[:: max(1, len(background) // 200)]

    _, matrix = build_matrix(
        session, loaded, source_ids=source_ids, start=start, end=end
    )
    explainer = explain.for_artifact(
        loaded.artifact, perturbation=perturbation, background=background
    )
    importance = explainer.global_importance(matrix, sample_rows=sample_rows)

    if use_cache:
        cache.PROFILE_CACHE.set(key, importance)

    return loaded, importance, False


__all__ = [
    "LoadedModel",
    "build_matrix",
    "explain_at",
    "global_importance",
    "load",
    "load_latest",
]
