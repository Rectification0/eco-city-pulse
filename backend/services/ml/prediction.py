"""The prediction service — FEAT-06 (tasks 9.1-9.6).

One forecast, assembled the same way the training rows were, scored by the
registered model, bounded by an interval calibrated on that model's own held-out
errors, explained by SHAP, and written down.

**Every step is a reuse, and that is the point.** Nothing here rebuilds a
feature, re-derives a column order, or re-decides a log transform:

* **9.1** the lag-feature fetcher is ``features.service.features_at`` (Phase 5),
  which loads only ``history_hours`` of context and builds the row with the same
  transformer the model was trained with;
* **9.2** the artifact comes from ``ml.serving.load_latest`` (Phase 8), which
  refuses a model whose feature fingerprint has drifted;
* **9.3** the interval comes from residual quantiles frozen into that artifact
  at training time (``ml.intervals``);
* **9.4** the attribution comes from ``ml.explain``, additive by construction.

A second implementation of any of those would be a second thing to keep in
sync, and the failure mode would be silent: a serving path that built features
slightly differently would return confident numbers that no longer mean what the
model learned. Phase 5 spent real effort making the two paths byte-identical;
this module exists to *use* that, not to work around it.

**The origin and the target are different times.** A request names *when you are
standing* (``at``) and *how far ahead* (``horizon``). The features describe the
origin, the forecast describes ``at + horizon``, and the row written to
``predictions`` is stamped with the target. Conflating them is the one mistake
that would make every stored prediction unscoreable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import and_, select, update
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from core.exceptions import InsufficientDataError
from db.models import Observation, Prediction
from services.datasets import station_key
from services.features import service as feature_service
from services.ml import explain, intervals, preprocessing, serving
from services.ml.explain import Attribution
from services.ml.serving import LoadedModel
from services.ml.targets import target_name

# specs §8 gives the unit as the ASCII "ug/m3" in the example response. Matched
# exactly rather than prettified to µg/m³: AC-9 is a contract test, and a
# contract that renders differently is a contract that was not met.
UNIT = "ug/m3"

# Rounding for the values that reach the API. Four decimals on a µg/m³ reading
# is already past the precision of any sensor; it exists so the contract test
# compares stable numbers.
DIGITS = 4

PREDICTION_CAVEAT = (
    "A forecast from one city's sensor network over one training window. "
    "Sensor placement is not uniform, so accuracy varies by district, and the "
    "attribution explains the model's output rather than the atmosphere (ETH-1)."
)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One forecast with everything needed to judge it (AC-9)."""

    prediction: float
    unit: str
    confidence_interval: tuple[float, float]
    attribution: Attribution
    coverage: float
    interval_method: str
    model: LoadedModel
    origin_time: datetime
    target_time: datetime
    lat: float
    lon: float
    prediction_id: int | None = None

    def top_features(self, limit: int = explain.DEFAULT_TOP_FEATURES) -> dict[str, float]:
        return self.attribution.top_features(limit)

    def as_contract(
        self, limit: int = explain.DEFAULT_TOP_FEATURES
    ) -> dict[str, Any]:
        """The exact shape specs §8 specifies, and nothing reordered.

        ``prediction``, ``unit``, ``confidence_interval``, ``top_features`` --
        the four keys of the example response, with the interval as a two-element
        list rather than an object because that is how the spec writes it.
        """
        return {
            "prediction": round(self.prediction, DIGITS),
            "unit": self.unit,
            "confidence_interval": [
                round(self.confidence_interval[0], DIGITS),
                round(self.confidence_interval[1], DIGITS),
            ],
            "top_features": self.top_features(limit),
        }

    def as_dict(self, limit: int = explain.DEFAULT_TOP_FEATURES) -> dict[str, Any]:
        """The contract, plus the provenance a caller needs to trust it."""
        return {
            **self.as_contract(limit),
            "target_time": self.target_time.isoformat(),
            "origin_time": self.origin_time.isoformat(),
            "horizon_hours": self.model.horizon,
            "lat": self.lat,
            "lon": self.lon,
            "coverage": self.coverage,
            "interval_method": self.interval_method,
            "base_value": round(self.attribution.base_value, DIGITS),
            "model": {
                "id": self.model.row.id,
                "name": self.model.row.name,
                "target": self.model.row.target,
                "trained_at": self.model.artifact.get("trained_at"),
                "mae": self.model.row.mae,
            },
            "prediction_id": self.prediction_id,
            "caveats": [PREDICTION_CAVEAT, intervals.INTERVAL_CAVEAT],
        }


def _as_utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def latest_observation_time(
    session: Session, *, lat: float | None = None, lon: float | None = None
) -> datetime | None:
    """The newest hour on record, which is what "now" means to this platform.

    A request that names no time wants a forecast from the present, and the
    present is the last reading that arrived -- not the wall clock, which in
    demo mode may be hours ahead of the seeded data.
    """
    statement = select(Observation.timestamp).order_by(Observation.timestamp.desc())
    if lat is not None and lon is not None:
        statement = statement.where(Observation.lat == lat, Observation.lon == lon)
    return session.scalars(statement.limit(1)).first()


def predict(
    session: Session,
    settings: Settings | None = None,
    *,
    lat: float,
    lon: float,
    horizon: int,
    at: datetime | None = None,
    coverage: float = intervals.DEFAULT_COVERAGE,
    model_name: str | None = None,
    persist: bool = True,
    top_features: int = explain.DEFAULT_TOP_FEATURES,
) -> PredictionResult:
    """Forecast PM2.5 at ``lat/lon`` for ``at + horizon`` hours (FEAT-06, AC-9)."""
    settings = settings or get_settings()
    intervals.validate_coverage(coverage)

    # --- 9.2: the registered model for this horizon -------------------------
    loaded = (
        serving.load_latest(session, target_name(horizon), name=model_name)
        if model_name
        else serving.load_latest(session, target_name(horizon))
    )

    origin = _as_utc(at) if at is not None else None
    if origin is None:
        newest = latest_observation_time(session, lat=lat, lon=lon)
        if newest is None:
            raise InsufficientDataError(
                "No observations are available to forecast from.",
                details={"lat": lat, "lon": lon},
            )
        origin = _as_utc(newest)

    origin = origin.replace(minute=0, second=0, microsecond=0)

    # --- 9.1: the features for that station at that hour --------------------
    row = feature_service.features_at(
        session, loaded.transformer, at=origin, lat=lat, lon=lon
    )
    matrix = preprocessing.encode(row, loaded.spec)[list(loaded.columns)]

    if matrix.isna().to_numpy().any():
        missing = [
            column for column in matrix.columns if bool(matrix[column].isna().iloc[0])
        ]
        raise InsufficientDataError(
            "Not enough history at this location to build every feature the "
            "model needs. A station needs its warm-up hours before it can be "
            "forecast from.",
            details={"missing_features": missing, "at": origin.isoformat()},
        )

    # --- 9.4 + 8.3: the value and the reasoning behind it -------------------
    explainer = explain.for_artifact(loaded.artifact)
    attribution = explainer.explain_one(matrix)
    value = float(attribution.prediction)

    # --- 9.3: the interval ---------------------------------------------------
    bounds, method = interval_for(loaded, value, coverage)

    target_time = origin + timedelta(hours=loaded.horizon)

    result = PredictionResult(
        prediction=value,
        unit=UNIT,
        confidence_interval=bounds,
        attribution=attribution,
        coverage=coverage,
        interval_method=method,
        model=loaded,
        origin_time=origin,
        target_time=target_time,
        lat=lat,
        lon=lon,
    )

    # --- 9.5: write it down --------------------------------------------------
    if persist:
        stored = Prediction(
            model_id=loaded.row.id,
            target_time=target_time,
            predicted_value=value,
            lat=lat,
            lon=lon,
        )
        session.add(stored)
        session.flush()
        result = replace(result, prediction_id=stored.id)

    return result


def interval_for(
    loaded: LoadedModel, prediction: float, coverage: float
) -> tuple[tuple[float, float], str]:
    """The interval, and which method produced it (task 9.3).

    The conformal quantiles are preferred. An artifact registered before
    calibration existed falls back to a normal approximation around its RMSE --
    weaker, because it assumes unbiased symmetric errors, so the method is named
    in the response rather than the two being presented as interchangeable.
    """
    payload = loaded.artifact.get("calibration")
    if payload:
        calibration = intervals.Calibration.from_dict(payload)
        if calibration.quantiles:
            return calibration.interval(prediction, coverage), intervals.CONFORMAL

    rmse = float((loaded.artifact.get("metrics") or {}).get("rmse") or 0.0)
    return intervals.normal_interval(prediction, rmse, coverage), intervals.NORMAL


# --- 9.6: backfilling the outcome -------------------------------------------


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """What one backfill pass resolved."""

    matched: int
    scanned: int
    still_pending: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "scanned": self.scanned,
            "still_pending": self.still_pending,
        }


def backfill_actuals(
    session: Session, *, limit: int = 5_000, now: datetime | None = None
) -> BackfillReport:
    """Fill ``actual_value`` where the forecast hour has since been observed.

    This is what turns ``predictions`` from a log into the drift-monitoring
    dataset design §6.1 wants: once a row has both numbers, the error of a
    *deployed* model on data it never trained on is measurable directly.

    Matching is on station **and** hour. Matching on hour alone would pair a
    forecast for one district with a reading from another and record the
    difference as model error, which is worse than leaving the row unscored.
    """
    now = _as_utc(now) if now else datetime.now(timezone.utc)

    pending = list(
        session.scalars(
            select(Prediction)
            .where(
                and_(
                    Prediction.actual_value.is_(None),
                    Prediction.target_time <= now,
                    Prediction.lat.is_not(None),
                    Prediction.lon.is_not(None),
                )
            )
            .order_by(Prediction.target_time)
            .limit(limit)
        )
    )

    if not pending:
        return BackfillReport(matched=0, scanned=0, still_pending=0)

    observed = _observations_for(session, pending)

    matched = 0
    for row in pending:
        key = (
            station_key(row.lat, row.lon),
            row.target_time.replace(minute=0, second=0, microsecond=0),
        )
        actual = observed.get(key)
        if actual is None:
            continue
        session.execute(
            update(Prediction).where(Prediction.id == row.id).values(actual_value=actual)
        )
        matched += 1

    return BackfillReport(
        matched=matched, scanned=len(pending), still_pending=len(pending) - matched
    )


def _observations_for(
    session: Session, pending: list[Prediction]
) -> dict[tuple[str, datetime], float]:
    """PM2.5 by (station, hour) across the window the pending rows span.

    One query for the whole batch rather than one per row: a backfill pass over
    a day of forecasts would otherwise be thousands of round trips.
    """
    earliest = min(row.target_time for row in pending)
    latest = max(row.target_time for row in pending)

    rows = session.execute(
        select(
            Observation.lat, Observation.lon, Observation.timestamp, Observation.pm25
        ).where(
            and_(
                Observation.timestamp >= earliest,
                Observation.timestamp <= latest,
                Observation.pm25.is_not(None),
            )
        )
    ).all()

    return {
        (
            station_key(lat, lon),
            pd.Timestamp(timestamp).to_pydatetime().replace(
                minute=0, second=0, microsecond=0
            ),
        ): float(pm25)
        for lat, lon, timestamp, pm25 in rows
    }


def scored_predictions(
    session: Session, *, model_id: int | None = None, limit: int = 500
) -> list[Prediction]:
    """Predictions that now have an outcome — the drift dataset."""
    statement = (
        select(Prediction)
        .where(Prediction.actual_value.is_not(None))
        .order_by(Prediction.target_time.desc())
        .limit(limit)
    )
    if model_id is not None:
        statement = statement.where(Prediction.model_id == model_id)
    return list(session.scalars(statement))


__all__ = [
    "DIGITS",
    "PREDICTION_CAVEAT",
    "UNIT",
    "BackfillReport",
    "PredictionResult",
    "backfill_actuals",
    "interval_for",
    "latest_observation_time",
    "predict",
    "scored_predictions",
]
