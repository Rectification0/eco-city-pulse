"""PCA and the Environmental Stress Index — FEAT-04 (tasks 6.1-6.3, design §9).

The problem ESI solves is that a dashboard cannot show five correlated numbers
and call it a summary. PM2.5, PM10, temperature, humidity and traffic move
together; most of their joint variation lies along a single axis, and PCA finds
it. design §9 in three steps:

1. **Standardize** every continuous variable. PCA maximises variance, and
   variance is scale-dependent -- left in raw units, PM10 (tens to hundreds)
   would dominate humidity (a percentage) for no reason but its units.
2. **Fit PCA and take PC1**, the dominant axis of environmental variation.
3. **Normalize PC1 to 0-100** against the fit window, so the number is readable.

**Sign is chosen, not accepted.** A principal component is only defined up to
sign: the same data can yield PC1 or its exact negative depending on the LAPACK
build, and a score that silently inverts between runs would be worse than no
score. PC1 is therefore oriented so that it increases with PM2.5, which makes
"high ESI" mean "dirtier air" by construction rather than by luck.

**What ESI is not.** It is a *relative* position within the fitted window, not
an absolute or health-calibrated measure: 80 means "high for this city in this
period", never "unsafe". It describes; it does not explain, and it makes no
causal claim about what drove the value (ETH-1). The loadings ship with every
score precisely so the number cannot be read as a black box (task 6.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core.exceptions import InsufficientDataError
from services.datasets import MEASUREMENT_COLUMNS

# AC-6: the score is bounded, and the bounds are the contract.
ESI_MIN = 0.0
ESI_MAX = 100.0

# The column PC1 is oriented against. PM2.5 is the platform's target variable
# and the pollutant the whole system is built around, so "stress rises with
# PM2.5" is the orientation that makes the index mean what its name says.
ANCHOR_COLUMN = "pm25"

# Below this, a covariance matrix is being estimated from almost nothing and
# the components would be noise wearing a component's name.
MIN_ROWS = 30

# Deterministic by construction: the full SVD has no random initialisation, so
# the fit is reproducible without depending on a seed at all. The seed is set
# anyway, for the day someone switches to the randomized solver (task 6.7).
SVD_SOLVER = "full"
RANDOM_STATE = 42

ESI_CAVEAT = (
    "ESI is a relative position within the window it was fitted on, not an "
    "absolute or health-calibrated measure: 80 means 'high for this city in "
    "this period', not 'unsafe'. It summarises how several correlated readings "
    "moved together and makes no claim about what caused them (ETH-1)."
)

CLIPPING_CAVEAT = (
    "Scores are clipped to 0-100 (AC-6). A reading more extreme than anything "
    "in the fit window therefore saturates rather than exceeding the scale."
)


@dataclass(frozen=True, slots=True)
class ComponentLoadings:
    """One principal component, and what it is made of.

    ``loadings`` are the component's coefficients over the standardized
    columns: how much each variable contributes, and in which direction. They
    are what keeps the index interpretable (design §9, task 6.6) -- a PC1 that
    loads heavily on PM2.5 and traffic and negatively on temperature is a
    winter-inversion axis, and a reader can see that from the numbers.
    """

    index: int
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    loadings: dict[str, float]

    @property
    def drivers(self) -> list[tuple[str, float]]:
        """Columns by absolute contribution, strongest first."""
        return sorted(self.loadings.items(), key=lambda item: abs(item[1]), reverse=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "explained_variance_ratio": round(self.explained_variance_ratio, 4),
            "cumulative_variance_ratio": round(self.cumulative_variance_ratio, 4),
            "loadings": {name: round(value, 4) for name, value in self.loadings.items()},
            "drivers": [name for name, _ in self.drivers],
        }


@dataclass(frozen=True, slots=True)
class ESIModel:
    """A fitted standardizer + PCA + the 0-100 mapping (tasks 6.1-6.3).

    Serialisable, like the feature transformer of Phase 5 and for the same
    reason: a score means nothing without the window it was calibrated against,
    so the calibration travels with it.
    """

    columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    explained_variance_ratio: tuple[float, ...]
    pc1_min: float
    pc1_max: float
    anchor: str
    fit_rows: int
    fitted_at: datetime | None = None

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def loadings(self) -> tuple[ComponentLoadings, ...]:
        cumulative = 0.0
        result = []
        for index, (component, ratio) in enumerate(
            zip(self.components, self.explained_variance_ratio, strict=True)
        ):
            cumulative += ratio
            result.append(
                ComponentLoadings(
                    index=index + 1,
                    explained_variance_ratio=ratio,
                    cumulative_variance_ratio=cumulative,
                    loadings=dict(zip(self.columns, component, strict=True)),
                )
            )
        return tuple(result)

    @property
    def is_degenerate(self) -> bool:
        """True when PC1 has no spread to normalise against.

        Happens on a constant or near-constant window. The index is then
        undefined rather than zero, and saying so beats reporting a flat 0.
        """
        return not np.isfinite(self.pc1_max - self.pc1_min) or (
            self.pc1_max - self.pc1_min
        ) <= 0

    # --- Scoring ------------------------------------------------------------

    def project(self, frame: pd.DataFrame) -> np.ndarray:
        """Standardize with the fitted parameters, then project onto the components.

        The scaler is applied, never refitted: re-standardising a serving window
        against its own mean would make the score mean something different from
        one request to the next.
        """
        matrix = frame[list(self.columns)].to_numpy(dtype="float64")
        standardized = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        return standardized @ np.asarray(self.components).T

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """The ESI, on 0-100, for every row (task 6.3, AC-6)."""
        if frame.empty:
            return pd.Series(dtype="float64", index=frame.index)

        pc1 = self.project(frame)[:, 0]

        if self.is_degenerate:
            return pd.Series(np.full(len(frame), np.nan), index=frame.index)

        scaled = (pc1 - self.pc1_min) / (self.pc1_max - self.pc1_min)
        return pd.Series(
            np.clip(scaled * ESI_MAX, ESI_MIN, ESI_MAX), index=frame.index
        )

    # --- Serialisation ------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "means": list(self.means),
            "scales": list(self.scales),
            "components": [list(row) for row in self.components],
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "pc1_min": self.pc1_min,
            "pc1_max": self.pc1_max,
            "anchor": self.anchor,
            "fit_rows": self.fit_rows,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ESIModel:
        fitted_at = payload.get("fitted_at")
        return cls(
            columns=tuple(payload["columns"]),
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
            components=tuple(
                tuple(float(value) for value in row) for row in payload["components"]
            ),
            explained_variance_ratio=tuple(
                float(value) for value in payload["explained_variance_ratio"]
            ),
            pc1_min=float(payload["pc1_min"]),
            pc1_max=float(payload["pc1_max"]),
            anchor=str(payload.get("anchor", ANCHOR_COLUMN)),
            fit_rows=int(payload.get("fit_rows", 0)),
            fitted_at=datetime.fromisoformat(fitted_at) if fitted_at else None,
        )


@dataclass(frozen=True, slots=True)
class ReductionResult:
    """A fitted PCA, the projected rows, and the ESI derived from PC1."""

    model: ESIModel
    scores: pd.DataFrame
    rows_used: int
    rows_dropped: int
    flipped: bool
    caveats: tuple[str, ...]

    @property
    def esi_summary(self) -> dict[str, float | None]:
        series = self.scores["esi"].dropna()
        if series.empty:
            return {"mean": None, "median": None, "min": None, "max": None, "latest": None}
        return {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "latest": round(float(series.iloc[-1]), 2),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.model.columns),
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "components": [item.as_dict() for item in self.model.loadings],
            "esi": self.esi_summary,
            "pc1_oriented_by": self.model.anchor,
            "pc1_sign_flipped": self.flipped,
            "caveats": list(self.caveats),
        }


def usable_rows(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> tuple[pd.DataFrame, int]:
    """Rows with every selected column present.

    PCA has no notion of a missing value, so incomplete rows are excluded and
    *counted*. Dropping them silently would let an ESI computed from a tenth of
    the window look exactly like one computed from all of it.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")

    complete = frame[frame[list(columns)].notna().all(axis=1)]
    return complete, len(frame) - len(complete)


def fit(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    n_components: int | None = None,
    anchor: str = ANCHOR_COLUMN,
) -> tuple[ESIModel, bool]:
    """Standardize, fit PCA, orient PC1, and calibrate the 0-100 mapping.

    Returns the model and whether PC1's sign had to be flipped -- reported
    rather than hidden, because a reader comparing loadings across two runs
    deserves to know the axis was reoriented.
    """
    complete, _ = usable_rows(frame, columns)

    if len(complete) < MIN_ROWS:
        raise InsufficientDataError(
            f"PCA needs at least {MIN_ROWS} complete observations; "
            f"{len(complete)} available.",
            details={"required": MIN_ROWS, "available": len(complete)},
        )

    matrix = complete[list(columns)].to_numpy(dtype="float64")

    # --- 6.1: standardize ---------------------------------------------------
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    # A column with zero variance gets scale 1.0 from sklearn rather than a
    # division by zero; it then contributes nothing, which is correct.
    scales = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)

    # --- 6.2: fit PCA -------------------------------------------------------
    components = n_components or len(columns)
    pca = PCA(
        n_components=min(components, len(columns), len(complete)),
        svd_solver=SVD_SOLVER,
        random_state=RANDOM_STATE,
    )
    projected = pca.fit_transform(standardized)

    # --- Orientation --------------------------------------------------------
    anchor_index = columns.index(anchor) if anchor in columns else 0
    loading = pca.components_[0][anchor_index]
    flipped = bool(loading < 0)
    if flipped:
        # Flip the component *and* the projection together, so the stored
        # loadings always describe the axis the scores were measured on.
        pca.components_[0] = -pca.components_[0]
        projected[:, 0] = -projected[:, 0]

    # --- 6.3: calibrate the 0-100 mapping -----------------------------------
    pc1 = projected[:, 0]

    return (
        ESIModel(
            columns=tuple(columns),
            means=tuple(float(value) for value in scaler.mean_),
            scales=tuple(float(value) for value in scales),
            components=tuple(
                tuple(float(value) for value in row) for row in pca.components_
            ),
            explained_variance_ratio=tuple(
                float(value) for value in pca.explained_variance_ratio_
            ),
            pc1_min=float(pc1.min()),
            pc1_max=float(pc1.max()),
            anchor=columns[anchor_index],
            fit_rows=len(complete),
            fitted_at=datetime.now(timezone.utc),
        ),
        flipped,
    )


def reduce_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    n_components: int | None = None,
    model: ESIModel | None = None,
) -> ReductionResult:
    """Fit (or replay) the reduction and score every usable row (FEAT-04).

    Passing ``model`` scores new data against an existing calibration, which is
    what a dashboard wants: an ESI that moves because the air changed, not
    because the window did.
    """
    # A supplied model carries its own column set: scoring against a different
    # one would silently reorder the matrix it was calibrated on.
    columns = model.columns if model is not None else columns
    complete, dropped = usable_rows(frame, columns)

    flipped = False
    if model is None:
        model, flipped = fit(frame, columns=columns, n_components=n_components)

    projected = model.project(complete)
    scores = pd.DataFrame(
        {
            f"pc{index + 1}": projected[:, index]
            for index in range(projected.shape[1])
        },
        index=complete.index,
    )
    scores["esi"] = model.score(complete)

    for carried in ("timestamp", "station"):
        if carried in complete.columns:
            scores.insert(0, carried, complete[carried])

    caveats = [ESI_CAVEAT, CLIPPING_CAVEAT]
    if model.is_degenerate:
        caveats.append(
            "PC1 has no spread in this window, so the ESI is undefined rather "
            "than zero."
        )
    if dropped:
        caveats.append(
            f"{dropped} row(s) had a missing reading and were excluded from the "
            "projection; PCA has no notion of a missing value."
        )

    return ReductionResult(
        model=model,
        scores=scores,
        rows_used=len(complete),
        rows_dropped=dropped,
        flipped=flipped,
        caveats=tuple(caveats),
    )


__all__ = [
    "ANCHOR_COLUMN",
    "CLIPPING_CAVEAT",
    "ESI_CAVEAT",
    "ESI_MAX",
    "ESI_MIN",
    "MIN_ROWS",
    "ComponentLoadings",
    "ESIModel",
    "ReductionResult",
    "fit",
    "reduce_frame",
    "usable_rows",
]
