"""t-SNE projection — EDA Studio only (task 6.5, specs §6.2, design §9).

t-SNE turns the five-dimensional reading space into two coordinates that can be
scattered on a screen, so clusters -- winter nights, monsoon afternoons, a
station that behaves unlike its neighbours -- become visible.

**It is strictly a visualisation.** design §9: "t-SNE is strictly EDA-only. It
has no stable out-of-sample transform, so it never feeds a model or the ESI."
That is not a policy choice, it is what the algorithm is: ``TSNE`` exposes
``fit_transform`` and no ``transform``, because the embedding is optimised for
the points it was given and a new point has no defined position in it. Feeding
these coordinates to a model would mean re-embedding the whole dataset for every
prediction and getting different axes each time. So this module returns
coordinates and a caveat, and nothing persists them.

**What the axes are not.** t-SNE distances are local. Cluster *membership* is
meaningful; the distance between two clusters, their sizes, and the orientation
of the plot are artefacts of the optimisation. The caveat says so, and travels
in the payload rather than being left to whoever writes the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from core.exceptions import InsufficientDataError
from services.datasets import MEASUREMENT_COLUMNS

# t-SNE is roughly quadratic in the number of points and produces a scatter no
# one can read past a few thousand marks. The cap keeps an EDA Studio request
# to about a second rather than a minute.
DEFAULT_MAX_POINTS = 1500

# sklearn's rule of thumb: perplexity is a guess at the local neighbourhood
# size, and it must stay below the sample size.
DEFAULT_PERPLEXITY = 30.0

RANDOM_STATE = 42

# Below this the embedding is fitting noise.
MIN_ROWS = 50

TSNE_CAVEAT = (
    "t-SNE preserves local neighbourhoods only. Which points sit together is "
    "meaningful; the distance between clusters, their relative sizes and the "
    "orientation of the axes are artefacts of the optimisation and carry no "
    "interpretation. The axes have no units and no out-of-sample transform, so "
    "this projection is for inspection only and never feeds a model (design §9)."
)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """A 2D embedding, with enough context to colour and read the scatter."""

    columns: tuple[str, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    timestamps: tuple[datetime, ...]
    stations: tuple[str, ...]
    hour_of_day: tuple[int, ...]
    perplexity: float
    points: int
    rows_available: int
    subsampled: bool
    caveat: str = TSNE_CAVEAT

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "x": [round(value, 4) for value in self.x],
            "y": [round(value, 4) for value in self.y],
            "timestamps": [value.isoformat() for value in self.timestamps],
            "stations": list(self.stations),
            "hour_of_day": list(self.hour_of_day),
            "perplexity": self.perplexity,
            "points": self.points,
            "rows_available": self.rows_available,
            "subsampled": self.subsampled,
            "caveat": self.caveat,
        }


def subsample(frame: pd.DataFrame, max_points: int) -> tuple[pd.DataFrame, bool]:
    """Thin the frame to at most ``max_points``, evenly across the window.

    Evenly spaced rather than randomly sampled: the frame is time-sorted, so a
    stride keeps the whole period represented and, being deterministic, gives
    the same picture on every call. A random sample would redraw the scatter on
    each refresh for no gain.
    """
    if len(frame) <= max_points:
        return frame, False

    positions = np.linspace(0, len(frame) - 1, max_points).astype(int)
    return frame.iloc[np.unique(positions)], True


def project(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    perplexity: float = DEFAULT_PERPLEXITY,
    max_points: int = DEFAULT_MAX_POINTS,
    random_state: int = RANDOM_STATE,
) -> ProjectionResult:
    """Embed the readings in two dimensions (task 6.5, Module 4).

    Standardized first, for the same reason PCA is: t-SNE works on distances,
    and an unscaled PM10 axis would dominate every neighbourhood.
    """
    complete = frame[frame[list(columns)].notna().all(axis=1)] if len(frame) else frame

    if len(complete) < MIN_ROWS:
        raise InsufficientDataError(
            f"t-SNE needs at least {MIN_ROWS} complete observations; "
            f"{len(complete)} available.",
            details={"required": MIN_ROWS, "available": len(complete)},
        )

    sampled, was_subsampled = subsample(complete, max_points)

    # Perplexity must stay below the sample size or sklearn refuses; clamping
    # here turns a 500 from deep inside sklearn into a projection that works.
    effective_perplexity = float(min(perplexity, max(5.0, (len(sampled) - 1) / 3.0)))

    matrix = StandardScaler().fit_transform(
        sampled[list(columns)].to_numpy(dtype="float64")
    )
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        # PCA initialisation plus a fixed seed makes the embedding reproducible,
        # which a random init would not be even with the seed set.
    ).fit_transform(matrix)

    timestamps = pd.to_datetime(sampled["timestamp"], utc=True)

    return ProjectionResult(
        columns=tuple(columns),
        x=tuple(float(value) for value in embedding[:, 0]),
        y=tuple(float(value) for value in embedding[:, 1]),
        # Element-wise rather than .dt.to_pydatetime(), whose return type is
        # changing in a future pandas and which warns about it in this one.
        timestamps=tuple(value.to_pydatetime() for value in timestamps),
        stations=tuple(sampled.get("station", pd.Series([""] * len(sampled)))),
        hour_of_day=tuple(int(hour) for hour in timestamps.dt.hour),
        perplexity=effective_perplexity,
        points=len(sampled),
        rows_available=len(complete),
        subsampled=was_subsampled,
    )


__all__ = [
    "DEFAULT_MAX_POINTS",
    "DEFAULT_PERPLEXITY",
    "MIN_ROWS",
    "RANDOM_STATE",
    "TSNE_CAVEAT",
    "ProjectionResult",
    "project",
    "subsample",
]
