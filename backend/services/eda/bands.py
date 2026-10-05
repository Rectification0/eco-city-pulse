"""CPCB PM2.5 bands — the backend's single definition (task 12.1, VIZ-6).

Until Phase 12 the bands lived only in ``frontend/src/charts/theme.ts``: the
browser coloured a map pin by them and nothing on the server needed to know
they existed. The Visual EDA charts change that. A grouped boxplot *by band*,
a pair plot *coloured by band* and the HTML report all need the edges on the
server, and a second hand-typed copy would drift the first time someone moved
a breakpoint in one place and not the other.

**Why a drift test rather than code generation.** Generating ``theme.ts`` from
this file would put a build step between a Python constant and the frontend
bundle for six numbers that have not changed since CPCB published them. A test
that parses ``theme.ts`` and fails on any disagreement gives the same guarantee
— the two copies cannot ship different — for one regex
(``tests/test_eda_bands.py``).

**Edges are inclusive on the upper side**, matching ``band()`` in ``theme.ts``
(``pm25 <= entry.limit``): exactly 30 µg/m³ is *Good*, 30.01 is
*Satisfactory*. Off-by-one at an edge is the kind of disagreement nobody sees
until two charts colour the same reading differently.

**Descriptive, not a health judgement** (ETH-1). A band names a concentration
range; it makes no claim about what that concentration does to anyone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Band:
    """One PM2.5 band: readings ``<= limit`` (and above the previous limit)."""

    limit: float
    label: str
    color: str

    def as_dict(self) -> dict[str, object]:
        return {
            "limit": None if math.isinf(self.limit) else self.limit,
            "label": self.label,
            "color": self.color,
        }


# Order matters: lookup walks it and takes the first band whose limit is not
# exceeded, and every chart lists the bands in this order.
PM25_BANDS: tuple[Band, ...] = (
    Band(30.0, "Good", "#0ca30c"),
    Band(60.0, "Satisfactory", "#8bc34a"),
    Band(90.0, "Moderate", "#fab219"),
    Band(120.0, "Poor", "#ec835a"),
    Band(250.0, "Very poor", "#d03b3b"),
    Band(math.inf, "Severe", "#9333ea"),
)

BAND_LABELS: tuple[str, ...] = tuple(band.label for band in PM25_BANDS)


def band_for(pm25: float | None) -> Band | None:
    """The band a single reading falls in, or None for no reading.

    None rather than a "No reading" band: a missing PM2.5 is not a level of
    pollution, and grouping it alongside the real bands would put a count of
    sensor gaps on the same axis as a count of clean hours.
    """
    if pm25 is None or not math.isfinite(pm25):
        return None
    for band in PM25_BANDS:
        if pm25 <= band.limit:
            return band
    return PM25_BANDS[-1]  # pragma: no cover - the last limit is +inf


def band_labels(values: pd.Series) -> pd.Series:
    """Vectorised ``band_for(...).label``, as an ordered categorical.

    Ordered so a ``groupby`` lists Good before Severe rather than
    alphabetically, and categorical so an empty band still has a defined
    position. Missing readings stay missing.
    """
    limits = np.array([band.limit for band in PM25_BANDS[:-1]])
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    # searchsorted(side="left") on the upper limits gives the index of the first
    # limit >= value, which is exactly the inclusive-upper-edge rule above.
    positions = np.searchsorted(limits, numeric, side="left")
    labels = np.array(BAND_LABELS, dtype=object)[np.clip(positions, 0, len(BAND_LABELS) - 1)]
    labels = np.where(np.isfinite(numeric), labels, None)
    return pd.Series(
        pd.Categorical(labels, categories=BAND_LABELS, ordered=True),
        index=values.index,
    )


def color_for(label: str) -> str:
    """The band colour for a label. Raises on an unknown label."""
    for band in PM25_BANDS:
        if band.label == label:
            return band.color
    raise KeyError(label)


__all__ = [
    "BAND_LABELS",
    "PM25_BANDS",
    "Band",
    "band_for",
    "band_labels",
    "color_for",
]
