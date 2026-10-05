"""CPCB PM2.5 bands (task 12.1, VIZ-6).

The bands exist twice — here for grouping and the report, in ``theme.ts`` for
the map and the charts — and the property worth defending is that the two never
disagree. A reading coloured "Moderate" on the dashboard and grouped as "Poor"
in the boxplot beside it would discredit both.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from services.eda import bands

THEME = Path(__file__).resolve().parents[2] / "frontend" / "src" / "charts" / "theme.ts"

# `{ limit: 30, label: 'Good', color: '#0ca30c' },` inside AQI_BANDS.
ENTRY = re.compile(
    r"\{\s*limit:\s*(?P<limit>[\d.]+|Infinity),\s*label:\s*'(?P<label>[^']+)',"
    r"\s*color:\s*'(?P<color>#[0-9a-fA-F]{6})'\s*\}"
)


def _frontend_bands() -> list[tuple[float, str, str]]:
    source = THEME.read_text(encoding="utf-8")
    block = source[source.index("export const AQI_BANDS") :]
    # Up to the array's closing bracket, not the `]` in `Band[]`.
    block = block[: block.index("\n]")]
    return [
        (
            math.inf if match["limit"] == "Infinity" else float(match["limit"]),
            match["label"],
            match["color"].lower(),
        )
        for match in ENTRY.finditer(block)
    ]


def test_the_backend_bands_match_the_frontend_theme_exactly() -> None:
    """The drift test design §12.1 chose over code generation: six edges that
    almost never change do not justify a build step, but they do justify a
    test that fails the moment the copies diverge."""
    frontend = _frontend_bands()

    assert len(frontend) == len(bands.PM25_BANDS), "theme.ts could not be parsed"
    assert frontend == [
        (band.limit, band.label, band.color.lower()) for band in bands.PM25_BANDS
    ]


def test_an_edge_value_belongs_to_the_lower_band() -> None:
    """``pm25 <= limit``, as ``band()`` in theme.ts does — exactly 30 is Good."""
    assert bands.band_for(30.0).label == "Good"
    assert bands.band_for(30.01).label == "Satisfactory"
    assert bands.band_for(250.0).label == "Very poor"
    assert bands.band_for(250.01).label == "Severe"
    assert bands.band_for(0.0).label == "Good"


def test_a_missing_reading_has_no_band_rather_than_a_pseudo_band() -> None:
    """A sensor gap is not a level of pollution; grouping it beside "Good"
    would put a count of outages on the same axis as a count of clean hours."""
    assert bands.band_for(None) is None
    assert bands.band_for(float("nan")) is None


def test_the_vectorised_labels_agree_with_the_scalar_lookup() -> None:
    values = pd.Series([0.0, 30.0, 30.5, 60.0, 89.9, 120.0, 121.0, 250.0, 900.0, np.nan])

    labels = bands.band_labels(values)

    expected = [None if (b := bands.band_for(v)) is None else b.label for v in values]
    assert [None if pd.isna(label) else label for label in labels] == expected


def test_band_labels_sort_by_severity_not_alphabetically() -> None:
    """Ordered so a groupby lists Good … Severe, the way the scale reads."""
    labels = bands.band_labels(pd.Series([300.0, 10.0, 100.0]))

    assert labels.cat.ordered
    assert list(labels.cat.categories) == list(bands.BAND_LABELS)
    assert list(labels.sort_values()) == ["Good", "Poor", "Severe"]
