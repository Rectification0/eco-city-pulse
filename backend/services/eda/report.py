"""Automated EDA report — BACSE301 Module 5 (task 4.7).

Produces one **self-contained HTML file**: inline CSS, inline SVG, no scripts,
no CDN, no fonts to fetch. That is a deliberate constraint rather than
minimalism for its own sake -- the report has to open from a USB stick, survive
being emailed, and render on a machine with no network, which is the same
offline promise DR-1 makes about the data itself.

**PDF.** The report carries a print stylesheet and page-break rules, so
"Print -> Save as PDF" in any browser produces a clean paginated document. A
server-side PDF renderer (WeasyPrint, wkhtmltopdf) would add ~100 MB of system
libraries to the image for a Should-Have feature, so the HTML is the artefact
and the browser is the converter. ``to_pdf`` will use WeasyPrint if it happens
to be installed, and says plainly what to do when it is not.

**Charts.** Hand-written SVG against the validated palette: one hue for
magnitude, a blue-red diverging ramp with a grey midpoint for correlation
(which is polarity, not magnitude), thin marks, recessive grid, and direct
labels rather than a colour-only legend. Both light and dark are stepped for
their own surface.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import Settings, get_settings
from services.datasets import MEASUREMENT_COLUMNS, DatasetWindow
from services.eda.cache import DatasetVersion
from services.eda.decomposition import DecompositionResult
from services.eda.profile import StatisticalProfile

REPORT_FILENAME = "eda_report.html"

# ETH-1, verbatim from specs §11. A constant rather than prose in the template
# for two reasons: line wrapping cannot break the sentence apart, and the UI
# (task 10.16) ships the identical wording rather than a paraphrase of it.
ETHICS_DISCLAIMER = (
    "Predictions rely on sensor placement which may exhibit "
    "geographic/socioeconomic bias. Correlation shown does not equal causation."
)

# Human labels and units, so a chart axis never reads "pm25".
COLUMN_LABELS: dict[str, str] = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "temp": "Temperature",
    "humidity": "Humidity",
    "traffic_score": "Traffic",
}
COLUMN_UNITS: dict[str, str] = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "temp": "°C",
    "humidity": "%",
    "traffic_score": "index",
}

HISTOGRAM_BINS = 28

# Diverging ramp for correlation: blue <-> red poles with a neutral grey
# midpoint. Correlation has a sign, so a one-hue sequential ramp would hide the
# difference between -0.8 and +0.8, and a rainbow would invent an ordering.
# Darkest first for the blue arm, lightest first for the red arm, so both are
# written in the direction the index walks them.
DIVERGING_NEGATIVE = ("#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb")
DIVERGING_POSITIVE = ("#f9d7d7", "#f0a9a9", "#e87675", "#e34948", "#c62f2e", "#8f1f1e")

# Above this magnitude the cell is dark enough that its label switches to white.
# 0.7 rather than 0.5: at 0.55 the red arm is still mid-tone, and white on it
# falls under 4.5:1.
STRONG_CORRELATION = 0.7


@dataclass(frozen=True, slots=True)
class ReportResult:
    html: str
    path: str | None
    generated_at: datetime
    sections: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "generated_at": self.generated_at.isoformat(),
            "sections": list(self.sections),
            "bytes": len(self.html.encode("utf-8")),
        }


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _label(column: str) -> str:
    return COLUMN_LABELS.get(column, column)


def _unit(column: str) -> str:
    return COLUMN_UNITS.get(column, "")


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:,.{digits}f}"


# --- Charts -----------------------------------------------------------------


def missingness_chart(statistics: StatisticalProfile) -> str:
    """Horizontal bars: how complete each column is.

    Magnitude, one series, so one hue and no legend -- the heading names it.
    Every bar is directly labelled, which is also what keeps the chart readable
    for a viewer who cannot separate the fill from the surface.
    """
    rows = list(statistics.univariate)
    if not rows:
        return "<p class='empty'>No columns to report.</p>"

    row_height, gap, label_width, track_width = 26, 8, 120, 420
    height = len(rows) * (row_height + gap)
    widest = max((r.missing_pct for r in rows), default=0.0) or 1.0
    scale = max(widest, 5.0)

    bars = []
    for index, stat in enumerate(rows):
        y = index * (row_height + gap)
        width = max(2.0, (stat.missing_pct / scale) * track_width)
        bars.append(
            f'<rect class="track" x="{label_width}" y="{y}" width="{track_width}" '
            f'height="{row_height}" rx="4"/>'
            f'<rect class="bar" x="{label_width}" y="{y}" width="{width:.1f}" '
            f'height="{row_height}" rx="4"/>'
            f'<text class="cat" x="{label_width - 10}" y="{y + row_height / 2}" '
            f'text-anchor="end" dominant-baseline="central">{_esc(_label(stat.column))}</text>'
            f'<text class="val" x="{label_width + width + 8:.1f}" y="{y + row_height / 2}" '
            f'dominant-baseline="central">{stat.missing_pct:.2f}% '
            f'({stat.missing:,})</text>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {label_width + track_width + 110} {height}" '
        f'role="img" aria-label="Missing values by column, as a percentage of rows">'
        f'{"".join(bars)}</svg>'
    )


def histogram(frame: pd.DataFrame, column: str) -> str:
    """Distribution of one column. One series, one hue, 2px gaps between bars."""
    values = frame[column].dropna() if column in frame.columns else pd.Series(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return "<p class='empty'>Not enough observations to plot.</p>"

    counts, edges = np.histogram(values.to_numpy(), bins=HISTOGRAM_BINS)
    peak = counts.max() or 1
    width, height, pad = 380, 130, 22
    plot_height = height - pad
    slot = width / len(counts)

    bars = []
    for index, count in enumerate(counts):
        bar_height = (count / peak) * plot_height
        x = index * slot
        bars.append(
            f'<rect class="bar" x="{x + 1:.1f}" y="{plot_height - bar_height:.1f}" '
            f'width="{max(1.0, slot - 2):.1f}" height="{bar_height:.1f}" rx="2"/>'
        )

    axis = (
        f'<line class="axis" x1="0" y1="{plot_height}" x2="{width}" y2="{plot_height}"/>'
        f'<text class="tick" x="0" y="{height - 4}">{_fmt(float(edges[0]), 1)}</text>'
        f'<text class="tick" x="{width}" y="{height - 4}" text-anchor="end">'
        f'{_fmt(float(edges[-1]), 1)}</text>'
    )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Distribution of {_esc(_label(column))}">'
        f'{"".join(bars)}{axis}</svg>'
    )


def _diverging_colour(value: float | None) -> str:
    """Blue for negative, red for positive, grey at zero."""
    if value is None or not math.isfinite(value):
        return "var(--surface-2)"
    magnitude = min(abs(value), 1.0)
    if magnitude < 0.05:
        return "var(--diverging-mid)"
    ramp = DIVERGING_POSITIVE if value > 0 else DIVERGING_NEGATIVE
    index = min(len(ramp) - 1, int(magnitude * len(ramp)))
    return ramp[index] if value > 0 else ramp[len(ramp) - 1 - index]


def correlation_heatmap(statistics: StatisticalProfile, method: str = "pearson") -> str:
    """Correlation matrix as a labelled grid.

    Every cell carries its number. A heatmap read by colour alone is the single
    most misread artefact in an EDA dashboard, and the labels mean the chart
    still works in greyscale, in print, and for a colour-blind reader.
    """
    matrix = getattr(statistics.bivariate, method)
    columns = list(statistics.bivariate.columns)
    if not columns:
        return "<p class='empty'>No numeric columns to correlate.</p>"

    # Separate top and left gutters: the left one has to hold "Temperature"
    # end-aligned, the top only needs the cap height of one line.
    cell, gutter_left, gutter_top = 74, 96, 22
    size = len(columns)
    width = gutter_left + size * cell
    height = gutter_top + size * cell

    parts = []
    for index, column in enumerate(columns):
        x = gutter_left + index * cell + cell / 2
        parts.append(
            f'<text class="cat" x="{x}" y="{gutter_top - 8}" text-anchor="middle">'
            f"{_esc(_label(column))}</text>"
        )
        y = gutter_top + index * cell + cell / 2
        parts.append(
            f'<text class="cat" x="{gutter_left - 10}" y="{y}" text-anchor="end" '
            f'dominant-baseline="central">{_esc(_label(column))}</text>'
        )

    for row, a in enumerate(columns):
        for col, b in enumerate(columns):
            value = matrix.get(a, {}).get(b)
            x = gutter_left + col * cell
            y = gutter_top + row * cell
            fill = _diverging_colour(value)
            strong = value is not None and abs(value) >= STRONG_CORRELATION
            text_class = "cell-value strong" if strong else "cell-value"
            parts.append(
                f'<rect class="cell" x="{x + 1}" y="{y + 1}" width="{cell - 2}" '
                f'height="{cell - 2}" rx="4" fill="{fill}"/>'
                f'<text class="{text_class}" x="{x + cell / 2}" y="{y + cell / 2}" '
                f'text-anchor="middle" dominant-baseline="central">'
                f"{'—' if value is None else f'{value:+.2f}'}</text>"
            )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(method.title())} correlation matrix">'
        f'{"".join(parts)}</svg>'
    )


def _line_path(
    values: tuple[float, ...],
    width: float,
    height: float,
    *,
    low: float,
    high: float,
) -> str:
    """One polyline, scaled against bounds the caller supplies.

    The bounds are a parameter rather than computed per series on purpose. Two
    lines in one panel, each normalised to its own range, is a dual-axis chart
    wearing a disguise: the curves would appear to track each other no matter
    how far apart their actual values were. Everything sharing a panel shares a
    scale.
    """
    if len([v for v in values if math.isfinite(v)]) < 2:
        return ""
    span = (high - low) or 1.0
    step = width / max(1, len(values) - 1)
    points = [
        f"{index * step:.2f},{height - ((value - low) / span) * height:.2f}"
        for index, value in enumerate(values)
        if math.isfinite(value)
    ]
    return "M" + " L".join(points)


def _panel_bounds(series: list[tuple[str, tuple[float, ...]]]) -> tuple[float, float]:
    finite = [v for _, values in series for v in values if math.isfinite(v)]
    if not finite:
        return (0.0, 1.0)
    return (min(finite), max(finite))


def decomposition_chart(result: DecompositionResult) -> str:
    """Four stacked panels: observed+trend, seasonal, residual.

    Small multiples rather than one crowded axis. Observed and trend share a
    panel because that overlay is the comparison worth making; both are
    directly labelled, so identity never rests on colour alone.
    """
    width, panel = 720, 120
    panels = [
        ("Observed and trend", [("observed", result.observed), ("trend", result.trend)]),
        ("Seasonal", [("seasonal", result.seasonal)]),
        ("Residual", [("residual", result.residual)]),
    ]

    blocks = []
    offset = 0
    for title, series in panels:
        low, high = _panel_bounds(series)
        rows = []
        for name, values in series:
            path = _line_path(values, width, panel - 40, low=low, high=high)
            if path:
                rows.append(f'<path class="line {name}" d="{path}"/>')
        # A legend for two series, none for one: with a single line the panel
        # title already names it, and a legend box would be noise.
        legend = (
            " ".join(
                f'<tspan class="key {name}">■</tspan> '
                f'<tspan class="cat">{name}</tspan>'
                for name, _ in series
            )
            if len(series) > 1
            else ""
        )
        scale = f"{_fmt(low, 1)} to {_fmt(high, 1)}"
        blocks.append(
            f'<g transform="translate(0 {offset})">'
            f'<text class="panel-title" x="0" y="12">{_esc(title)}</text>'
            f'<text class="tick" x="0" y="26">{_esc(scale)}</text>'
            f'<text class="legend" x="{width}" y="12" text-anchor="end">{legend}</text>'
            f'<g transform="translate(0 30)">{"".join(rows)}</g>'
            f"</g>"
        )
        offset += panel

    return (
        f'<svg class="chart" viewBox="0 0 {width} {offset}" role="img" '
        f'aria-label="STL decomposition of {_esc(_label(result.column))}">'
        f'{"".join(blocks)}</svg>'
    )


# --- Document ---------------------------------------------------------------

STYLES = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --diverging-mid: #f0efec;
  --cell-ink: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --surface-2: #232321;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --diverging-mid: #383835;
    --cell-ink: #ffffff;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface-1: #1a1a19;
  --surface-2: #232321;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --diverging-mid: #383835;
  --cell-ink: #ffffff;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px 64px;
  background: var(--page);
  color: var(--text-primary);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 14px; margin: 0 0 8px; font-weight: 600; color: var(--text-secondary); }
p { margin: 0 0 12px; color: var(--text-secondary); }
.lede { color: var(--text-secondary); margin-bottom: 24px; }
section {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
  margin-bottom: 18px;
}
.section-note { font-size: 13px; color: var(--text-muted); margin: 0 0 16px; }
.kpis { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }
.kpi {
  flex: 1 1 150px;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}
.kpi .value { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; }
.kpi .label { font-size: 12px; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: 0.04em; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th { color: var(--text-muted); font-weight: 600; text-transform: uppercase;
  font-size: 11px; letter-spacing: 0.04em; }
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
.chart { width: 100%; height: auto; overflow: visible; }
.chart .bar { fill: var(--series-1); }
.chart .track { fill: var(--surface-2); }
.chart .cat { fill: var(--text-secondary); font-size: 12px; }
.chart .val { fill: var(--text-muted); font-size: 11px; }
.chart .tick { fill: var(--text-muted); font-size: 10px; }
.chart .axis { stroke: var(--axis); stroke-width: 1; }
.chart .cell { stroke: var(--surface-1); stroke-width: 2; }
.chart .cell-value { fill: var(--cell-ink); font-size: 12px; }
.chart .cell-value.strong { fill: #ffffff; font-weight: 600; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .line.observed, .chart .line.seasonal, .chart .line.residual { stroke: var(--series-1); }
.chart .line.trend { stroke: var(--series-2); }
.chart .panel-title { fill: var(--text-secondary); font-size: 12px; font-weight: 600; }
.chart .legend { font-size: 11px; }
.chart .key.observed, .chart .key.seasonal, .chart .key.residual { fill: var(--series-1); }
.chart .key.trend { fill: var(--series-2); }
.badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--text-secondary); background: var(--surface-2); }
.caveat {
  border-left: 3px solid var(--series-2);
  background: var(--surface-2);
  padding: 10px 14px;
  border-radius: 0 8px 8px 0;
  font-size: 13px;
  color: var(--text-secondary);
  margin: 12px 0 0;
}
.empty { color: var(--text-muted); font-size: 13px; font-style: italic; }
footer { max-width: 980px; margin: 24px auto 0; color: var(--text-muted); font-size: 12px; }

@media print {
  body { background: #ffffff; padding: 0; }
  section { break-inside: avoid; page-break-inside: avoid; border-color: #d8d8d2; }
  .grid-2 { display: block; }
  .grid-2 > * { margin-bottom: 16px; }
  h2 { break-after: avoid; }
}
"""


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="value">{_esc(value)}</div><div class="label">{_esc(label)}</div></div>'


def _univariate_table(statistics: StatisticalProfile) -> str:
    head = (
        "<tr><th>Column</th><th>n</th><th>Missing</th><th>Mean</th><th>Median</th>"
        "<th>Std</th><th>IQR</th><th>Min</th><th>Max</th><th>Skew</th><th>Kurtosis</th></tr>"
    )
    rows = []
    for stat in statistics.univariate:
        rows.append(
            "<tr>"
            f"<td>{_esc(_label(stat.column))} <span class='badge'>{_esc(_unit(stat.column))}</span></td>"
            f"<td>{stat.count:,}</td>"
            f"<td>{stat.missing_pct:.2f}%</td>"
            f"<td>{_fmt(stat.mean)}</td><td>{_fmt(stat.median)}</td>"
            f"<td>{_fmt(stat.std)}</td><td>{_fmt(stat.iqr)}</td>"
            f"<td>{_fmt(stat.minimum)}</td><td>{_fmt(stat.maximum)}</td>"
            f"<td>{_fmt(stat.skewness)}</td><td>{_fmt(stat.kurtosis)}</td>"
            "</tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _distribution_list(statistics: StatisticalProfile) -> str:
    items = []
    for assessment in statistics.distributions:
        verdict = (
            "log1p recommended"
            if assessment.recommend_log_transform
            else "no transform"
        )
        items.append(
            f"<tr><td>{_esc(_label(assessment.column))}</td>"
            f"<td>{_fmt(assessment.skewness)}</td>"
            f"<td>{_fmt(assessment.log_skewness)}</td>"
            f"<td><span class='badge'>{_esc(verdict)}</span></td>"
            f"<td style='text-align:left'>{_esc(assessment.rationale)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Column</th><th>Skew</th><th>Skew after log1p</th>"
        f"<th>Verdict</th><th style='text-align:left'>Why</th></tr></thead>"
        f"<tbody>{''.join(items)}</tbody></table>"
    )


def _correlation_table(statistics: StatisticalProfile) -> str:
    rows = []
    for pair in sorted(
        statistics.bivariate.pairs, key=lambda p: abs(p.pearson or 0), reverse=True
    ):
        rows.append(
            f"<tr><td>{_esc(_label(pair.a))} &middot; {_esc(_label(pair.b))}</td>"
            f"<td>{_fmt(pair.pearson, 3)}</td><td>{_fmt(pair.spearman, 3)}</td>"
            f"<td>{pair.sample_size:,}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Pair</th><th>Pearson</th><th>Spearman</th>"
        f"<th>n</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render(
    *,
    frame: pd.DataFrame,
    statistics: StatisticalProfile,
    window: DatasetWindow,
    version: DatasetVersion,
    decomposition_result: DecompositionResult | None = None,
    decomposition_note: str = "",
    settings: Settings | None = None,
    persist: bool = True,
) -> ReportResult:
    """Build the report and, by default, write it to ``data/processed``."""
    settings = settings or get_settings()
    generated_at = datetime.now(timezone.utc)
    sections = ["overview", "missingness", "univariate", "distributions", "correlation"]

    covered = "—"
    if window.start and window.end:
        covered = f"{window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}"

    histograms = "".join(
        f"<div><h3>{_esc(_label(column))} <span class='badge'>{_esc(_unit(column))}</span>"
        f"</h3>{histogram(frame, column)}</div>"
        for column in MEASUREMENT_COLUMNS
        if column in frame.columns
    )

    stl_section = ""
    if decomposition_result is not None:
        sections.append("decomposition")
        strength = decomposition_result.strength
        note = (
            f"<p class='caveat'>{_esc(decomposition_result.caveat)}</p>"
            if decomposition_result.caveat
            else ""
        )
        stl_section = f"""
  <section>
    <h2>Time-series decomposition</h2>
    <p class="section-note">
      STL on {_esc(_label(decomposition_result.column))} at station
      {_esc(decomposition_result.station)}, period {decomposition_result.period} hours.
      Trend strength {strength.trend:.2f}, seasonal strength {strength.seasonal:.2f}
      (both 0–1: the share of variation the component explains beyond the noise).
      Showing the last {decomposition_result.points_returned:,} of
      {decomposition_result.points_analysed:,} analysed hours.
    </p>
    {decomposition_chart(decomposition_result)}
    {note}
  </section>"""
    elif decomposition_note:
        stl_section = f"""
  <section>
    <h2>Time-series decomposition</h2>
    <p class="caveat">Not included: {_esc(decomposition_note)}</p>
  </section>"""

    caveats = "".join(f"<p class='caveat'>{_esc(c)}</p>" for c in statistics.caveats)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eco-City Pulse — EDA Report</title>
<meta name="description" content="Automated exploratory data analysis of urban environmental observations.">
<style>{STYLES}</style>
</head>
<body>
<main>
  <h1>Exploratory Data Analysis</h1>
  <p class="lede">
    Eco-City Pulse &middot; generated {generated_at:%Y-%m-%d %H:%M} UTC &middot;
    dataset <code>{_esc(version.fingerprint)}</code>
  </p>

  <div class="kpis">
    {_kpi(f"{window.rows:,}", "observations")}
    {_kpi(str(window.stations), "stations")}
    {_kpi(covered, "period covered")}
    {_kpi(f"{version.flagged_rows:,}", "flagged anomalies")}
  </div>

  <section>
    <h2>Completeness</h2>
    <p class="section-note">
      Share of rows with no reading, per column. Gaps are the input to the
      imputation stage, not an error in themselves.
    </p>
    {missingness_chart(statistics)}
  </section>

  <section>
    <h2>Univariate statistics</h2>
    <p class="section-note">
      Centre, spread and shape per column. Kurtosis is excess kurtosis, so 0 is
      the normal distribution.
    </p>
    {_univariate_table(statistics)}
  </section>

  <section>
    <h2>Distributions</h2>
    <p class="section-note">
      Each column's shape, and whether a log transform earns its cost in
      interpretability.
    </p>
    <div class="grid-2">{histograms}</div>
    <div style="margin-top:18px">{_distribution_list(statistics)}</div>
  </section>

  <section>
    <h2>Correlation</h2>
    <p class="section-note">
      Pearson (linear) below; Spearman (monotone, rank-based) in the table. They
      disagree where a relationship is monotone but curved.
    </p>
    {correlation_heatmap(statistics, "pearson")}
    <div style="margin-top:18px">{_correlation_table(statistics)}</div>
    {caveats}
  </section>
{stl_section}

  <section>
    <h2>How to read this report</h2>
    <p>
      Every figure here describes <strong>association and distribution</strong>,
      not cause. Sensor placement is not random, so the readings describe the
      places instruments happen to be, which may differ systematically from the
      places people are.
    </p>
    <p class="caveat">{_esc(ETHICS_DISCLAIMER)}</p>
  </section>
</main>
<footer>
  Self-contained: no scripts, no external assets. Print to PDF for a paginated copy.
</footer>
</body>
</html>
"""

    path: str | None = None
    if persist:
        directory = settings.data_processed_path
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / REPORT_FILENAME
        target.write_text(document, encoding="utf-8")
        path = str(target)

    return ReportResult(
        html=document,
        path=path,
        generated_at=generated_at,
        sections=tuple(sections),
    )


def to_pdf(document: str, destination: Path) -> Path:
    """Render the HTML to PDF, if a renderer is installed.

    Not a hard dependency: WeasyPrint pulls in ~100 MB of Cairo/Pango system
    libraries, which is poor value in the image for a Should-Have feature when
    every browser already converts the print stylesheet correctly.
    """
    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDF export needs WeasyPrint (`pip install weasyprint`, plus its "
            "system libraries). Without it, open the HTML report and use the "
            "browser's Print to PDF -- the report ships a print stylesheet for "
            "exactly that."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=document).write_pdf(str(destination))
    return destination


__all__ = [
    "COLUMN_LABELS",
    "ETHICS_DISCLAIMER",
    "STRONG_CORRELATION",
    "COLUMN_UNITS",
    "REPORT_FILENAME",
    "ReportResult",
    "correlation_heatmap",
    "decomposition_chart",
    "histogram",
    "missingness_chart",
    "render",
    "to_pdf",
]
