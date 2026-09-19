"""The HTML EDA report (task 4.7, Module 5).

The report's defining property is that it is **self-contained** -- one file,
no scripts, no network. That is what most of these tests check, because it is
the property that silently breaks first: one convenient CDN font and the report
no longer opens on the machine it was written for.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from services.datasets import MEASUREMENT_COLUMNS, describe_window
from services.eda import profile, report
from services.eda.cache import DatasetVersion
from tests.test_eda_profile import frame_from

VERSION = DatasetVersion(
    rows=120, max_id=900, latest_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
    flagged_rows=4,
)


def sample_frame():
    import numpy as np

    rng = np.random.default_rng(29)
    size = 400
    return frame_from(
        pm25=list(rng.lognormal(3.2, 0.5, size)),
        pm10=list(rng.lognormal(4.0, 0.5, size)),
        temp=list(rng.normal(24, 5, size)),
        humidity=list(rng.uniform(30, 80, size)),
        traffic_score=list(rng.uniform(10, 90, size)),
    )


@pytest.fixture
def rendered(tmp_path):
    from core.config import Settings

    frame = sample_frame()
    settings = Settings(_env_file=None, data_processed_dir=str(tmp_path))
    return report.render(
        frame=frame,
        statistics=profile.build(frame),
        window=describe_window(frame),
        version=VERSION,
        settings=settings,
    )


# --- Self-containment -------------------------------------------------------


def test_the_report_fetches_nothing(rendered) -> None:
    """No CDN, no font, no image. It has to open from a USB stick."""
    assert "http://" not in rendered.html
    assert "https://" not in rendered.html
    assert "@import" not in rendered.html
    assert "<img" not in rendered.html


def test_the_report_runs_no_script(rendered) -> None:
    assert "<script" not in rendered.html.lower()
    assert "onclick" not in rendered.html.lower()


def test_the_report_is_a_complete_document(rendered) -> None:
    assert rendered.html.startswith("<!DOCTYPE html>")
    assert "</html>" in rendered.html
    assert '<meta charset="utf-8">' in rendered.html


def test_the_report_carries_a_print_stylesheet(rendered) -> None:
    """PDF is produced by the browser, so the print rules are the feature."""
    assert "@media print" in rendered.html
    assert "page-break-inside" in rendered.html or "break-inside" in rendered.html


def test_dark_mode_is_defined_for_both_the_os_and_a_toggle(rendered) -> None:
    assert "prefers-color-scheme: dark" in rendered.html
    assert '[data-theme="dark"]' in rendered.html


# --- Content ----------------------------------------------------------------


def test_every_section_is_present(rendered) -> None:
    for section in ("Completeness", "Univariate", "Distributions", "Correlation"):
        assert section in rendered.html


def test_every_measurement_column_appears(rendered) -> None:
    """AC-3 reaches the report, not just the API."""
    for column in MEASUREMENT_COLUMNS:
        assert report.COLUMN_LABELS[column] in rendered.html


def test_units_are_shown_so_a_number_is_never_bare(rendered) -> None:
    assert "µg/m³" in rendered.html
    assert "°C" in rendered.html


def test_the_ethics_disclaimer_is_present(rendered) -> None:
    """ETH-1, verbatim. An exported report travels further than the dashboard
    does, so it carries the same sentence rather than a paraphrase."""
    assert report.ETHICS_DISCLAIMER in rendered.html
    assert "does not equal causation" in report.ETHICS_DISCLAIMER


def test_the_dataset_fingerprint_is_stamped_on_it(rendered) -> None:
    """Two reports of the same data should be identifiable as such."""
    assert VERSION.fingerprint in rendered.html


def test_the_report_is_written_to_disk(rendered, tmp_path) -> None:
    assert rendered.path == str(tmp_path / report.REPORT_FILENAME)
    assert (tmp_path / report.REPORT_FILENAME).is_file()


def test_persistence_can_be_skipped(tmp_path) -> None:
    from core.config import Settings

    frame = sample_frame()
    result = report.render(
        frame=frame,
        statistics=profile.build(frame),
        window=describe_window(frame),
        version=VERSION,
        settings=Settings(_env_file=None, data_processed_dir=str(tmp_path)),
        persist=False,
    )

    assert result.path is None
    assert not (tmp_path / report.REPORT_FILENAME).exists()


# --- Escaping ---------------------------------------------------------------


def test_values_are_html_escaped() -> None:
    """The station key and column labels reach the document as text."""
    escaped = report._esc('<script>alert("x")</script>')

    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped


# --- Charts -----------------------------------------------------------------


def test_the_charts_are_inline_svg(rendered) -> None:
    assert rendered.html.count("<svg") >= 5


def test_no_svg_coordinate_escapes_its_viewbox(rendered) -> None:
    """A cheap guard against label collisions and clipped marks."""
    problems = []
    for width, height, body in re.findall(
        r'<svg[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"(.*?)</svg>', rendered.html, re.S
    ):
        for attr, limit in (("x", float(width)), ("y", float(height))):
            for value in re.findall(rf'\b{attr}="(-?[\d.]+)"', body):
                if float(value) > limit + 1:
                    problems.append((attr, value, limit))

    assert problems == []


def test_the_correlation_heatmap_labels_every_cell() -> None:
    """A heatmap read by colour alone is unreadable in greyscale, in print, and
    for a colour-blind reader."""
    frame = sample_frame()
    svg = report.correlation_heatmap(profile.build(frame))

    cells = len(MEASUREMENT_COLUMNS) ** 2
    assert svg.count("cell-value") >= cells


def test_correlation_colour_diverges_around_zero() -> None:
    """Positive and negative must not share a hue, or the sign is invisible."""
    positive = report._diverging_colour(0.9)
    negative = report._diverging_colour(-0.9)
    neutral = report._diverging_colour(0.0)

    assert positive != negative
    assert neutral == "var(--diverging-mid)"


def test_stronger_correlations_get_deeper_colour() -> None:
    weak = report._diverging_colour(0.2)
    strong = report._diverging_colour(0.95)

    assert weak != strong
    assert report.DIVERGING_POSITIVE.index(strong) > report.DIVERGING_POSITIVE.index(weak)


def test_a_missing_correlation_renders_without_a_colour_claim() -> None:
    assert report._diverging_colour(None) == "var(--surface-2)"


def test_a_histogram_of_too_little_data_says_so() -> None:
    assert "Not enough" in report.histogram(frame_from(pm25=[1.0]), "pm25")


def test_an_empty_profile_does_not_break_the_charts() -> None:
    empty = frame_from(pm25=[1.0]).iloc[0:0]

    assert "No columns" in report.missingness_chart(
        profile.StatisticalProfile(
            rows=0,
            univariate=(),
            bivariate=profile.bivariate_profile(empty),
            distributions=(),
        )
    )


# --- Decomposition section --------------------------------------------------


@pytest.mark.skipif(
    not __import__("services.eda.decomposition", fromlist=["x"]).is_available(),
    reason="statsmodels could not be loaded in this environment",
)
def test_the_decomposition_panels_share_one_scale() -> None:
    """Two lines in one panel, each normalised to its own range, is a dual-axis
    chart in disguise: the curves would appear to track each other however far
    apart their values actually were.
    """
    from services.eda import decomposition
    from tests.test_eda_decomposition import seasonal_frame

    result = decomposition.decompose(seasonal_frame(), max_points=200)
    svg = report.decomposition_chart(result)

    # The observed/trend panel prints its shared range once, not once per line.
    low = min(min(result.observed), min(result.trend))
    high = max(max(result.observed), max(result.trend))
    assert f"{low:,.1f} to {high:,.1f}" in svg


def test_a_missing_decomposition_is_explained_not_omitted(tmp_path) -> None:
    from core.config import Settings

    frame = sample_frame()
    result = report.render(
        frame=frame,
        statistics=profile.build(frame),
        window=describe_window(frame),
        version=VERSION,
        decomposition_result=None,
        decomposition_note="series too short for period 24",
        settings=Settings(_env_file=None, data_processed_dir=str(tmp_path)),
        persist=False,
    )

    assert "Not included" in result.html
    assert "too short" in result.html


# --- PDF --------------------------------------------------------------------


def test_pdf_export_explains_the_browser_route_when_weasyprint_is_absent(
    tmp_path, monkeypatch
) -> None:
    """A Should-Have feature is not worth 100 MB of system libraries in the
    image, so the message has to make the alternative obvious."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("weasyprint"):
            raise ImportError("no weasyprint")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(RuntimeError, match="Print to PDF"):
        report.to_pdf("<html></html>", tmp_path / "out.pdf")
