"""PCA and the ESI — FEAT-04 (tasks 6.1-6.3, 6.7; AC-6, design §9).

Two properties carry the phase, and both are asserted directly rather than
inferred from a plausible-looking number.

**AC-6: the score is bounded.** 0-100, including for a row more extreme than
anything the model was fitted on.

**The axis has a fixed direction.** A principal component is defined only up to
sign, so the same data can produce PC1 or its exact negative depending on the
LAPACK build. An ESI that silently inverted between runs would be worse than no
ESI, so the orientation is pinned to PM2.5 and tested from both sides.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from services.datasets import MEASUREMENT_COLUMNS, prepare_frame
from services.eda import reduction, service
from services.eda.reduction import ESI_MAX, ESI_MIN, ESIModel

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_frame(rows: int = 400, *, seed: int = 5, stations: int = 1) -> pd.DataFrame:
    """Correlated readings with a real dominant axis.

    PM2.5 rises with traffic and falls with temperature -- the winter-inversion
    structure the demo generator also carries -- so PC1 is a meaningful axis
    rather than an artefact of noise.
    """
    rng = np.random.default_rng(seed)
    records = []
    for station in range(stations):
        for hour in range(rows):
            traffic = 40 + 30 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 5)
            pm25 = 60 + 0.8 * traffic + rng.normal(0, 8)
            records.append(
                {
                    "id": station * rows + hour + 1,
                    "source_id": 1,
                    "timestamp": START + timedelta(hours=hour),
                    "lat": 28.6 + station * 0.1,
                    "lon": 77.2,
                    "pm25": float(pm25),
                    "pm10": float(pm25 * 1.8 + rng.normal(0, 5)),
                    "temp": float(30 - 0.1 * pm25 + rng.normal(0, 2)),
                    "humidity": float(rng.uniform(30, 80)),
                    "traffic_score": float(traffic),
                    "is_anomaly": False,
                }
            )
    return prepare_frame(pd.DataFrame(records))


# --- Standardization (task 6.1) ---------------------------------------------


def test_variables_are_standardized_before_pca() -> None:
    """PCA maximises variance, and variance is scale-dependent: left in raw
    units PM10 would dominate humidity for no reason but its units."""
    frame = build_frame()

    model, _ = reduction.fit(frame)

    for column, mean in zip(model.columns, model.means, strict=True):
        assert mean == pytest.approx(frame[column].mean())
    assert all(scale > 0 for scale in model.scales)


def test_a_constant_column_does_not_divide_by_zero() -> None:
    frame = build_frame(rows=200)
    frame["humidity"] = 55.0

    model, _ = reduction.fit(frame)

    assert all(np.isfinite(scale) and scale > 0 for scale in model.scales)
    assert np.isfinite(model.project(frame)).all()


# --- PCA (task 6.2) ----------------------------------------------------------


def test_explained_variance_and_loadings_are_reported() -> None:
    frame = build_frame()

    result = reduction.reduce_frame(frame)
    first = result.model.loadings[0]

    assert 0.0 < first.explained_variance_ratio <= 1.0
    assert set(first.loadings) == set(MEASUREMENT_COLUMNS)
    assert first.cumulative_variance_ratio == pytest.approx(
        first.explained_variance_ratio
    )


def test_cumulative_variance_reaches_one_over_all_components() -> None:
    frame = build_frame()

    result = reduction.reduce_frame(frame)

    assert result.model.loadings[-1].cumulative_variance_ratio == pytest.approx(1.0)


def test_pc1_is_the_pollution_axis_on_pollution_shaped_data() -> None:
    """The check that the component means something: on data where PM2.5 rises
    with traffic and falls with temperature, PC1 should say exactly that."""
    frame = build_frame()

    loadings = reduction.reduce_frame(frame).model.loadings[0].loadings

    assert loadings["pm25"] > 0
    assert loadings["pm10"] > 0
    assert loadings["traffic_score"] > 0
    assert loadings["temp"] < 0


def test_the_fit_is_reproducible() -> None:
    """Task 6.7. The full SVD has no random initialisation, so reproducibility
    does not even depend on the seed -- which is a stronger guarantee than a
    fixed seed, and worth pinning."""
    frame = build_frame()

    first, _ = reduction.fit(frame)
    second, _ = reduction.fit(frame)

    assert first.components == second.components
    assert first.explained_variance_ratio == second.explained_variance_ratio
    assert first.pc1_min == second.pc1_min


def test_a_window_too_small_for_a_covariance_matrix_is_refused() -> None:
    frame = build_frame(rows=10)

    with pytest.raises(InsufficientDataError, match="at least"):
        reduction.fit(frame)


# --- Orientation ------------------------------------------------------------


def test_pc1_is_oriented_to_increase_with_pm25() -> None:
    frame = build_frame()

    result = reduction.reduce_frame(frame)
    correlation = np.corrcoef(result.scores["pc1"], frame.loc[result.scores.index, "pm25"])[0, 1]

    assert correlation > 0
    assert result.model.anchor == "pm25"


def test_an_axis_pointing_away_from_pm25_is_turned_around_and_says_so() -> None:
    """The case the orientation step exists for.

    Here four of the five columns move together and PM2.5 moves against them,
    so the raw PC1 comes back with a negative PM2.5 loading -- a perfectly good
    axis pointing the wrong way for an index called *stress*. It is flipped,
    the flip is reported, and the stored loadings describe the axis the scores
    were actually measured on.
    """
    rng = np.random.default_rng(3)
    hours = np.arange(300)
    driver = 40 + 30 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 3, 300)
    frame = prepare_frame(
        pd.DataFrame(
            {
                "id": hours + 1,
                "source_id": 1,
                "timestamp": [START + timedelta(hours=int(h)) for h in hours],
                "lat": 28.6,
                "lon": 77.2,
                "pm25": -driver + rng.normal(0, 1, 300),
                "pm10": 2 * driver + rng.normal(0, 1, 300),
                "temp": driver + rng.normal(0, 1, 300),
                "humidity": driver + rng.normal(0, 1, 300),
                "traffic_score": driver + rng.normal(0, 1, 300),
                "is_anomaly": False,
            }
        )
    )

    result = reduction.reduce_frame(frame)
    loadings = result.model.loadings[0].loadings

    assert result.flipped
    assert loadings["pm25"] > 0
    # The scores were flipped with the component, so the two still agree.
    correlation = np.corrcoef(
        result.scores["pc1"], frame.loc[result.scores.index, "pm25"]
    )[0, 1]
    assert correlation > 0


# --- The ESI (task 6.3, AC-6) ------------------------------------------------


def test_the_esi_is_bounded_between_zero_and_one_hundred() -> None:
    frame = build_frame()

    esi = reduction.reduce_frame(frame).scores["esi"]

    assert esi.min() >= ESI_MIN
    assert esi.max() <= ESI_MAX


def test_the_fit_window_spans_the_full_scale() -> None:
    """Min-max against the fit window, per design §9: the dirtiest hour in the
    window is 100 and the cleanest is 0."""
    frame = build_frame()

    esi = reduction.reduce_frame(frame).scores["esi"]

    assert esi.min() == pytest.approx(ESI_MIN, abs=1e-6)
    assert esi.max() == pytest.approx(ESI_MAX, abs=1e-6)


def test_a_reading_beyond_the_fit_window_saturates_rather_than_escaping() -> None:
    """AC-6 is a hard bound, so an unprecedented hour reads 100 -- and the
    caveat in the payload says that is what happened."""
    frame = build_frame()
    model, _ = reduction.fit(frame)

    extreme = frame.iloc[[0]].copy()
    extreme["pm25"] = 5000.0
    extreme["pm10"] = 9000.0
    extreme["traffic_score"] = 500.0

    assert model.score(extreme).iloc[0] == ESI_MAX
    assert any("clipped" in caveat for caveat in reduction.reduce_frame(frame).caveats)


def test_a_dirtier_hour_scores_higher_than_a_cleaner_one() -> None:
    frame = build_frame()

    result = reduction.reduce_frame(frame)
    scored = result.scores.join(frame["pm25"])
    dirtiest = scored.loc[scored["pm25"].idxmax(), "esi"]
    cleanest = scored.loc[scored["pm25"].idxmin(), "esi"]

    assert dirtiest > cleanest


def test_a_window_with_no_spread_reports_undefined_rather_than_zero() -> None:
    frame = build_frame(rows=100)
    model, _ = reduction.fit(frame)
    degenerate = replace(model, pc1_min=1.0, pc1_max=1.0)

    assert degenerate.is_degenerate
    assert model.score(frame).notna().all()
    assert degenerate.score(frame).isna().all()


# --- Missing data -----------------------------------------------------------


def test_incomplete_rows_are_excluded_and_counted() -> None:
    """PCA has no notion of a missing value. Dropping rows silently would let
    an ESI from a tenth of the window look like one from all of it."""
    frame = build_frame()
    frame.loc[0:9, "pm25"] = np.nan

    result = reduction.reduce_frame(frame)

    assert result.rows_dropped == 10
    assert result.rows_used == len(frame) - 10
    assert any("excluded" in caveat for caveat in result.caveats)


# --- Interpretability and provenance (task 6.6) -----------------------------


def test_every_result_carries_its_loadings_and_its_caveats() -> None:
    """design §9: the loadings are shown so the index stays interpretable
    rather than a black box."""
    frame = build_frame()

    payload = reduction.reduce_frame(frame).as_dict()

    assert payload["components"][0]["loadings"]
    assert payload["components"][0]["drivers"][0] in MEASUREMENT_COLUMNS
    assert any("not an absolute" in caveat for caveat in payload["caveats"])


def test_the_index_disclaims_causation_and_calibration() -> None:
    """ETH-1, and the misreading an index invites: the ESI compresses
    association within one window, and is neither an explanation nor a health
    threshold. Both disclaimers ship inside the payload rather than being left
    to whoever writes the UI."""
    frame = build_frame()

    text = " ".join(reduction.reduce_frame(frame).caveats).lower()

    assert "makes no claim about what caused them" in text
    assert "not 'unsafe'" in text
    assert "eth-1" in text


def test_a_model_survives_a_round_trip() -> None:
    frame = build_frame()
    model, _ = reduction.fit(frame)

    restored = ESIModel.from_dict(model.as_dict())

    assert restored == model
    pd.testing.assert_series_equal(restored.score(frame), model.score(frame))


def test_the_fitted_pca_is_written_beside_the_other_artefacts(
    settings, tmp_path
) -> None:
    """Task 6.2. Explained variance and loadings are the parts worth keeping:
    they are what makes a stored ESI readable months later."""
    frame = build_frame()
    result = reduction.reduce_frame(frame)

    path = service.persist_pca_model(
        result, settings=settings.model_copy(update={"data_processed_dir": str(tmp_path)})
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    assert Path(path).name == service.PCA_MODEL_FILENAME
    assert payload["components"][0]["loadings"]["pm25"] > 0
    assert payload["components"][0]["explained_variance_ratio"] > 0
    assert payload["esi"]["max"] == pytest.approx(100.0, abs=1e-6)
    assert ESIModel.from_dict(payload["model"]) == result.model


def test_a_stored_model_scores_new_data_on_the_original_calibration() -> None:
    """What the dashboard needs: an ESI that moves because the air changed, not
    because the window did."""
    frame = build_frame(rows=400)
    model, _ = reduction.fit(frame.iloc[:200])

    replayed = reduction.reduce_frame(frame.iloc[200:], model=model)

    assert replayed.model is model
    assert replayed.scores["esi"].between(ESI_MIN, ESI_MAX).all()
