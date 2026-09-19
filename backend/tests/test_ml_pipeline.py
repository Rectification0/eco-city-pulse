"""Preprocessing, selection, the ladder and the metrics (tasks 7.3-7.8, 7.11).

The pieces between the split and the registry. Each is small enough to test on
its own, and each is a place a plausible-looking pipeline goes wrong: an encoder
that learns its categories from the window, a selection stage that reads the
test set, a baseline that is not actually persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from services.datasets import MEASUREMENT_COLUMNS
from services.features.spec import DEFAULT_SPEC, SEASONS
from services.features.transformer import FeatureTransformer
from services.ml import evaluation, models, preprocessing, selection
from tests.test_features_transformer import realistic_frame

# --- The model matrix (task 7.3) --------------------------------------------


def featured_frame(rows: int = 400) -> tuple[pd.DataFrame, FeatureTransformer]:
    frame = realistic_frame(rows=rows)
    transformer = FeatureTransformer.fit(frame)
    return transformer.transform(frame), transformer


def test_the_matrix_carries_the_measurements_and_the_engineered_features() -> None:
    """The reading at *t* is known when a forecast for *t + 1h* is made, and it
    is the single most informative input there is -- excluding it would handicap
    every model against the baseline it has to beat."""
    featured, transformer = featured_frame()

    matrix = preprocessing.encode(featured, transformer.spec)

    for column in MEASUREMENT_COLUMNS:
        assert column in matrix.columns
    assert "pm25_lag_24h" in matrix.columns


def test_season_is_one_hot_encoded_against_the_spec_not_the_window() -> None:
    """A fitted encoder would learn "this window contains winter and summer"
    and meet monsoon at serving time with no column for it."""
    featured, transformer = featured_frame()
    assert featured["season"].nunique() == 1  # a single-season window

    matrix = preprocessing.encode(featured, transformer.spec)

    for label in SEASONS:
        assert f"season_{label}" in matrix.columns
    assert "season" not in matrix.columns


def test_the_matrix_is_numeric_and_ordered_by_the_spec() -> None:
    featured, transformer = featured_frame()

    matrix = preprocessing.encode(featured, transformer.spec)

    assert list(matrix.columns) == list(preprocessing.model_columns(transformer.spec))
    assert matrix.to_numpy().dtype == np.float64


def test_a_frame_missing_a_feature_is_refused() -> None:
    """A model quietly fitted on fewer columns than it claims is worse than a
    failure."""
    featured, transformer = featured_frame()

    with pytest.raises(ValueError, match="missing model columns"):
        preprocessing.encode(featured.drop(columns=["pm25_lag_1h"]), transformer.spec)


def test_the_scaler_lives_inside_the_model() -> None:
    """Structural, not disciplinary: there is no call site at which the scaler
    could be handed test rows (AC-8)."""
    wrapped = preprocessing.with_scaler(Ridge(), scale=True)
    bare = preprocessing.with_scaler(Ridge(), scale=False)

    assert preprocessing.fitted_scaler(wrapped) is not None
    assert preprocessing.fitted_scaler(bare) is None


def test_a_fitted_scaler_reports_the_rows_it_saw() -> None:
    featured, transformer = featured_frame()
    matrix = preprocessing.encode(featured, transformer.spec).dropna()
    train = matrix.iloc[:200]

    model = preprocessing.with_scaler(Ridge(), scale=True)
    model.fit(train, np.arange(len(train), dtype="float64"))
    scaler = preprocessing.fitted_scaler(model)

    assert scaler.n_samples_seen_ == 200
    assert np.allclose(scaler.mean_, train.mean().to_numpy())


# --- Feature selection (task 7.4) -------------------------------------------


def test_a_constant_column_is_dropped_with_its_reason() -> None:
    matrix = pd.DataFrame(
        {"pm25": np.arange(100.0), "flat": np.ones(100), "noise": np.random.default_rng(1).normal(size=100)}
    )
    target = pd.Series(np.arange(100.0))

    result = selection.select(matrix, target)

    assert "flat" not in result.kept
    assert "near-constant" in result.dropped["flat"]


def test_one_of_a_collinear_pair_is_dropped() -> None:
    """``pm25`` and ``pm25_log`` are the same measurement twice; a tree would
    split its importance between two names and the Phase 8 explanation would
    understate both."""
    base = np.linspace(10, 200, 200)
    matrix = pd.DataFrame(
        {"pm25": base, "pm25_copy": base * 2 + 0.001, "temp": np.sin(base)}
    )
    target = pd.Series(base + 1)

    result = selection.select(matrix, target)

    assert "pm25_copy" not in result.kept
    assert "|r|=" in result.dropped["pm25_copy"]


def test_the_kept_column_is_the_one_closer_to_the_target() -> None:
    rng = np.random.default_rng(3)
    signal = np.linspace(0, 100, 300)
    matrix = pd.DataFrame(
        {
            "useful": signal + rng.normal(0, 0.5, 300),
            "noisy_twin": signal + rng.normal(0, 8, 300),
        }
    )
    target = pd.Series(signal)

    result = selection.select(matrix, target, max_correlation=0.8)

    assert "useful" in result.kept
    assert "noisy_twin" not in result.kept


def test_pm25_is_never_dropped() -> None:
    """It is the persistence baseline's only input; removing it would make the
    comparison of AC-7 meaningless."""
    matrix = pd.DataFrame({"pm25": np.ones(100), "other": np.arange(100.0)})
    target = pd.Series(np.arange(100.0))

    result = selection.select(matrix, target)

    assert "pm25" in result.kept


def test_selection_reports_what_it_did() -> None:
    matrix = pd.DataFrame({"pm25": np.arange(100.0), "flat": np.ones(100)})

    payload = selection.select(matrix, pd.Series(np.arange(100.0))).as_dict()

    assert payload["kept_count"] == 1
    assert payload["dropped_count"] == 1


def test_applying_a_selection_preserves_its_order() -> None:
    matrix = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    result = selection.SelectionResult(kept=("c", "a"))

    assert list(selection.apply(matrix, result).columns) == ["c", "a"]


# --- The ladder (tasks 7.5-7.8) ---------------------------------------------


def test_the_baseline_predicts_the_current_reading() -> None:
    """Persistence, exactly: not "the row above", which across a missing hour
    would be some older reading."""
    matrix = pd.DataFrame({"pm25": [10.0, 20.0, 30.0], "temp": [1.0, 2.0, 3.0]})

    model = models.NaiveLag1().fit(matrix, pd.Series([11.0, 21.0, 31.0]))

    assert list(model.predict(matrix)) == [10.0, 20.0, 30.0]


def test_the_baseline_has_nothing_to_overfit_with() -> None:
    """Its score is a property of the data, not of a modelling choice -- which
    is what makes it an honest floor."""
    matrix = pd.DataFrame({"pm25": [10.0, 20.0]})

    trained = models.NaiveLag1().fit(matrix, pd.Series([1.0, 2.0]))
    untrained_predictions = models.NaiveLag1().fit(matrix, pd.Series([99.0, 99.0]))

    assert list(trained.predict(matrix)) == list(untrained_predictions.predict(matrix))


def test_the_baseline_refuses_a_matrix_without_its_column() -> None:
    with pytest.raises(ValueError, match="persistence baseline needs"):
        models.NaiveLag1().fit(pd.DataFrame({"temp": [1.0]}), pd.Series([1.0]))


def test_the_ladder_is_ordered_baseline_first() -> None:
    """A report read top to bottom sets the bar before it shows anything
    clearing it."""
    ladder = models.build_ladder()

    assert ladder[0].name == models.BASELINE_MODEL
    assert ladder[0].is_baseline
    assert ladder[-1].name == models.PRODUCTION_MODEL


def test_only_the_linear_model_asks_for_scaling() -> None:
    """Trees are invariant to monotone rescaling; scaling them costs a
    transform and buys nothing."""
    by_name = models.ladder_by_name()

    assert by_name["ridge"].scale
    assert not by_name["random_forest"].scale
    assert not by_name["xgboost"].scale


def test_every_rung_is_seeded() -> None:
    """No hyperparameter search, and no run-to-run drift."""
    first = models.ladder_by_name()["random_forest"].build()
    second = models.ladder_by_name()["random_forest"].build()

    assert first.get_params()["random_state"] == second.get_params()["random_state"]


def test_every_rung_carries_its_hyperparameters_for_the_model_lab() -> None:
    for spec in models.build_ladder():
        if spec.is_baseline:
            continue
        assert spec.hyperparameters, f"{spec.name} reports no hyperparameters"
        assert spec.notes


# --- Metrics (task 7.11) ----------------------------------------------------


def test_the_three_metrics_are_computed() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 18.0, 33.0, 39.0])

    metrics = evaluation.score(truth, predicted)

    assert metrics.mae == pytest.approx(2.0)
    assert metrics.rmse == pytest.approx(np.sqrt(np.mean((truth - predicted) ** 2)))
    assert metrics.rows == 4


def test_a_perfect_forecast_scores_perfectly() -> None:
    truth = np.array([1.0, 2.0, 3.0])

    metrics = evaluation.score(truth, truth)

    assert metrics.mae == 0.0
    assert metrics.r2 == pytest.approx(1.0)


def test_rmse_punishes_a_single_large_miss_more_than_mae_does() -> None:
    """Which is why both are reported: read together they say whether the error
    is evenly spread or concentrated in a few bad hours."""
    truth = np.zeros(10)
    spread = np.full(10, 2.0)
    concentrated = np.array([20.0, *np.zeros(9)])

    even = evaluation.score(truth, spread)
    spiky = evaluation.score(truth, concentrated)

    assert even.mae == spiky.mae
    assert spiky.rmse > even.rmse


def test_skill_is_the_fractional_improvement_over_the_baseline() -> None:
    model = evaluation.Metrics(mae=8.0, rmse=10.0, r2=0.8, rows=100)
    baseline = evaluation.Metrics(mae=10.0, rmse=13.0, r2=0.7, rows=100)

    assert evaluation.skill(model, baseline) == pytest.approx(0.2)


def test_a_model_worse_than_doing_nothing_reports_negative_skill() -> None:
    """A result worth seeing plainly rather than hiding behind an R²."""
    model = evaluation.Metrics(mae=12.0, rmse=15.0, r2=0.6, rows=100)
    baseline = evaluation.Metrics(mae=10.0, rmse=13.0, r2=0.7, rows=100)

    assert evaluation.skill(model, baseline) < 0


def test_fold_summaries_report_the_spread_not_just_the_mean() -> None:
    """A model whose MAE doubles between the first fold and the last is not
    stable over time, which an average alone would hide."""
    folds = [
        evaluation.Metrics(mae=mae, rmse=mae * 1.3, r2=0.7, rows=100)
        for mae in (5.0, 7.0, 9.0)
    ]

    summary = evaluation.summarise_folds(folds)

    assert summary["folds"] == 3
    assert summary["mae_mean"] == pytest.approx(7.0)
    assert summary["mae_std"] > 0
    assert summary["mae_per_fold"] == [5.0, 7.0, 9.0]


def test_no_folds_is_reported_rather_than_faked() -> None:
    assert evaluation.summarise_folds([])["folds"] == 0


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        evaluation.score(np.array([1.0, 2.0]), np.array([1.0]))


def test_the_r2_caveat_is_stated_where_r2_is_produced() -> None:
    """On an autocorrelated series R² flatters: persistence alone scores above
    0.9 at one hour."""
    assert "autocorrelated" in evaluation.R2_CAVEAT
    assert "baseline" in evaluation.R2_CAVEAT


def test_the_default_spec_still_produces_the_columns_the_matrix_expects() -> None:
    """Guards the seam between Phase 5 and Phase 7: a feature renamed there
    would otherwise surface as a confusing KeyError here."""
    columns = preprocessing.model_columns(DEFAULT_SPEC)

    assert set(MEASUREMENT_COLUMNS) <= set(columns)
    assert "pm25_rolling_mean_24h" in columns
