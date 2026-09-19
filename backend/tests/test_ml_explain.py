"""Explainability (tasks 8.1-8.4, design §10.2).

One property carries this phase, and it is asserted for every rung of the
ladder:

    prediction = base_value + Σ contributions

That identity is what separates an explanation from a decoration. A feature
importance that does not add up cannot be checked, and an unfalsifiable
explanation of an opaque model is exactly the thing FEAT-06 exists to remove.

The fixtures are built so the right answer is known in advance: ``pm25``
dominates by construction, so a summary that fails to rank it first is wrong
rather than merely surprising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from services.ml import explain, models, preprocessing

COLUMNS = ("pm25", "temp", "traffic_score", "humidity")


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A matrix whose dominant driver is known by construction."""
    rng = np.random.default_rng(11)
    rows = 600
    matrix = pd.DataFrame(
        {
            "pm25": rng.uniform(20, 180, rows),
            "temp": rng.normal(24, 5, rows),
            "traffic_score": rng.uniform(10, 90, rows),
            "humidity": rng.uniform(30, 80, rows),
        }
    )
    target = (
        0.9 * matrix["pm25"]
        + 0.3 * matrix["traffic_score"]
        - 0.2 * matrix["temp"]
        + rng.normal(0, 2, rows)
    )
    return matrix, target


def fitted(name: str, dataset: tuple[pd.DataFrame, pd.Series]):
    """One rung of the ladder, fitted on the fixture."""
    matrix, target = dataset
    spec = models.ladder_by_name()[name]
    model = preprocessing.with_scaler(spec.build(), scale=spec.scale)
    model.fit(matrix, target)
    return model


LADDER = ("naive_lag1", "ridge", "random_forest", "xgboost")


# --- The identity that makes an explanation checkable -----------------------


@pytest.mark.parametrize("name", LADDER)
def test_contributions_sum_to_the_prediction(
    name: str, dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """SHAP's additivity, asserted rather than assumed, for every model."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted(name, dataset), COLUMNS)

    attributions = explainer.explain(matrix.iloc[:25])

    for attribution in attributions:
        assert attribution.is_additive, (
            f"{name}: {attribution.base_value} + {attribution.total} "
            f"!= {attribution.prediction}"
        )


@pytest.mark.parametrize("name", ("naive_lag1", "ridge"))
def test_the_exact_strategies_are_exact_to_floating_point(
    name: str, dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Ridge and persistence are solved in closed form, so their additivity is
    not an approximation that happens to be close."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted(name, dataset), COLUMNS)

    attribution = explainer.explain_one(matrix.iloc[[7]])
    error = abs(attribution.base_value + attribution.total - attribution.prediction)

    assert error < 1e-9


def test_a_bare_linear_model_without_a_scaler_is_also_exact(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """The other branch of the linear strategy: no pipeline, no fitted means."""
    matrix, target = dataset
    model = Ridge(alpha=1.0).fit(matrix, target)

    attribution = explain.ModelExplainer(model, COLUMNS).explain_one(matrix.iloc[[3]])

    assert attribution.is_additive
    assert attribution.method == "linear_exact"


# --- Which strategy each model gets (task 8.1) ------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("naive_lag1", "persistence_exact"),
        ("ridge", "linear_exact"),
        ("random_forest", "shap_tree:tree_path_dependent"),
        ("xgboost", "shap_tree:tree_path_dependent"),
    ],
)
def test_each_model_is_explained_by_the_right_method(
    name: str, expected: str, dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Computing a known-exact answer in closed form is faster *and* more
    honest than approximating it with a sampler."""
    explainer = explain.ModelExplainer(fitted(name, dataset), COLUMNS)

    assert explainer.method == expected


def test_the_baseline_attributes_everything_to_pm25(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Degenerate, and worth stating: the baseline's entire reasoning is "it
    will stay as it is"."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("naive_lag1", dataset), COLUMNS)

    attribution = explainer.explain_one(matrix.iloc[[0]])

    assert attribution.contributions["pm25"] != 0
    assert all(
        value == 0 for name, value in attribution.contributions.items() if name != "pm25"
    )


# --- Per-prediction attribution (task 8.3) ----------------------------------


@pytest.mark.parametrize("name", LADDER)
def test_top_features_are_ranked_by_magnitude(
    name: str, dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted(name, dataset), COLUMNS)

    top = explainer.explain_one(matrix.iloc[[5]]).top_features(3)

    magnitudes = [abs(value) for value in top.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(top) <= 3


def test_contributions_keep_their_sign(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """"Traffic: +4.1" and "traffic: −4.1" are opposite explanations of the same
    forecast; an absolute-valued importance would show them identically."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("ridge", dataset), COLUMNS)

    cleanest = matrix["pm25"].idxmin()
    dirtiest = matrix["pm25"].idxmax()
    low = explainer.explain_one(matrix.loc[[cleanest]])
    high = explainer.explain_one(matrix.loc[[dirtiest]])

    assert low.contributions["pm25"] < 0 < high.contributions["pm25"]


def test_a_single_row_is_required_for_explain_one(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("ridge", dataset), COLUMNS)

    with pytest.raises(ValueError, match="exactly one row"):
        explainer.explain_one(matrix.iloc[:3])


def test_the_attribution_payload_carries_its_caveat(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """ETH-1: an attribution describes the model, not the atmosphere."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("xgboost", dataset), COLUMNS)

    payload = explainer.explain_one(matrix.iloc[[1]]).as_dict()

    assert "not evidence that the feature caused" in payload["caveat"]
    assert payload["additive"]
    assert payload["top_features"]


# --- Global importance (task 8.2) -------------------------------------------


@pytest.mark.parametrize("name", LADDER)
def test_the_dominant_driver_is_ranked_first(
    name: str, dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """The fixture makes PM2.5 dominant by construction, so a summary that
    misses it is wrong rather than merely surprising."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted(name, dataset), COLUMNS)

    importance = explainer.global_importance(matrix, sample_rows=200)

    assert importance.features[0].feature == "pm25"


def test_shares_are_a_proportion_of_the_whole(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("xgboost", dataset), COLUMNS)

    importance = explainer.global_importance(matrix, sample_rows=200)

    assert sum(item.share for item in importance.features) == pytest.approx(1.0)
    assert all(0 <= item.share <= 1 for item in importance.features)


def test_direction_separates_a_one_way_driver_from_a_two_way_one(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """A feature can matter enormously and push both ways depending on its
    value, which mean |SHAP| alone would hide."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("ridge", dataset), COLUMNS)

    importance = explainer.global_importance(matrix, sample_rows=300)
    by_name = {item.feature: item for item in importance.features}

    # pm25 spans low and high values around its mean, so it does both.
    assert by_name["pm25"].direction == "mixed"


def test_the_summary_is_sampled_evenly_and_reproducibly(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """A stride rather than a random draw: the same request twice gives the
    same answer, and the whole period stays represented."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("xgboost", dataset), COLUMNS)

    first = explainer.global_importance(matrix, sample_rows=150)
    second = explainer.global_importance(matrix, sample_rows=150)

    assert first.rows == second.rows == 150
    assert [item.feature for item in first.features] == [
        item.feature for item in second.features
    ]
    assert first.features[0].mean_abs_shap == second.features[0].mean_abs_shap


def test_the_ranking_is_stable_well_below_the_default_sample(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Why the default is 500 and not 5000: mean |SHAP| settles long before the
    cost does, and TreeSHAP is linear in rows."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("xgboost", dataset), COLUMNS)

    small = explainer.global_importance(matrix, sample_rows=100)
    large = explainer.global_importance(matrix, sample_rows=500)

    assert [item.feature for item in small.features[:3]] == [
        item.feature for item in large.features[:3]
    ]


def test_an_empty_matrix_is_refused(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("ridge", dataset), COLUMNS)

    with pytest.raises(ValueError, match="empty matrix"):
        explainer.global_importance(matrix.iloc[:0])


# --- Analyst configuration (task 8.4) ---------------------------------------


def test_the_interventional_perturbation_needs_a_background_sample(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """It integrates over a reference set; the path-dependent variant reads the
    trees themselves and needs none."""
    matrix, _ = dataset
    explainer = explain.ModelExplainer(
        fitted("xgboost", dataset), COLUMNS, perturbation=explain.INTERVENTIONAL
    )

    with pytest.raises(ValueError, match="background sample"):
        explainer.explain(matrix.iloc[:5])


def test_the_interventional_perturbation_works_with_one(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    matrix, _ = dataset
    explainer = explain.ModelExplainer(
        fitted("random_forest", dataset),
        COLUMNS,
        perturbation=explain.INTERVENTIONAL,
        background=matrix.iloc[:100],
    )

    attribution = explainer.explain_one(matrix.iloc[[10]])

    assert attribution.is_additive
    assert attribution.method == "shap_tree:interventional"


def test_an_unsupported_perturbation_for_a_model_says_what_to_use_instead(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """XGBoost 3 emits categorical-split metadata even for entirely numeric
    features, and SHAP's interventional path refuses it. Reported rather than
    silently downgraded: an analyst who asked for interventional values and
    quietly received path-dependent ones would draw conclusions from a method
    they did not choose.
    """
    matrix, _ = dataset
    explainer = explain.ModelExplainer(
        fitted("xgboost", dataset),
        COLUMNS,
        perturbation=explain.INTERVENTIONAL,
        background=matrix.iloc[:100],
    )

    with pytest.raises(explain.ExplainerUnavailableError) as caught:
        explainer.explain(matrix.iloc[:5])

    assert explain.PATH_DEPENDENT in str(caught.value)


def test_an_unknown_perturbation_is_refused(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    with pytest.raises(ValueError, match="unknown perturbation"):
        explain.ModelExplainer(fitted("ridge", dataset), COLUMNS, perturbation="magic")


def test_a_matrix_missing_an_explained_column_is_refused(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    matrix, _ = dataset
    explainer = explain.ModelExplainer(fitted("ridge", dataset), COLUMNS)

    with pytest.raises(ValueError, match="missing explained columns"):
        explainer.explain(matrix.drop(columns=["temp"]).iloc[:3])


# --- Building from an artifact (task 8.1) -----------------------------------


def test_an_explainer_can_be_built_from_a_registered_artifact(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """What the Model Lab does: load the artifact, explain the model."""
    matrix, _ = dataset
    artifact = {"estimator": fitted("xgboost", dataset), "columns": list(COLUMNS)}

    explainer = explain.for_artifact(artifact)

    assert explainer.columns == COLUMNS
    assert explainer.explain_one(matrix.iloc[[2]]).is_additive


def test_a_forest_is_explained_without_the_scaler_it_never_had(
    dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Trees are handed the matrix unwrapped (task 7.3), so the explainer must
    not assume a pipeline is there."""
    matrix, target = dataset
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(matrix, target)

    attribution = explain.ModelExplainer(model, COLUMNS).explain_one(matrix.iloc[[4]])

    assert attribution.is_additive
