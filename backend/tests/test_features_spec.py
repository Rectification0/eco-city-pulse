"""The feature contract (task 5.5, specs §6.1).

A spec that cannot survive a round trip through JSON cannot travel with a model
artefact, and a spec that cannot travel with the artefact cannot be replayed at
inference time. So the serialisation is tested as carefully as the arithmetic.
"""

from __future__ import annotations

import pytest

from services.features.spec import (
    DEFAULT_SPEC,
    SPEC_VERSION,
    TEMPORAL_FEATURES,
    FeatureSpec,
    LagSpec,
    RollingSpec,
)

# --- The features specs §6.1 names ------------------------------------------


def test_the_default_spec_is_the_feature_set_the_requirements_name() -> None:
    assert DEFAULT_SPEC.feature_names == (
        *TEMPORAL_FEATURES,
        "pm25_lag_1h",
        "pm25_lag_24h",
        "temp_lag_3h",
        "pm25_rolling_mean_24h",
        "traffic_score_rolling_std_6h",
    )


def test_a_fitted_log_column_extends_the_set() -> None:
    fitted = DEFAULT_SPEC.with_log_columns(("pm25",))

    assert fitted.feature_names[-1] == "pm25_log"
    assert DEFAULT_SPEC.feature_names[-1] != "pm25_log"  # the original is unchanged


def test_history_hours_covers_the_longest_lookback() -> None:
    """What an inference request has to load. 24 here: the 24-hour lag and the
    24-hour window reach equally far back."""
    assert DEFAULT_SPEC.history_hours == 24
    assert FeatureSpec(lags=(LagSpec("pm25", 72),), rollings=()).history_hours == 72


def test_a_strictly_prior_window_reaches_one_hour_further() -> None:
    spec = FeatureSpec(
        lags=(), rollings=(RollingSpec("pm25", 24, "mean", include_current=False),)
    )

    assert spec.history_hours == 25


def test_source_columns_are_the_measurements_the_spec_reads() -> None:
    assert DEFAULT_SPEC.source_columns == ("pm25", "temp", "traffic_score")


# --- Validation -------------------------------------------------------------


def test_a_lag_on_an_unknown_column_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown column"):
        LagSpec("no2", 1)


def test_a_zero_hour_lag_is_refused() -> None:
    """A lag of zero is the column itself, under a name that claims otherwise."""
    with pytest.raises(ValueError, match="lag hours"):
        LagSpec("pm25", 0)


def test_an_unknown_rolling_statistic_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown statistic"):
        RollingSpec("pm25", 24, "kurtosis")


def test_two_specs_producing_the_same_name_are_refused() -> None:
    """Silently keeping one of two identically-named features would make the
    matrix narrower than the spec claims."""
    with pytest.raises(ValueError, match="duplicate feature names"):
        FeatureSpec(lags=(LagSpec("pm25", 1), LagSpec("pm25", 1)))


def test_an_impossible_coverage_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="min_window_coverage"):
        FeatureSpec(min_window_coverage=1.5)


def test_a_rolling_std_requires_two_observations() -> None:
    assert RollingSpec("traffic_score", 6, "std").min_periods(0.1) == 2
    assert RollingSpec("pm25", 24, "mean").min_periods(0.5) == 12


# --- Serialisation ----------------------------------------------------------


def test_a_spec_survives_a_round_trip() -> None:
    original = DEFAULT_SPEC.with_log_columns(("pm25", "pm10"))

    restored = FeatureSpec.from_dict(original.as_dict())

    assert restored == original
    assert restored.feature_names == original.feature_names


def test_the_serialised_form_spells_out_the_derived_names() -> None:
    """A reader of the stored spec should not have to re-implement the naming
    rule to know what the columns are called."""
    payload = DEFAULT_SPEC.as_dict()

    assert payload["feature_names"] == list(DEFAULT_SPEC.feature_names)
    assert payload["history_hours"] == DEFAULT_SPEC.history_hours
    assert payload["spec_version"] == SPEC_VERSION


def test_a_spec_from_an_incompatible_version_is_rejected_not_guessed() -> None:
    payload = DEFAULT_SPEC.as_dict()
    payload["spec_version"] = SPEC_VERSION + 1

    with pytest.raises(ValueError, match="cannot be read by this build"):
        FeatureSpec.from_dict(payload)


def test_a_spec_is_immutable() -> None:
    with pytest.raises(AttributeError):
        DEFAULT_SPEC.temporal = False  # type: ignore[misc]
