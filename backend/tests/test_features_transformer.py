"""The single deterministic transformer (task 5.5, 5.7; design §8, AC-8).

The exit criterion of Phase 5 -- "feature set reproducible and byte-identical
between training and inference paths" -- is a claim that can be *checked*, so
it is checked literally: build the features over a long history, build them
again over the trailing window a prediction request would load, and compare the
two rows byte for byte after serialisation.

That test is the reason the rest of the phase is shaped the way it is. Every
design decision that looks fussy elsewhere -- no MICE in the feature path, a
time-based grid rather than a row shift, the log decision frozen at fit time --
exists so that this assertion can hold.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from services.datasets import prepare_frame
from services.features import service
from services.features.spec import DEFAULT_SPEC, FeatureSpec, LagSpec
from services.features.transformer import (
    FeatureTransformer,
    complete_mask,
    feature_matrix,
)
from tests.test_features_windows import START, build_frame


def realistic_frame(rows: int = 400, *, stations: int = 1, seed: int = 11) -> pd.DataFrame:
    """Lognormal pollutants, so the skew that drives task 5.4 is really there."""
    rng = np.random.default_rng(seed)
    records = []
    for station in range(stations):
        for hour in range(rows):
            records.append(
                {
                    "id": station * rows + hour + 1,
                    "source_id": 1,
                    "timestamp": START + timedelta(hours=hour),
                    "lat": 28.6 + station * 0.1,
                    "lon": 77.2,
                    "pm25": float(rng.lognormal(4.0, 0.7)),
                    "pm10": float(rng.lognormal(4.7, 0.6)),
                    "temp": float(rng.normal(22, 4)),
                    "humidity": float(rng.uniform(30, 80)),
                    "traffic_score": float(rng.uniform(20, 80)),
                    "is_anomaly": False,
                }
            )
    return prepare_frame(pd.DataFrame(records))


# --- Fitting (task 5.4) -----------------------------------------------------


def test_fitting_chooses_the_log_columns_and_records_why() -> None:
    frame = realistic_frame()

    transformer = FeatureTransformer.fit(frame)

    assert "pm25" in transformer.spec.log_columns
    assert transformer.skew_evidence["pm25"]["rationale"]
    assert transformer.is_fitted


def test_a_symmetric_column_is_not_transformed() -> None:
    """The transform costs interpretability, so it has to buy something."""
    frame = build_frame(rows=200)  # pm25 is a linear ramp: no skew at all

    transformer = FeatureTransformer.fit(frame)

    assert transformer.spec.log_columns == ()


def test_the_log_feature_is_log1p_of_the_column() -> None:
    frame = realistic_frame()

    result = FeatureTransformer.fit(frame).transform(frame)

    assert result["pm25_log"].iloc[10] == pytest.approx(np.log1p(frame["pm25"].iloc[10]))


def test_fitting_is_deterministic() -> None:
    frame = realistic_frame()

    first = FeatureTransformer.fit(frame)
    second = FeatureTransformer.fit(frame)

    assert first.fingerprint == second.fingerprint
    assert first.spec == second.spec


def test_transforming_twice_gives_the_same_numbers() -> None:
    frame = realistic_frame(rows=200)
    transformer = FeatureTransformer.fit(frame)

    pd.testing.assert_frame_equal(
        transformer.transform(frame), transformer.transform(frame), check_exact=True
    )


# --- The exit criterion -----------------------------------------------------


def _row(frame: pd.DataFrame, at: datetime, names: list[str]) -> pd.Series:
    return frame[frame["timestamp"] == at][names].iloc[0]


def test_training_and_inference_paths_produce_the_identical_row() -> None:
    """Phase 5's exit criterion.

    The training path sees 400 hours. The inference path sees only what
    ``history_start`` says the hour needs. Both must describe that hour the same
    way, or a model trained on one and served by the other is being fed a
    different feature set than it learned.
    """
    frame = realistic_frame()
    transformer = FeatureTransformer.fit(frame)
    names = list(transformer.feature_names)

    at = START + timedelta(hours=300)
    full = transformer.transform(frame)

    start = service.history_start(at, transformer.spec)
    window = frame[frame["timestamp"].between(start, at)].reset_index(drop=True)
    partial = transformer.transform(window)

    pd.testing.assert_series_equal(
        _row(full, at, names), _row(partial, at, names), check_names=False
    )


def test_the_two_paths_serialise_to_identical_bytes() -> None:
    """"Byte-identical" taken at its word: equal floats that round-trip to
    different text would still break a stored feature file."""
    frame = realistic_frame()
    transformer = FeatureTransformer.fit(frame)
    names = list(transformer.feature_names)
    at = START + timedelta(hours=300)

    full = transformer.transform(frame)
    window = frame[
        frame["timestamp"].between(service.history_start(at, transformer.spec), at)
    ].reset_index(drop=True)
    partial = transformer.transform(window)

    assert _row(full, at, names).to_json() == _row(partial, at, names).to_json()


def test_the_paths_agree_even_when_the_window_starts_inside_a_gap() -> None:
    """The repair stage is window-local, so a short outage near the window edge
    must not make the two paths diverge."""
    frame = realistic_frame()
    outage = frame["timestamp"].between(
        START + timedelta(hours=280), START + timedelta(hours=281)
    )
    frame.loc[outage, "pm25"] = np.nan

    transformer = FeatureTransformer.fit(frame)
    names = list(transformer.feature_names)
    at = START + timedelta(hours=300)

    full, _ = service.build(frame, transformer)
    window = frame[
        frame["timestamp"].between(service.history_start(at, transformer.spec), at)
    ].reset_index(drop=True)
    partial, _ = service.build(window, transformer)

    pd.testing.assert_series_equal(
        _row(full, at, names), _row(partial, at, names), check_names=False
    )


def test_a_serialised_transformer_rebuilds_the_same_features(tmp_path) -> None:
    """The artefact-to-serving path: fit here, reload there, same numbers."""
    frame = realistic_frame()
    fitted = FeatureTransformer.fit(frame)
    path = fitted.save(tmp_path / "feature_spec.json")

    reloaded = FeatureTransformer.load(path)

    assert reloaded.fingerprint == fitted.fingerprint
    assert reloaded.spec == fitted.spec
    pd.testing.assert_frame_equal(
        fitted.transform(frame), reloaded.transform(frame), check_exact=True
    )


# --- Transform mechanics ----------------------------------------------------


def test_rows_come_back_in_the_order_they_went_in() -> None:
    frame = realistic_frame(rows=50)
    shuffled = frame.sample(frac=1.0, random_state=5)

    result = FeatureTransformer.fit(frame).transform(shuffled)

    assert list(result.index) == list(shuffled.index)
    assert list(result["timestamp"]) == list(shuffled["timestamp"])


def test_an_unsorted_frame_still_gets_correct_lags() -> None:
    """Sorting happens inside the transform: an unsorted shift is wrong, not
    merely untidy."""
    frame = build_frame(rows=60)
    shuffled = frame.sample(frac=1.0, random_state=3)

    result = FeatureTransformer(spec=FeatureSpec(lags=(LagSpec("pm25", 1),), rollings=())).transform(
        shuffled
    )
    at_30 = result[result["timestamp"] == START + timedelta(hours=30)].iloc[0]

    assert at_30["pm25_lag_1h"] == pytest.approx(29.0)


def test_a_duplicate_index_is_refused() -> None:
    frame = realistic_frame(rows=20)
    frame.index = [0] * len(frame)

    with pytest.raises(ValueError, match="unique row index"):
        FeatureTransformer().transform(frame)


def test_a_frame_without_a_station_column_derives_one() -> None:
    frame = realistic_frame(rows=60).drop(columns=["station"])

    result = FeatureTransformer().transform(frame)

    assert result["pm25_lag_1h"].iloc[30] == pytest.approx(frame["pm25"].iloc[29])


def test_a_frame_with_no_timestamp_is_refused() -> None:
    frame = realistic_frame(rows=10).drop(columns=["timestamp"])

    with pytest.raises(ValueError, match="missing required columns"):
        FeatureTransformer().transform(frame)


def test_stations_are_featured_independently() -> None:
    frame = realistic_frame(rows=100, stations=3)

    result = FeatureTransformer.fit(frame).transform(frame)

    for _, group in result.groupby("station"):
        ordered = group.sort_values("timestamp")
        assert np.isnan(ordered["pm25_lag_1h"].iloc[0])
        assert ordered["pm25_lag_1h"].iloc[1] == pytest.approx(ordered["pm25"].iloc[0])


# --- The matrix handed to Phase 7 -------------------------------------------


def test_the_matrix_comes_back_in_the_spec_order() -> None:
    frame = realistic_frame(rows=100)
    transformer = FeatureTransformer.fit(frame)
    result = transformer.transform(frame)

    matrix = feature_matrix(result, transformer.spec)

    assert list(matrix.columns) == list(transformer.feature_names)


def test_an_incomplete_frame_is_refused_by_the_matrix() -> None:
    frame = realistic_frame(rows=60)
    result = FeatureTransformer().transform(frame).drop(columns=["pm25_lag_24h"])

    with pytest.raises(ValueError, match="missing engineered columns"):
        feature_matrix(result, DEFAULT_SPEC)


def test_the_complete_mask_excludes_only_the_warm_up() -> None:
    frame = realistic_frame(rows=200)
    transformer = FeatureTransformer.fit(frame)

    mask = complete_mask(transformer.transform(frame), transformer.spec)

    assert not mask.iloc[:24].any()
    assert mask.iloc[24:].all()


def test_the_modelled_subset_is_never_silently_dropped() -> None:
    """The transform returns every input row, warm-up included. Which rows a
    model may use is Phase 7's decision, not this layer's."""
    frame = realistic_frame(rows=100)

    result = FeatureTransformer.fit(frame).transform(frame)

    assert len(result) == len(frame)


# --- Provenance -------------------------------------------------------------


def test_the_fit_window_is_recorded() -> None:
    """AC-8 turns on the log decision coming from training data alone; the
    stored window is how a reviewer checks that without rerunning anything."""
    frame = realistic_frame(rows=200)

    transformer = FeatureTransformer.fit(frame)

    assert transformer.fit_window is not None
    assert transformer.fit_window.rows == 200
    assert transformer.fit_window.start == START
    assert transformer.fitted_at is not None
    assert transformer.fitted_at.tzinfo == timezone.utc


def test_an_unfitted_transformer_still_builds_the_fixed_features() -> None:
    """Only the log decision needs fitting; the rest of the spec is fixed
    before any data is seen."""
    frame = realistic_frame(rows=100)

    result = FeatureTransformer().transform(frame)

    assert not FeatureTransformer().is_fitted
    assert result["pm25_lag_1h"].iloc[50] == pytest.approx(frame["pm25"].iloc[49])
