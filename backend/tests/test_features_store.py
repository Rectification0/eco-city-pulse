"""The feature store (task 5.6, design §2).

A feature file separated from the spec that built it is a set of unlabelled
numbers: ``pm25_rolling_mean_24h`` means something different if the coverage
rule changed. So the round trip under test is not just the frame -- it is the
frame, the transformer and the manifest together.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from core.config import Settings
from core.exceptions import DatasetNotFoundError
from services import datasets
from services.features import store
from services.features.transformer import FeatureTransformer
from tests.test_features_transformer import realistic_frame


@pytest.fixture
def store_settings(settings: Settings, tmp_path) -> Settings:
    """Hermetic settings whose processed directory is a temp dir."""
    return settings.model_copy(update={"data_processed_dir": str(tmp_path)})


def build(rows: int = 200) -> tuple[pd.DataFrame, FeatureTransformer]:
    frame = realistic_frame(rows=rows)
    transformer = FeatureTransformer.fit(frame)
    return transformer.transform(frame), transformer


def test_a_build_writes_the_frame_its_spec_and_a_manifest(store_settings: Settings) -> None:
    featured, transformer = build()

    stored = store.write(
        featured,
        transformer,
        window=datasets.describe_window(featured),
        settings=store_settings,
    )

    features_path, spec_path, manifest_path = store.paths(store_settings)
    assert features_path.exists()
    assert spec_path.exists()
    assert manifest_path.exists()
    assert stored.rows == len(featured)


def test_the_manifest_records_what_the_file_contains(store_settings: Settings) -> None:
    featured, transformer = build()

    store.write(
        featured,
        transformer,
        window=datasets.describe_window(featured),
        settings=store_settings,
        dataset_version={"fingerprint": "abc123"},
    )
    manifest = store.read_manifest(store_settings)

    assert manifest["rows"] == len(featured)
    assert manifest["features"] == list(transformer.feature_names)
    assert manifest["fingerprint"] == transformer.fingerprint
    assert manifest["dataset_version"]["fingerprint"] == "abc123"
    # The warm-up rows have no 24-hour lag; that is expected, and visible.
    assert manifest["nulls"]["pm25_lag_24h"] == 24


def test_the_stored_frame_reads_back_with_its_dtypes(store_settings: Settings) -> None:
    """CSV loses the timezone and the numeric dtype; both are restored on read
    rather than left for the caller to rediscover."""
    featured, transformer = build()
    store.write(
        featured,
        transformer,
        window=datasets.describe_window(featured),
        settings=store_settings,
    )

    restored, restored_transformer = store.read(store_settings)

    assert str(restored["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert restored_transformer.spec == transformer.spec
    pd.testing.assert_frame_equal(
        restored[list(transformer.feature_names)],
        featured[list(transformer.feature_names)].reset_index(drop=True),
        check_dtype=False,
        atol=1e-9,
    )


def test_the_stored_transformer_rebuilds_the_same_features(store_settings: Settings) -> None:
    """What Phase 9 does: load the spec the model was trained with, and use it."""
    frame = realistic_frame(rows=200)
    transformer = FeatureTransformer.fit(frame)
    store.write(
        transformer.transform(frame),
        transformer,
        window=datasets.describe_window(frame),
        settings=store_settings,
    )

    reloaded = store.load_transformer(store_settings)

    assert reloaded.fingerprint == transformer.fingerprint
    pd.testing.assert_frame_equal(
        reloaded.transform(frame), transformer.transform(frame), check_exact=True
    )


def test_completeness_is_reported_not_silently_dropped(store_settings: Settings) -> None:
    featured, transformer = build(rows=200)

    stored = store.write(
        featured,
        transformer,
        window=datasets.describe_window(featured),
        settings=store_settings,
    )

    assert stored.rows == 200
    assert stored.complete_rows == 176  # 200 less the 24-hour warm-up
    assert stored.complete_pct == 88.0


def test_reading_before_anything_was_built_says_what_to_run(store_settings: Settings) -> None:
    with pytest.raises(DatasetNotFoundError, match="scripts.build_features"):
        store.read(store_settings)

    with pytest.raises(DatasetNotFoundError, match="scripts.build_features"):
        store.load_transformer(store_settings)


def test_a_rebuild_replaces_the_previous_build(store_settings: Settings) -> None:
    """One canonical feature file, like Phase 3's cleaned frame -- not a pile of
    timestamped variants nobody can tell apart."""
    first, transformer = build(rows=100)
    store.write(
        first,
        transformer,
        window=datasets.describe_window(first),
        settings=store_settings,
    )

    second, transformer_2 = build(rows=200)
    store.write(
        second,
        transformer_2,
        window=datasets.describe_window(second),
        settings=store_settings,
    )

    restored, _ = store.read(store_settings)
    assert len(restored) == 200


def test_the_spec_file_is_readable_json(store_settings: Settings) -> None:
    """It is an artefact for people as much as for the next phase."""
    featured, transformer = build(rows=100)
    store.write(
        featured,
        transformer,
        window=datasets.describe_window(featured),
        settings=store_settings,
    )

    _, spec_path, _ = store.paths(store_settings)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))

    assert payload["spec"]["feature_names"] == list(transformer.feature_names)
    assert payload["fit_window"]["rows"] == 100
    assert "rationale" in payload["skew_evidence"]["pm25"]
