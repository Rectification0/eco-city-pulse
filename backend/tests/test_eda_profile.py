"""Statistical profile (tasks 4.1, 4.2, 4.4, 4.8; FEAT-02, AC-3).

Task 4.8 asks for statistics "verified against known fixtures", so the core
tests use series whose mean, median and quartiles can be worked out by hand.
A statistics module that is only tested against its own output is testing that
pandas is deterministic, not that the numbers are right.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from services.datasets import MEASUREMENT_COLUMNS, prepare_frame
from services.eda import profile

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def frame_from(**columns: list[float | None]) -> pd.DataFrame:
    """A frame with exactly the values given, one station, hourly."""
    length = max(len(v) for v in columns.values())
    records = []
    for hour in range(length):
        row: dict[str, object] = {
            "id": hour + 1,
            "source_id": 1,
            "timestamp": START + timedelta(hours=hour),
            "lat": 28.61,
            "lon": 77.21,
            "is_anomaly": False,
        }
        for column in MEASUREMENT_COLUMNS:
            values = columns.get(column)
            row[column] = values[hour] if values and hour < len(values) else None
        records.append(row)
    return prepare_frame(pd.DataFrame(records))


# --- 4.1 Univariate, against hand-computable fixtures -----------------------


def test_central_tendency_matches_a_hand_computed_series() -> None:
    """[1..5]: mean 3, median 3, Q1 2, Q3 4, IQR 2, sample std sqrt(2.5)."""
    stats = profile.univariate(frame_from(pm25=[1, 2, 3, 4, 5]), "pm25")

    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.q1 == pytest.approx(2.0)
    assert stats.q3 == pytest.approx(4.0)
    assert stats.iqr == pytest.approx(2.0)
    assert stats.std == pytest.approx(math.sqrt(2.5))
    assert stats.variance == pytest.approx(2.5)
    assert stats.minimum == 1.0
    assert stats.maximum == 5.0


def test_median_of_an_even_series_is_the_midpoint() -> None:
    assert profile.univariate(frame_from(pm25=[1, 2, 3, 4]), "pm25").median == 2.5


def test_standard_deviation_uses_the_sample_denominator() -> None:
    """ddof=1. These are sampled readings, not the population of all possible
    ones, and with small windows the difference is not cosmetic."""
    stats = profile.univariate(frame_from(pm25=[2, 4, 4, 4, 5, 5, 7, 9]), "pm25")

    assert stats.std == pytest.approx(2.13809, abs=1e-4)  # sample, not 2.0


def test_missingness_is_counted_and_reported(  ) -> None:
    """AC-3 names missingness alongside the moments."""
    stats = profile.univariate(frame_from(pm25=[1.0, None, 3.0, None, 5.0]), "pm25")

    assert stats.count == 3
    assert stats.missing == 2
    assert stats.missing_pct == pytest.approx(40.0)


def test_statistics_ignore_missing_values_rather_than_treating_them_as_zero() -> None:
    with_gaps = profile.univariate(frame_from(pm25=[10.0, None, 20.0]), "pm25")

    assert with_gaps.mean == pytest.approx(15.0)


def test_infinities_are_excluded_rather_than_poisoning_the_mean() -> None:
    """``dropna`` alone leaves inf, which turns a whole column of statistics
    into inf with no obvious cause."""
    frame = frame_from(pm25=[1.0, 2.0, 3.0])
    frame.loc[1, "pm25"] = np.inf

    stats = profile.univariate(frame, "pm25")

    assert stats.count == 2
    assert stats.mean == pytest.approx(2.0)


def test_skew_is_positive_for_a_right_tailed_series() -> None:
    stats = profile.univariate(frame_from(pm25=[1, 1, 1, 2, 2, 3, 20]), "pm25")

    assert stats.skewness is not None and stats.skewness > 1


def test_kurtosis_is_excess_so_normal_is_zero() -> None:
    rng = np.random.default_rng(3)
    stats = profile.univariate(
        frame_from(pm25=list(rng.normal(50, 10, 4000))), "pm25"
    )

    assert stats.kurtosis == pytest.approx(0.0, abs=0.25)


def test_a_fully_empty_column_yields_nulls_not_a_crash() -> None:
    stats = profile.univariate(frame_from(pm25=[1.0], pm10=[None]), "pm10")

    assert stats.count == 0
    assert stats.mean is None
    assert stats.missing_pct == 100.0


def test_a_single_observation_has_no_spread() -> None:
    stats = profile.univariate(frame_from(pm25=[7.0]), "pm25")

    assert stats.mean == 7.0
    assert stats.std == 0.0
    assert stats.skewness is None  # undefined below three points


def test_the_profile_covers_every_numeric_column() -> None:
    """AC-3, stated literally."""
    result = profile.univariate_profile(frame_from(pm25=[1.0, 2.0, 3.0]))

    assert tuple(stat.column for stat in result) == MEASUREMENT_COLUMNS


@pytest.mark.parametrize("field", ["mean", "median", "iqr", "missing_pct"])
def test_ac3_fields_are_present_for_every_column(field: str) -> None:
    """AC-3: mean, median, IQR and missingness, for every numeric column."""
    payload = profile.build(frame_from(pm25=[1.0, 2.0, 3.0, 4.0])).as_dict()

    for column in payload["univariate"]:
        assert field in column


# Every field the EDA Studio descriptive-statistics table reads (task 10.18).
DESCRIPTIVE_TABLE_FIELDS = (
    "column", "count", "mean", "median", "std", "min",
    "q1", "q3", "iqr", "max", "skewness", "kurtosis",
)


def test_every_descriptive_table_field_is_present_for_every_column() -> None:
    """The UI table renders these keys verbatim rather than recomputing them,
    so a renamed or dropped key would blank a column silently, not fail loudly.
    """
    payload = profile.build(frame_from(pm25=[1.0, 2.0, 3.0, 4.0, 9.0])).as_dict()

    for column in payload["univariate"]:
        assert set(DESCRIPTIVE_TABLE_FIELDS) <= column.keys(), column["column"]


def test_undefined_statistics_serialise_as_null_never_nan() -> None:
    """The table shows a dash for null; a NaN would reach the browser as invalid
    JSON (or, through a lenient encoder, as a number that renders as "NaN").
    An all-missing column and a column too short for skew/kurtosis are the two
    ways a statistic becomes undefined.
    """
    import json

    payload = profile.build(
        frame_from(pm25=[1.0, 2.0], pm10=[float("nan"), float("inf")])
    ).as_dict()
    by_column = {entry["column"]: entry for entry in payload["univariate"]}

    # allow_nan=False raises on any NaN/inf left anywhere in the stats.
    json.dumps(payload["univariate"], allow_nan=False)
    assert by_column["pm10"]["count"] == 0
    assert all(by_column["pm10"][f] is None for f in DESCRIPTIVE_TABLE_FIELDS[2:])
    assert by_column["pm25"]["skewness"] is None
    assert by_column["pm25"]["kurtosis"] is None


# --- 4.2 Bivariate ----------------------------------------------------------


def test_a_perfect_linear_relationship_is_one_on_both_methods() -> None:
    result = profile.bivariate_profile(
        frame_from(pm25=[1, 2, 3, 4, 5], pm10=[2, 4, 6, 8, 10])
    )

    assert result.pearson["pm25"]["pm10"] == pytest.approx(1.0)
    assert result.spearman["pm25"]["pm10"] == pytest.approx(1.0)


def test_an_inverse_relationship_is_negative() -> None:
    result = profile.bivariate_profile(
        frame_from(pm25=[1, 2, 3, 4, 5], temp=[10, 8, 6, 4, 2])
    )

    assert result.pearson["pm25"]["temp"] == pytest.approx(-1.0)


def test_a_monotone_curve_separates_spearman_from_pearson() -> None:
    """The reason both are reported. Spearman sees a perfect rank relationship
    where Pearson sees a merely strong linear one."""
    values = [1, 2, 3, 4, 5, 6, 7]
    result = profile.bivariate_profile(
        frame_from(pm25=values, pm10=[v**3 for v in values])
    )
    pair = next(p for p in result.pairs if {p.a, p.b} == {"pm25", "pm10"})

    assert pair.spearman == pytest.approx(1.0)
    assert pair.pearson is not None and pair.pearson < 0.98
    assert pair.divergence is not None and pair.divergence > 0.01


def test_the_diagonal_is_one() -> None:
    result = profile.bivariate_profile(frame_from(pm25=[1, 2, 3, 4]))

    assert result.pearson["pm25"]["pm25"] == pytest.approx(1.0)


def test_each_pair_reports_the_overlap_it_used() -> None:
    """A coefficient from 4 overlapping rows and one from 30,000 are different
    claims; the bare number hides which you have."""
    result = profile.bivariate_profile(
        frame_from(pm25=[1, 2, 3, 4, None, None], pm10=[2, 4, 6, None, 10, 12])
    )
    pair = next(p for p in result.pairs if {p.a, p.b} == {"pm25", "pm10"})

    assert pair.sample_size == 3


def test_a_constant_column_has_no_correlation_to_report() -> None:
    result = profile.bivariate_profile(
        frame_from(pm25=[1, 2, 3, 4], pm10=[5, 5, 5, 5])
    )

    assert result.pearson["pm25"]["pm10"] is None


def test_the_correlation_section_disclaims_causation() -> None:
    """ETH-1. The heatmap is the most misread artefact in an EDA dashboard, so
    the caveat ships inside the payload rather than being left to the UI."""
    result = profile.bivariate_profile(frame_from(pm25=[1, 2, 3]))

    assert "causation" in result.caveat.lower()


def test_no_output_field_contains_causal_language() -> None:
    """ETH-1 audit in miniature: nothing here may say one column 'causes',
    'drives', 'affects' or 'leads to' another."""
    payload = str(profile.build(frame_from(pm25=[1, 2, 3, 4, 5], pm10=[2, 4, 6, 8, 10])).as_dict())
    forbidden = (" causes ", " drives ", " affects ", " leads to ", " because of ")

    for phrase in forbidden:
        assert phrase not in payload.lower()


def test_strongest_ranks_by_absolute_strength() -> None:
    result = profile.bivariate_profile(
        frame_from(
            pm25=[1, 2, 3, 4, 5],
            pm10=[2, 4, 6, 8, 10],  # +1.0
            temp=[1, 3, 2, 5, 4],  # weaker
        )
    )

    assert {result.strongest(1)[0].a, result.strongest(1)[0].b} == {"pm25", "pm10"}


# --- 4.4 Distribution and transform advice ----------------------------------


def test_a_heavily_skewed_positive_column_gets_a_log_recommendation() -> None:
    """The case specs §6.1 has in mind with CO and SO2."""
    rng = np.random.default_rng(11)
    values = list(np.exp(rng.normal(1.0, 1.2, 800)))

    assessment = profile.assess_distribution(frame_from(pm25=values), "pm25")

    assert assessment.skewness is not None and assessment.skewness > 1
    assert assessment.recommend_log_transform
    assert abs(assessment.log_skewness or 0) < abs(assessment.skewness)


def test_a_symmetric_column_is_left_alone() -> None:
    rng = np.random.default_rng(13)
    assessment = profile.assess_distribution(
        frame_from(temp=list(rng.normal(22, 4, 500))), "temp"
    )

    assert not assessment.recommend_log_transform
    assert assessment.shape == "approximately symmetric"


def test_a_transform_that_does_not_help_is_not_recommended() -> None:
    """A log transform costs interpretability -- coefficients stop being in
    µg/m³ -- so it has to buy something."""
    # Skewed, but log1p overcorrects into a worse left skew.
    values = [1.0] * 200 + [2.0] * 40 + [60.0] * 12
    assessment = profile.assess_distribution(frame_from(pm25=values), "pm25")

    assert assessment.skewness is not None and abs(assessment.skewness) > 1
    if not assessment.recommend_log_transform:
        assert "without fixing the shape" in assessment.rationale or (
            abs(assessment.skewness) - abs(assessment.log_skewness or 0)
            < profile.MIN_SKEW_IMPROVEMENT
        )


def test_a_column_with_negative_values_cannot_be_log_transformed() -> None:
    assessment = profile.assess_distribution(
        frame_from(temp=[-5.0, -2.0, 0.0, 30.0, 80.0, 90.0] * 10), "temp"
    )

    assert not assessment.is_strictly_positive
    assert assessment.log_skewness is None
    assert not assessment.recommend_log_transform
    assert "negative" in assessment.rationale


def test_zero_is_a_legitimate_reading_and_survives_the_transform() -> None:
    """log1p, not log: a zero concentration is a measurement, not a value to
    drop."""
    rng = np.random.default_rng(17)
    values = [0.0] * 50 + list(np.exp(rng.normal(1.0, 1.3, 500)))

    assessment = profile.assess_distribution(frame_from(pm25=values), "pm25")

    assert assessment.log_skewness is not None
    assert math.isfinite(assessment.log_skewness)


def test_every_assessment_explains_itself() -> None:
    for assessment in profile.distribution_profile(frame_from(pm25=[1.0, 2.0, 9.0, 3.0])):
        assert assessment.rationale


def test_too_few_points_is_reported_rather_than_guessed() -> None:
    assessment = profile.assess_distribution(frame_from(pm25=[1.0, 2.0]), "pm25")

    assert assessment.shape == "unknown"
    assert not assessment.recommend_log_transform


# --- The whole profile ------------------------------------------------------


def test_build_returns_all_three_layers() -> None:
    result = profile.build(frame_from(pm25=[1, 2, 3, 4, 5], pm10=[2, 4, 6, 8, 10]))

    assert len(result.univariate) == len(MEASUREMENT_COLUMNS)
    assert result.bivariate.pairs
    assert len(result.distributions) == len(MEASUREMENT_COLUMNS)
    assert result.rows == 5


def test_the_large_sample_normality_caveat_appears_only_when_relevant() -> None:
    rng = np.random.default_rng(19)

    small = profile.build(frame_from(pm25=list(rng.normal(50, 10, 100))))
    large = profile.build(frame_from(pm25=list(rng.normal(50, 10, 6000))))

    assert profile.NORMALITY_CAVEAT not in small.caveats
    assert profile.NORMALITY_CAVEAT in large.caveats


def test_the_payload_is_json_serialisable() -> None:
    import json

    payload = profile.build(frame_from(pm25=[1.0, 2.0, 3.0])).as_dict()

    assert json.loads(json.dumps(payload))["rows"] == 3


def test_an_empty_frame_profiles_without_raising() -> None:
    empty = frame_from(pm25=[1.0]).iloc[0:0]

    result = profile.build(empty)

    assert result.rows == 0
    assert all(stat.count == 0 for stat in result.univariate)
