"""Prediction intervals (task 9.3, specs §8).

An interval is a claim about how often the truth falls inside it, so the tests
check exactly that: calibrate on one set of errors, then count how often the
resulting bounds contain the outcome.

The fixtures give the model a deliberate bias and a deliberate skew, because
those are the two properties a symmetric ``± q`` interval would silently lose.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.ml import intervals


@pytest.fixture(scope="module")
def residuals() -> tuple[np.ndarray, np.ndarray]:
    """A model that under-predicts, with a long upper tail.

    Pollution error looks like this: a forecast misses a spike by far more than
    it misses a calm hour, so the residual distribution leans one way.
    """
    rng = np.random.default_rng(17)
    predicted = rng.uniform(20, 120, 4000)
    # Lognormal error, shifted so the median sits near +2: biased and skewed.
    actual = predicted + rng.lognormal(1.0, 0.8, 4000) - 1.0
    return actual, predicted


# --- Calibration ------------------------------------------------------------


def test_calibration_reads_the_quantiles_off_the_residuals(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    actual, predicted = residuals

    calibration = intervals.calibrate(actual, predicted)

    assert calibration is not None
    assert calibration.rows == len(actual)
    assert calibration.rmse > 0
    assert len(calibration.quantiles) == len(intervals.QUANTILE_GRID)


def test_too_few_residuals_calibrate_to_nothing() -> None:
    """A tail estimated from a handful of points is noise wearing a number, so
    the caller is told to fall back rather than handed a confident interval."""
    rng = np.random.default_rng(3)
    small = rng.normal(size=intervals.MIN_CALIBRATION_ROWS - 1)

    assert intervals.calibrate(small, np.zeros_like(small)) is None


def test_the_quantiles_are_signed_so_bias_survives(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    """A symmetric interval would centre on the point forecast by assumption.
    These residuals are biased upward, and the calibration says so."""
    actual, predicted = residuals

    calibration = intervals.calibrate(actual, predicted)

    assert calibration.median_bias > 0


def test_the_interval_is_asymmetric_when_the_errors_are(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted)

    low, high = calibration.interval(100.0, 0.9)

    below = 100.0 - low
    above = high - 100.0
    assert above > below  # the long tail is upward, and the interval leans that way


# --- Coverage: the property that makes an interval mean something -----------


@pytest.mark.parametrize("coverage", [0.5, 0.8, 0.9, 0.95])
def test_the_interval_contains_about_the_share_it_claims(
    coverage: float, residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    """Calibrated on these errors, the bounds should contain that fraction of
    them. Anything else means the quantiles are not doing their job."""
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted)

    empirical = intervals.empirical_coverage(actual, predicted, calibration, coverage)

    assert empirical == pytest.approx(coverage, abs=0.02)


def test_a_wider_coverage_gives_a_wider_interval(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted)

    narrow = calibration.interval(80.0, 0.5)
    wide = calibration.interval(80.0, 0.95)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_the_interval_moves_with_the_prediction(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted)

    low_forecast = calibration.interval(30.0, 0.8)
    high_forecast = calibration.interval(130.0, 0.8)

    assert high_forecast[0] - low_forecast[0] == pytest.approx(100.0, abs=1e-6)


def test_the_lower_bound_never_goes_below_zero(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    """A negative PM2.5 concentration is not a measurement. An interval reaching
    below zero would be claiming something impossible rather than uncertain."""
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted)

    # A near-zero forecast plus the lower residual quantile lands below zero.
    low, high = calibration.interval(0.2, 0.99)

    assert calibration.lookup(0.005) < -0.2  # the raw bound really is negative
    assert low == 0.0
    assert high > low


def test_the_bounds_are_ordered(residuals: tuple[np.ndarray, np.ndarray]) -> None:
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted)

    for coverage in (0.5, 0.8, 0.99):
        low, high = calibration.interval(50.0, coverage)
        assert low <= high


# --- The fallback -----------------------------------------------------------


def test_the_normal_fallback_is_symmetric_and_widens_with_coverage() -> None:
    """Weaker than the conformal interval -- it assumes unbiased symmetric
    errors -- which is why the response names the method it used."""
    narrow = intervals.normal_interval(50.0, 10.0, 0.8)
    wide = intervals.normal_interval(50.0, 10.0, 0.95)

    assert 50.0 - narrow[0] == pytest.approx(narrow[1] - 50.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_the_normal_fallback_also_clamps_at_zero() -> None:
    low, _ = intervals.normal_interval(2.0, 10.0, 0.95)

    assert low == 0.0


# --- Round trip and validation ----------------------------------------------


def test_a_calibration_survives_the_artifact(
    residuals: tuple[np.ndarray, np.ndarray]
) -> None:
    """It is stored beside the model, because residual quantiles only describe
    the model that produced them."""
    actual, predicted = residuals
    calibration = intervals.calibrate(actual, predicted, window="2026-01..2026-02")

    restored = intervals.Calibration.from_dict(calibration.as_dict())

    assert restored.rows == calibration.rows
    assert restored.window == "2026-01..2026-02"
    assert restored.interval(60.0, 0.8) == calibration.interval(60.0, 0.8)


def test_an_impossible_coverage_is_refused() -> None:
    for coverage in (0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="coverage must be"):
            intervals.validate_coverage(coverage)


def test_the_payload_says_the_guarantee_is_approximate() -> None:
    """Conformal coverage assumes exchangeability, which a time series does not
    satisfy. Saying so is the difference between a caveat and a claim."""
    assert "exchangeability" in intervals.INTERVAL_CAVEAT
    assert "time series does not satisfy" in intervals.INTERVAL_CAVEAT
