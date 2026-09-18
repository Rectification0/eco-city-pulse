"""Missingness analysis (task 3.1, specs §5.3).

Frames are built by hand so the mechanism under test is known by construction:
an assertion that the analyser "found MAR" is only meaningful when the fixture
put a MAR mechanism there on purpose.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from services.datasets import MEASUREMENT_COLUMNS, prepare_frame
from services.quality import missingness
from services.quality.missingness import Mechanism

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_frame(
    rows: int = 400, *, stations: int = 1, seed: int = 7, **overrides: object
) -> pd.DataFrame:
    """A complete hourly frame with plausible values and no missingness."""
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
                    "pm25": float(rng.normal(60, 15)),
                    "pm10": float(rng.normal(120, 25)),
                    "temp": float(rng.normal(22, 4)),
                    "humidity": float(rng.uniform(30, 80)),
                    "traffic_score": float(rng.uniform(20, 80)),
                    "is_anomaly": False,
                }
            )
    frame = prepare_frame(pd.DataFrame(records))
    for column, values in overrides.items():
        frame[column] = values
    return frame


# --- Counts and gaps --------------------------------------------------------


def test_counts_and_percentages_are_reported_per_column() -> None:
    frame = build_frame(rows=100)
    frame.loc[0:9, "pm25"] = np.nan

    result = missingness.analyse_column(frame, "pm25")

    assert result.total == 100
    assert result.missing == 10
    assert result.missing_pct == pytest.approx(10.0)


def test_a_complete_column_has_no_mechanism_to_characterise() -> None:
    result = missingness.analyse_column(build_frame(rows=50), "pm25")

    assert result.mechanism is Mechanism.COMPLETE
    assert result.missing == 0
    assert result.associations == ()


def test_longest_gap_is_the_longest_contiguous_run() -> None:
    frame = build_frame(rows=100)
    frame.loc[5:6, "pm25"] = np.nan  # run of 2
    frame.loc[40:46, "pm25"] = np.nan  # run of 7

    assert missingness.analyse_column(frame, "pm25").longest_gap_rows == 7


def test_gaps_do_not_run_across_stations() -> None:
    """A run is only contiguous within one sensor's series."""
    frame = build_frame(rows=20, stations=2)
    station_a = frame[frame["station"] == frame["station"].iloc[0]].index
    station_b = frame[frame["station"] != frame["station"].iloc[0]].index

    # Three trailing nulls in A, three leading nulls in B: six in a row only if
    # the analyser wrongly treats the frame as one series.
    frame.loc[station_a[-3:], "pm25"] = np.nan
    frame.loc[station_b[:3], "pm25"] = np.nan

    assert missingness.analyse_column(frame, "pm25").longest_gap_rows == 3


def test_longest_null_run_handles_a_trailing_gap() -> None:
    series = pd.Series([1.0, np.nan, 2.0, np.nan, np.nan, np.nan])

    assert missingness.longest_null_run(series) == 3


def test_longest_null_run_of_a_complete_series_is_zero() -> None:
    assert missingness.longest_null_run(pd.Series([1.0, 2.0])) == 0


# --- Mechanism --------------------------------------------------------------


def test_random_missingness_is_reported_as_mcar() -> None:
    """Dropout independent of everything is exactly the MCAR case."""
    frame = build_frame(rows=600, seed=3)
    rng = np.random.default_rng(11)
    frame.loc[rng.choice(frame.index, size=60, replace=False), "pm25"] = np.nan

    assert missingness.analyse_column(frame, "pm25").mechanism is Mechanism.MCAR


def test_missingness_driven_by_another_column_is_reported_as_mar() -> None:
    """The condition that justifies MICE: missingness predictable from data
    that is present."""
    frame = build_frame(rows=800, seed=5)
    # PM10 drops out when temperature is high — a sensor that fails in heat.
    hot = frame["temp"] > frame["temp"].quantile(0.75)
    frame.loc[hot, "pm10"] = np.nan

    result = missingness.analyse_column(frame, "pm10")

    assert result.mechanism is Mechanism.MAR
    assert any(a.significant and a.related_to == "temp" for a in result.associations)
    assert "MICE" in result.justification


def test_a_tail_driven_mechanism_is_detected() -> None:
    """The reason this uses a rank test rather than a correlation.

    Missingness concentrated in the top few percent of a predictor is a huge
    effect that linear correlation barely registers; the earlier point-biserial
    implementation reported MCAR for exactly this shape.
    """
    frame = build_frame(rows=1500, seed=9)
    extreme = frame["pm25"] > frame["pm25"].quantile(0.97)
    rng = np.random.default_rng(4)
    # Saturation in the tail, plus a low background rate everywhere.
    frame.loc[extreme, "pm10"] = np.nan
    frame.loc[rng.choice(frame.index, size=15, replace=False), "pm10"] = np.nan

    result = missingness.analyse_column(frame, "pm10")

    assert result.mechanism is Mechanism.MAR
    strongest = max(result.associations, key=lambda a: a.effect)
    assert strongest.related_to == "pm25"
    assert strongest.direction == "higher"


def test_time_locked_dropout_is_detected_through_the_hour_of_day() -> None:
    """A feed that drops out at the same time every day is not random, and no
    measurement column would reveal it."""
    frame = build_frame(rows=1000, seed=13)
    night = frame["timestamp"].dt.hour.isin([2, 3, 4])
    frame.loc[night, "humidity"] = np.nan

    result = missingness.analyse_column(frame, "humidity")

    assert result.mechanism is Mechanism.MAR
    assert any(a.related_to == "hour_of_day" and a.significant for a in result.associations)


def test_mnar_is_never_inferred_from_the_data() -> None:
    """It is not identifiable from observed values, so no fixture should be
    able to make the analyser claim it."""
    frame = build_frame(rows=800, seed=21)
    # Censored from above: the values that would prove MNAR are the missing ones.
    frame.loc[frame["pm25"] > frame["pm25"].quantile(0.9), "pm25"] = np.nan

    report = missingness.analyse(frame)

    assert all(c.mechanism is not Mechanism.MNAR for c in report.columns)


def test_mnar_can_be_declared_on_domain_grounds() -> None:
    """The only honest route to the label: a stated instrument fact."""
    frame = build_frame(rows=300, seed=2)
    frame.loc[0:19, "pm25"] = np.nan

    report = missingness.analyse(
        frame, declared_mnar={"pm25": "the optical sensor saturates above 500 ug/m3"}
    )
    result = report.by_column()["pm25"]

    assert result.mechanism is Mechanism.MNAR
    assert "saturates" in result.justification
    assert "not identifiable" in result.caveat


def test_declaring_mnar_on_a_complete_column_is_ignored() -> None:
    frame = build_frame(rows=100)

    report = missingness.analyse(frame, declared_mnar={"pm25": "irrelevant"})

    assert report.by_column()["pm25"].mechanism is Mechanism.COMPLETE


def test_every_incomplete_column_carries_the_mnar_caveat() -> None:
    """The report must never imply MNAR has been excluded."""
    frame = build_frame(rows=200, seed=6)
    frame.loc[0:9, "pm25"] = np.nan

    report = missingness.analyse(frame)

    assert missingness.MNAR_CAVEAT in report.caveats
    assert report.by_column()["pm25"].caveat == missingness.MNAR_CAVEAT


def test_a_weak_but_significant_association_is_not_called_mar() -> None:
    """With thousands of rows, p < 0.05 is easy; an effect near zero still
    means nothing, so the effect-size floor has to do real work."""
    association = missingness.Association(
        related_to="temp",
        discrimination=0.502,
        p_value=1e-12,
        sample_size=50_000,
        significant=False,
    )

    assert association.effect < missingness.MIN_EFFECT_SIZE


# --- Report -----------------------------------------------------------------


def test_the_report_covers_every_measurement_column() -> None:
    report = missingness.analyse(build_frame(rows=100))

    assert {c.column for c in report.columns} == set(MEASUREMENT_COLUMNS)


def test_grid_completeness_counts_absent_hours() -> None:
    """A NaN is a row with no reading; an absent hour is a row never written.
    Only the first shows up in a null count."""
    frame = build_frame(rows=100)
    frame = frame.drop(index=frame.index[10:20]).reset_index(drop=True)

    grid = missingness.grid_completeness(frame)

    assert grid.expected_hours == 100
    assert grid.present_hours == 90
    assert grid.absent_hours == 10
    assert grid.completeness_pct == pytest.approx(90.0)


def test_grid_completeness_of_a_full_series_is_total() -> None:
    assert missingness.grid_completeness(build_frame(rows=50)).completeness_pct == 100.0


def test_co_missingness_reveals_a_station_outage() -> None:
    """Four sensors dropping out in the same hours is one outage, not four
    independent quirks — and that is what justifies filling locally."""
    frame = build_frame(rows=200)
    outage = frame.index[50:57]
    for column in ("pm25", "pm10", "temp", "humidity"):
        frame.loc[outage, column] = np.nan

    result = {c.column: c for c in missingness.co_missingness(frame)}

    assert result["pm25"].also_missing["temp"] == pytest.approx(1.0)
    assert result["pm25"].also_missing["traffic_score"] == pytest.approx(0.0)


def test_the_report_serialises_to_json_safe_primitives() -> None:
    frame = build_frame(rows=200, seed=8)
    frame.loc[0:9, "pm25"] = np.nan

    payload = missingness.analyse(frame).as_dict()

    import json

    assert json.loads(json.dumps(payload))["columns"][0]["column"] == "pm25"


def test_an_empty_frame_does_not_raise() -> None:
    empty = prepare_frame(pd.DataFrame(columns=list(build_frame(rows=1).columns)))

    report = missingness.analyse(empty)

    assert all(c.mechanism is Mechanism.COMPLETE for c in report.columns)
    assert report.grid.completeness_pct == 100.0
