"""Visual EDA — scatter, grouped, pair plot, Andrews curves (task 12.11).

Every statistic is checked against a fixture whose answer can be worked out by
hand, because a chart is only as trustworthy as the number printed under it and
"the plot looks right" is not a test. The rest pins the VIZ-6 rules that are
easy to lose in a refactor: the sample is deterministic, the coefficients come
from every row, flagged rows stay on the chart, and the scope a request asked
for is the scope it reads.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.exceptions import InsufficientDataError, SchemaValidationError
from services.datasets import MEASUREMENT_COLUMNS, prepare_frame
from services.eda import bands, cache, profile, service, visual

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build(
    rows: int,
    *,
    start: datetime = START,
    seed: int = 7,
    stations: int = 1,
    flagged: tuple[int, ...] = (),
    **overrides: list[float | None],
) -> pd.DataFrame:
    """Hourly readings with plausible joint structure; columns overridable."""
    rng = np.random.default_rng(seed)
    traffic = rng.uniform(10, 90, rows)
    data = {
        "pm25": 20 + 0.8 * traffic + rng.normal(0, 8, rows),
        "pm10": 45 + 1.4 * traffic + rng.normal(0, 12, rows),
        "temp": rng.normal(24, 4, rows),
        "humidity": rng.uniform(35, 85, rows),
        "traffic_score": traffic,
    }
    data.update({name: np.asarray(values, dtype="float64") for name, values in overrides.items()})

    records = []
    for index in range(rows):
        station = index % stations
        records.append(
            {
                "id": index + 1,
                "source_id": 1,
                "timestamp": start + timedelta(hours=index // stations),
                "lat": 28.6 + station * 0.01,
                "lon": 77.2,
                "is_anomaly": index in flagged,
                **{column: data[column][index] for column in MEASUREMENT_COLUMNS},
            }
        )
    return prepare_frame(pd.DataFrame(records))


def at(*timestamps: datetime) -> pd.DataFrame:
    """One row per timestamp, for checking calendar labels."""
    return prepare_frame(
        pd.DataFrame(
            [
                {
                    "id": index,
                    "source_id": 1,
                    "timestamp": moment,
                    "lat": 28.6,
                    "lon": 77.2,
                    "is_anomaly": False,
                    **{column: 50.0 for column in MEASUREMENT_COLUMNS},
                }
                for index, moment in enumerate(timestamps)
            ]
        )
    )


# --- 12.2 Groupings -----------------------------------------------------------


def test_hour_of_day_is_read_on_the_ist_clock_not_utc() -> None:
    """03:30 UTC is 09:00 IST. Labelled in UTC, every diurnal peak in the
    grouped boxplot would sit five and a half hours early."""
    frame = at(datetime(2026, 1, 5, 3, 30, tzinfo=timezone.utc))

    assert list(visual.grouping_series(frame, "hour_of_day")) == ["09:00"]


def test_time_of_day_buckets_the_local_hour_into_four_six_hour_bands() -> None:
    local_hours = [0, 5, 6, 11, 12, 17, 18, 23]
    # Local hour h is UTC h − 5:30.
    moments = [
        datetime(2026, 1, 5, tzinfo=timezone.utc) + timedelta(hours=h, minutes=-330)
        for h in local_hours
    ]

    labels = list(visual.grouping_series(at(*moments), "time_of_day"))

    assert labels == [
        "night", "night", "morning", "morning",
        "afternoon", "afternoon", "evening", "evening",
    ]


def test_the_ist_offset_moves_the_calendar_day_as_well_as_the_hour() -> None:
    """Friday 20:00 UTC is already Saturday 01:30 IST — a weekend hour. The
    weekday, weekend and month labels all follow the local date."""
    frame = at(datetime(2026, 1, 2, 20, 0, tzinfo=timezone.utc))

    assert list(visual.grouping_series(frame, "day_of_week")) == ["Sat"]
    assert list(visual.grouping_series(frame, "is_weekend")) == ["weekend"]
    assert list(visual.grouping_series(frame, "month")) == ["Jan"]


def test_groupings_are_ordered_like_the_axis_they_label() -> None:
    """Mon before Tue, not alphabetically; 09:00 before 10:00."""
    frame = build(24 * 8)

    days = visual.grouping_series(frame, "day_of_week")
    hours = visual.grouping_series(frame, "hour_of_day")

    assert list(days.cat.categories) == list(visual.DAY_NAMES)
    assert list(hours.cat.categories)[:3] == ["00:00", "01:00", "02:00"]


def test_a_row_without_pm25_has_no_band_group() -> None:
    frame = build(3, pm25=[10.0, None, 200.0])

    labels = visual.grouping_series(frame, "pm25_band")

    assert labels.iloc[0] == "Good"
    assert pd.isna(labels.iloc[1])
    assert labels.iloc[2] == "Very poor"


def test_an_unknown_grouping_is_refused_by_name() -> None:
    with pytest.raises(SchemaValidationError, match="district"):
        visual.grouping_series(build(5), "district")


# --- 12.3 Scatter -------------------------------------------------------------


def test_the_ols_line_recovers_an_exact_linear_relationship() -> None:
    """y = 2x + 1 exactly: slope 2, intercept 1, r = 1, r² = 1."""
    xs = [float(value) for value in range(1, 21)]
    frame = build(20, traffic_score=xs, pm25=[2 * x + 1 for x in xs])

    result = visual.scatter(frame, "traffic_score", "pm25")

    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(1.0)
    assert result.r_squared == pytest.approx(1.0)
    assert result.pearson == pytest.approx(1.0)
    assert result.line_x == pytest.approx((1.0, 20.0))
    assert result.line_y == pytest.approx((3.0, 41.0))


def test_the_scatter_r_is_the_heatmap_cell_for_the_same_pair() -> None:
    """One number, one definition: a scatter that disagreed with the heatmap
    beside it in the third decimal would be the first thing a reader noticed.
    Gaps included, because pairwise-complete handling is where two
    implementations would drift."""
    pm25 = list(build(400, seed=3)["pm25"])
    for index in range(0, 400, 9):
        pm25[index] = None
    frame = build(400, seed=3, pm25=pm25)

    heatmap = profile.build(frame).bivariate
    result = visual.scatter(frame, "traffic_score", "pm25")

    assert result.pearson == heatmap.pearson["traffic_score"]["pm25"]
    assert result.spearman == heatmap.spearman["traffic_score"]["pm25"]
    assert result.rows_used == 400 - len(range(0, 400, 9))


def test_r_squared_from_the_residuals_equals_pearson_squared() -> None:
    """Computed independently, so agreement is evidence rather than identity."""
    frame = build(500, seed=11)

    result = visual.scatter(frame, "traffic_score", "pm25")
    r = np.corrcoef(frame["traffic_score"], frame["pm25"])[0, 1]

    assert result.r_squared == pytest.approx(r**2, abs=1e-12)


def test_a_large_scatter_is_thinned_deterministically_and_says_so() -> None:
    """VIZ-6: at most 2,000 marks, the same marks on every call, and the
    payload admits it is a sample."""
    frame = build(5000)

    first = visual.scatter(frame, "traffic_score", "pm25")
    second = visual.scatter(frame, "traffic_score", "pm25")

    assert first.points_returned <= visual.SCATTER_MAX_POINTS
    assert first.sampled
    assert first.rows_used == 5000
    assert first.x == second.x and first.y == second.y
    # Even stride: the whole window stays represented, ends included.
    assert first.x[0] == frame["traffic_score"].iloc[0]
    assert first.x[-1] == frame["traffic_score"].iloc[-1]


def test_the_fit_uses_every_row_not_the_sample() -> None:
    """The line spans the full x range even when the extreme rows were not
    sampled, and the coefficients match a fit on all rows."""
    frame = build(5000, seed=5)

    result = visual.scatter(frame, "traffic_score", "pm25")
    slope, intercept = np.polyfit(frame["traffic_score"], frame["pm25"], 1)

    assert result.slope == pytest.approx(slope)
    assert result.intercept == pytest.approx(intercept)
    assert result.line_x == pytest.approx(
        (frame["traffic_score"].min(), frame["traffic_score"].max())
    )


def test_flagged_rows_stay_on_the_scatter() -> None:
    """AC-5: drawn and marked, never dropped to tidy the cloud."""
    flagged = (3, 17, 40)
    frame = build(60, flagged=flagged)

    result = visual.scatter(frame, "traffic_score", "pm25")

    assert result.anomalies_in_rows == 3
    assert result.anomalies_in_points == 3
    assert sum(result.is_anomaly) == 3
    assert len(result.x) == 60


def test_colouring_a_scatter_labels_each_point_with_its_group() -> None:
    frame = build(48)

    result = visual.scatter(frame, "traffic_score", "pm25", color_by="time_of_day")

    assert result.categories == ("night", "morning", "afternoon", "evening")
    assert len(result.groups) == result.points_returned
    assert set(result.groups) == set(result.categories)


def test_a_scatter_of_a_column_against_itself_is_refused() -> None:
    with pytest.raises(SchemaValidationError):
        visual.scatter(build(10), "pm25", "pm25")


def test_a_scatter_with_too_few_rows_is_a_data_error_not_a_crash() -> None:
    with pytest.raises(InsufficientDataError):
        visual.scatter(build(2), "traffic_score", "pm25")


def test_every_chart_disclaims_causation() -> None:
    """ETH-1 travels in each payload, first, before the chart's own caveat."""
    frame = build(200)

    for result in (
        visual.scatter(frame, "traffic_score", "pm25"),
        visual.grouped(frame, "pm25", "hour_of_day"),
        visual.pair_plot(frame),
        visual.andrews(frame),
    ):
        assert "causation" in result.caveats[0].lower()
        assert len(result.caveats) >= 2


# --- 12.4 Grouped bars and boxes ---------------------------------------------


def test_the_interval_is_t_based_on_a_hand_computed_fixture() -> None:
    """[2, 4, 4, 4, 5, 5, 7, 9]: mean 5, s = √(32/7), half-width
    t₀.₉₇₅,₇ · s / √8 ≈ 2.3646 · 2.1381 / 2.8284 ≈ 1.7875."""
    mean, sd, low, high = visual.mean_interval(np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float))

    assert mean == 5.0
    assert sd == pytest.approx(math.sqrt(32 / 7))
    assert high - mean == pytest.approx(1.78748, abs=1e-4)
    assert mean - low == pytest.approx(high - mean)


def test_the_t_interval_is_wider_than_the_normal_one_for_a_small_group() -> None:
    """The reason for t: with n = 5 the 1.96 interval understates uncertainty."""
    values = np.array([10.0, 12.0, 9.0, 14.0, 11.0])

    mean, sd, _, high = visual.mean_interval(values)

    assert high - mean > 1.96 * sd / math.sqrt(5)
    assert high - mean == pytest.approx(stats.t.ppf(0.975, 4) * sd / math.sqrt(5))


def test_a_single_value_has_no_spread_rather_than_a_zero_one() -> None:
    mean, sd, low, high = visual.mean_interval(np.array([42.0]))

    assert (mean, sd, low, high) == (42.0, None, None, None)


def test_cells_under_thirty_rows_are_marked_thin() -> None:
    """30 is the threshold: 29 is thin, 30 is not."""
    frame = build(59)
    # Move the last 30 rows to a second station: groups of 29 and 30.
    frame.iloc[29:, frame.columns.get_loc("lat")] = 28.7
    frame = prepare_frame(frame.drop(columns="station"))

    result = visual.grouped(frame, "pm25", "station")

    assert [(bar.n, bar.thin) for bar in result.bars] == [(29, True), (30, False)]


def test_quartiles_and_fences_on_a_hand_computed_fixture() -> None:
    """1…9 and 100. Linear quartiles: Q1 3.25, median 5.5, Q3 7.75, so the
    upper bound is 7.75 + 1.5 · 4.5 = 14.5. The upper whisker stops at the
    last value inside it (9), and 100 is the one value past it."""
    frame = build(10, pm25=[1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

    result = visual.grouped(frame, "pm25", "is_weekend")
    (box,) = result.boxes

    assert (box.q1, box.median, box.q3) == (3.25, 5.5, 7.75)
    assert box.lower_fence == 1.0
    assert box.upper_fence == 9.0
    assert box.whisker_outliers == 1
    assert box.outliers == (100.0,)


def test_only_the_most_extreme_outliers_are_shipped_with_the_full_count() -> None:
    """60 values past the whisker: all 60 counted, the 50 furthest sent."""
    values = [10.0] * 400 + [1000.0 + i for i in range(60)]
    # One group: a constant PM2.5 keeps every row in the "Good" band.
    frame = build(len(values), temp=values, pm25=[10.0] * len(values))

    (box,) = visual.grouped(frame, "temp", "pm25_band").boxes

    assert box.whisker_outliers == 60
    assert len(box.outliers) == visual.BOX_MAX_OUTLIERS
    assert min(box.outliers) == 1010.0
    assert max(box.outliers) == 1059.0


def test_a_split_produces_one_bar_per_group_and_split_cell() -> None:
    frame = build(24 * 14)

    result = visual.grouped(frame, "pm25", "time_of_day", split_by="is_weekend")

    assert result.split_categories == ("weekday", "weekend")
    assert len(result.bars) == 4 * 2
    assert sum(bar.n for bar in result.bars) == result.rows_used
    # Boxes stay by group only.
    assert [box.group for box in result.boxes] == list(result.categories)


def test_splitting_by_the_grouping_itself_is_refused() -> None:
    with pytest.raises(SchemaValidationError):
        visual.grouped(build(10), "pm25", "month", split_by="month")


def test_boxes_count_flagged_rows_separately_from_whisker_outliers() -> None:
    """The two tests differ (design §12.1), so the payload carries both."""
    values = [10.0] * 40 + [500.0]
    frame = build(len(values), pm25=values, flagged=(0, 1))

    result = visual.grouped(frame, "pm25", "is_weekend")
    (box,) = result.boxes

    assert box.anomalies_flagged == 2
    assert box.whisker_outliers == 1
    assert result.anomalies_in_rows == 2


def test_grouping_by_band_skips_rows_without_pm25_but_keeps_them_elsewhere() -> None:
    frame = build(6, pm25=[10.0, None, 50.0, 70.0, 100.0, 300.0])

    by_band = visual.grouped(frame, "temp", "pm25_band")
    by_day = visual.grouped(frame, "temp", "is_weekend")

    assert by_band.rows_used == 5
    assert by_day.rows_used == 6
    assert by_band.categories == ("Good", "Satisfactory", "Moderate", "Poor", "Severe")


# --- 12.5 Pair plot -----------------------------------------------------------


def test_the_pair_plot_samples_complete_rows_only() -> None:
    """One row is one mark in every panel, so a row missing humidity would
    appear in six panels and vanish from four."""
    humidity = list(build(3000)["humidity"])
    humidity[10] = None
    frame = build(3000, humidity=humidity)

    result = visual.pair_plot(frame)

    assert result.rows_used == 2999
    assert result.points_returned <= visual.PAIRPLOT_MAX_POINTS
    assert result.sampled
    assert all(len(values) == result.points_returned for values in result.values.values())
    assert set(result.values) == set(MEASUREMENT_COLUMNS)


def test_each_pair_plot_point_carries_its_band() -> None:
    frame = build(100)

    result = visual.pair_plot(frame)

    assert list(result.bands) == [
        bands.band_for(value).label for value in result.values["pm25"]
    ]
    assert result.color_by == "pm25_band"
    assert result.groups == result.bands


def test_the_pair_plot_coefficients_are_the_heatmap_matrix() -> None:
    """The sample is a sample; the coefficients are not (design §12.1)."""
    frame = build(3000, seed=13)

    result = visual.pair_plot(frame)

    assert result.pearson == profile.build(frame).bivariate.pearson


def test_flagged_rows_stay_in_the_pair_plot() -> None:
    frame = build(80, flagged=(5, 50))

    result = visual.pair_plot(frame)

    assert result.anomalies_in_points == 2
    assert sum(result.is_anomaly) == 2


# --- 12.6 Andrews curves ------------------------------------------------------


def test_the_andrews_function_matches_its_definition() -> None:
    """At t = π/2: x₁/√2 + x₂·1 + x₃·0 + x₄·sin π + x₅·cos π = 1/√2 + 2 − 5."""
    t = np.array([math.pi / 2])

    (value,) = visual.andrews_curves(np.array([[1, 2, 3, 4, 5]]), t)[0]

    assert value == pytest.approx(1 / math.sqrt(2) + 2 - 5)


def test_the_t_grid_is_128_points_over_minus_pi_to_pi() -> None:
    result = visual.andrews(build(100))

    assert len(result.t) == 128
    assert result.t[0] == pytest.approx(-math.pi)
    assert result.t[-1] == pytest.approx(math.pi)
    assert all(len(item.mean_curve) == 128 for item in result.classes)


def test_the_mean_curve_is_the_curve_of_the_mean_row() -> None:
    """f is linear in the row, so the class mean curve can be computed exactly
    from every row — and equals the average of all its curves."""
    frame = build(48, seed=21)  # 12 rows per time-of-day class, all drawn

    result = visual.andrews(frame)

    for item in result.classes:
        assert not item.sampled
        averaged = np.mean(np.array(item.curves), axis=0)
        assert np.allclose(item.mean_curve, averaged)


def test_the_mean_curve_uses_every_row_even_when_the_curves_are_sampled() -> None:
    frame = build(24 * 40, seed=8)
    columns = list(visual.ANDREWS_COLUMNS)

    result = visual.andrews(frame)

    values = frame[columns]
    z = (values - values.mean()) / values.std(ddof=0)
    classes = visual.grouping_series(frame, "time_of_day")
    for item in result.classes:
        assert item.sampled and len(item.curves) == visual.ANDREWS_CURVES_PER_CLASS
        expected = visual.andrews_curves(
            z[classes == item.label].mean().to_numpy(), np.array(result.t)
        )[0]
        assert np.allclose(item.mean_curve, expected)


def test_a_rare_class_keeps_its_curves_instead_of_being_thinned_away() -> None:
    """Sampled within class: a Severe hour among hundreds of Moderate ones is
    still drawn, which a proportional sample would not do."""
    pm25 = [75.0] * 600 + [400.0] * 4
    frame = build(len(pm25), pm25=pm25)

    result = visual.andrews(frame, class_by="pm25_band")
    curves = {item.label: len(item.curves) for item in result.classes}

    assert curves == {"Moderate": 60, "Severe": 4}


def test_andrews_column_order_is_fixed_and_returned() -> None:
    """Order changes the picture, so it is part of the payload."""
    result = visual.andrews(build(50))

    assert result.columns == ("pm25", "pm10", "traffic_score", "temp", "humidity")
    assert result.as_dict()["columns"] == list(visual.ANDREWS_COLUMNS)


def test_andrews_standardises_so_no_column_dominates_by_its_units() -> None:
    frame = build(200)

    result = visual.andrews(frame)

    for column in visual.ANDREWS_COLUMNS:
        assert result.means[column] == pytest.approx(frame[column].mean())
        assert result.stds[column] == pytest.approx(frame[column].std(ddof=0))


def test_a_constant_column_does_not_turn_the_curves_into_nan() -> None:
    frame = build(40, humidity=[60.0] * 40)

    result = visual.andrews(frame)

    assert all(
        value is not None and math.isfinite(value)
        for item in result.classes
        for value in item.mean_curve
    )


def test_andrews_refuses_classes_that_would_be_unreadable_as_curves() -> None:
    with pytest.raises(SchemaValidationError):
        visual.andrews(build(50), class_by="hour_of_day")


def test_flagged_curves_are_marked_not_dropped() -> None:
    frame = build(40, flagged=(0, 4))

    result = visual.andrews(frame)

    assert result.anomalies_in_rows == 2
    assert sum(sum(item.is_anomaly) for item in result.classes) == 2


# --- 12.7 Service: scope first, then everything else -------------------------


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """The service with its three database touch points replaced by recorders,
    so the order and arguments can be asserted without PostgreSQL."""
    calls: dict[str, list] = {"resolve": [], "version": [], "load": []}
    cache.PROFILE_CACHE.clear()

    def resolve(session, source_ids, *, settings=None):
        calls["resolve"].append(source_ids)
        return (7, 9) if not source_ids else tuple(source_ids)

    def version(session, *, source_ids=None, start=None, end=None):
        calls["version"].append(source_ids)
        return cache.DatasetVersion(
            rows=len(source_ids or ()), max_id=1, latest_timestamp=None, flagged_rows=0
        )

    def load(session, *, source_ids=None, start=None, end=None, limit=None):
        calls["load"].append(source_ids)
        return build(120)

    monkeypatch.setattr(service.datasets, "resolve_source_ids", resolve)
    monkeypatch.setattr(service.cache, "dataset_version", version)
    monkeypatch.setattr(service.datasets, "load_observations", load)
    yield calls
    cache.PROFILE_CACHE.clear()


@pytest.mark.parametrize(
    "call",
    [
        lambda: service.scatter_plot(None, color_by="month"),
        lambda: service.grouped_summary(None, split_by="is_weekend"),
        lambda: service.pair_plot(None),
        lambda: service.andrews_curves(None),
    ],
    ids=["scatter", "grouped", "pairplot", "andrews"],
)
def test_every_chart_reads_the_resolved_scope_everywhere(recorded, call) -> None:
    """VIZ-6 and the CLAUDE.md scope rule: the ids the configured scope
    resolves to reach the fingerprint, the query and the window alike. A chart
    that queried with the raw ``None`` would pool every provenance (ETH-1)."""
    result = call()

    assert recorded["resolve"] == [None]
    assert recorded["version"] == [(7, 9)]
    assert recorded["load"] == [(7, 9)]
    assert result.window.source_ids == (7, 9)
    assert result.as_dict()["window"]["source_ids"] == [7, 9]


def test_a_repeated_chart_request_is_served_from_the_cache(recorded) -> None:
    first = service.scatter_plot(None)
    second = service.scatter_plot(None)

    assert (first.cached, second.cached) == (False, True)
    assert len(recorded["load"]) == 1


def test_two_scopes_never_share_a_cached_chart(recorded) -> None:
    """The scope is in the key, so asking for source 3 after the default scope
    computes afresh rather than serving the other provenance's answer."""
    service.scatter_plot(None)
    other = service.scatter_plot(None, source_ids=(3,))

    assert other.cached is False
    assert recorded["load"] == [(7, 9), (3,)]


def test_different_chart_parameters_never_share_a_cached_answer(recorded) -> None:
    service.grouped_summary(None, group_by="hour_of_day")
    other = service.grouped_summary(None, group_by="month")

    assert other.cached is False
    assert other.chart.group_by == "month"


# --- 12.8 Request validation (SEC-1) -----------------------------------------


def test_chart_requests_reject_unknown_names_before_any_query_runs() -> None:
    from pydantic import ValidationError

    from api.routes.eda import (
        AndrewsRequest,
        GroupedRequest,
        PairPlotRequest,
        ScatterRequest,
    )

    for bad in (
        lambda: ScatterRequest(x="co"),
        lambda: ScatterRequest(x="pm25", y="pm25"),
        lambda: ScatterRequest(color_by="district"),
        lambda: GroupedRequest(measure="so2"),
        lambda: GroupedRequest(group_by="weekday"),
        lambda: GroupedRequest(group_by="month", split_by="month"),
        lambda: PairPlotRequest(color_by="colour"),
        lambda: AndrewsRequest(class_by="hour_of_day"),
    ):
        with pytest.raises(ValidationError):
            bad()


def test_chart_requests_have_working_defaults() -> None:
    """Every body is optional: the EDA Studio's first paint sends none."""
    from api.routes.eda import (
        AndrewsRequest,
        GroupedRequest,
        PairPlotRequest,
        ScatterRequest,
    )

    assert (ScatterRequest().x, ScatterRequest().y) == ("traffic_score", "pm25")
    assert GroupedRequest().group_by == "hour_of_day"
    assert PairPlotRequest().color_by == "pm25_band"
    assert AndrewsRequest().class_by == "time_of_day"


def test_every_chart_payload_validates_against_its_response_model() -> None:
    """The dataclasses and the Pydantic models describe one contract; a field
    renamed on one side fails here rather than as a 500 in the browser."""
    from api.routes.eda import (
        AndrewsResponse,
        GroupedResponse,
        PairPlotResponse,
        ScatterResponse,
    )

    frame = build(300, flagged=(1, 2))
    provenance = {"cached": False, "dataset_version": {}, "window": {}}

    for model, result in (
        (ScatterResponse, visual.scatter(frame, "traffic_score", "pm25", color_by="month")),
        (GroupedResponse, visual.grouped(frame, "pm25", "hour_of_day", split_by="is_weekend")),
        (PairPlotResponse, visual.pair_plot(frame)),
        (AndrewsResponse, visual.andrews(frame)),
    ):
        model.model_validate({**result.as_dict(), **provenance})
