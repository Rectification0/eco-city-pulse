"""t-SNE projection (task 6.5, specs §6.2, design §9).

The assertions are about the guard rails rather than about where any point
lands. t-SNE coordinates have no units and no stable meaning between runs of
different data, so a test that pinned a coordinate would be pinning noise.

What *is* worth pinning: the same data gives the same picture, the request
cannot be made to embed a hundred thousand points, and the caveat about what
the axes do not mean ships inside the payload.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.exceptions import InsufficientDataError
from services.eda import manifold
from tests.test_eda_reduction import build_frame


def test_the_projection_is_two_dimensional_and_covers_every_point() -> None:
    frame = build_frame(rows=200)

    result = manifold.project(frame, max_points=200)

    assert result.points == 200
    assert len(result.x) == len(result.y) == 200
    assert len(result.timestamps) == 200


def test_the_same_data_gives_the_same_picture() -> None:
    """A scatter that redrew itself on every refresh would be unreadable, so
    the seed and the PCA initialisation are both fixed."""
    frame = build_frame(rows=150)

    first = manifold.project(frame, max_points=150)
    second = manifold.project(frame, max_points=150)

    assert np.allclose(first.x, second.x)
    assert np.allclose(first.y, second.y)


def test_a_large_window_is_thinned_evenly_across_the_period() -> None:
    """Evenly spaced rather than randomly sampled: the whole period stays
    represented, and the sample is the same on every call."""
    frame = build_frame(rows=1000)

    result = manifold.project(frame, max_points=200)

    assert result.subsampled
    assert result.points <= 200
    assert result.rows_available == 1000
    assert result.timestamps[0] == frame["timestamp"].iloc[0]
    assert result.timestamps[-1] == frame["timestamp"].iloc[-1]


def test_subsampling_leaves_a_small_frame_alone() -> None:
    frame = build_frame(rows=100)

    sampled, was_subsampled = manifold.subsample(frame, 500)

    assert not was_subsampled
    assert len(sampled) == 100


def test_perplexity_is_clamped_below_the_sample_size() -> None:
    """sklearn refuses a perplexity at or above the sample size; clamping turns
    a 500 from deep inside sklearn into a projection that works."""
    frame = build_frame(rows=60)

    result = manifold.project(frame, perplexity=90.0, max_points=60)

    assert result.perplexity < 60
    assert result.points == 60


def test_too_few_rows_is_refused_with_a_domain_error() -> None:
    frame = build_frame(rows=20)

    with pytest.raises(InsufficientDataError, match="at least"):
        manifold.project(frame)


def test_incomplete_rows_are_excluded() -> None:
    frame = build_frame(rows=200)
    frame.loc[0:19, "pm25"] = np.nan

    result = manifold.project(frame, max_points=500)

    assert result.rows_available == 180
    assert result.points == 180


def test_the_payload_says_what_the_axes_do_not_mean() -> None:
    """design §9: local neighbourhoods are meaningful; distances between
    clusters, their sizes and the orientation of the plot are not."""
    frame = build_frame(rows=120)

    payload = manifold.project(frame, max_points=120).as_dict()

    assert "local neighbourhoods" in payload["caveat"]
    assert "never feeds a model" in payload["caveat"]


def test_the_projection_carries_context_for_colouring_the_scatter() -> None:
    frame = build_frame(rows=120, stations=2)

    result = manifold.project(frame, max_points=240)

    assert set(result.stations) <= set(frame["station"])
    assert all(0 <= hour <= 23 for hour in result.hour_of_day)


def test_there_is_no_out_of_sample_transform() -> None:
    """Not an oversight -- the algorithm has none, which is why design §9 keeps
    t-SNE out of every inference path. This asserts the module offers no such
    door."""
    assert not hasattr(manifold, "transform")
    assert not hasattr(manifold.ProjectionResult, "transform")
