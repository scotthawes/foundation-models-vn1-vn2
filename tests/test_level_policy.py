"""The newsvendor level: samples from per-week quantiles, summed over the coverage window.

The property that matters is that the level is the quantile of the *total*, not the sum of weekly
quantiles — and that a degenerate forecast makes it exactly the amount the horizon needs.
"""
from __future__ import annotations

import numpy as np
import pytest

from vn2.level_policy import newsvendor_levels, sample_paths


def constant_quantiles(value: float, series: int = 3, horizon: int = 8) -> np.ndarray:
    return np.full((series, 9, horizon), value)


def test_a_degenerate_forecast_orders_exactly_the_coverage_need():
    paths = sample_paths(constant_quantiles(2.0), draws=32)
    for coverage in (2.0, 3.5):
        assert newsvendor_levels(paths, coverage, service=0.833) == pytest.approx(2.0 * coverage)


def test_a_higher_service_level_never_orders_less():
    rng = np.random.default_rng(0)
    quantiles = np.sort(rng.random((5, 9, 8)) * 10, axis=1)
    paths = sample_paths(quantiles, draws=256)

    low = newsvendor_levels(paths, 3.0, service=0.6)
    high = newsvendor_levels(paths, 3.0, service=0.9)
    assert (high >= low).all()


def test_the_level_is_not_the_sum_of_per_week_quantiles():
    """The defect this module exists to fix: summing q0.833 per week overstates a convex spread."""
    rng = np.random.default_rng(1)
    quantiles = np.sort(rng.random((200, 9, 8)) * 10, axis=1)
    paths = sample_paths(quantiles, draws=2_000)

    newsvendor = newsvendor_levels(paths, 3.0, service=0.833)
    # What the earlier attempt did: take each week's own q0.833 and add them up.
    naive = np.quantile(paths[:, :, :3], 0.833, axis=0).sum(axis=1)

    assert newsvendor.shape == (200,)
    assert not np.allclose(newsvendor, naive), "summed weekly quantiles should differ from the total"
