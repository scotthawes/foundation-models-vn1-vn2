"""Order to a quantile of the *summed* demand, not the sum of per-week quantiles.

The earlier quantile attempt was theoretically wrong: `order_up_to` adds three and a half weeks of
per-week quantiles, which is not the quantile of the three-and-a-half-week total. Sums of quantiles
are not quantiles of sums, and the error grows with the horizon.

VN2's cost ratio sets the service level exactly: a lost sale costs 1 EUR, a unit held costs 0.2 EUR,
so ordering to the q = 1/(1+0.2) = 0.833 quantile of total demand minimises expected cost
(the newsvendor fractile). This module builds that level from Chronos's nine per-week quantiles:
sample demand paths (independent across weeks), sum each path over the coverage window, and take the
0.833 quantile of the totals.

Independence is an assumption, and a consequential one — it understates the spread of the sum
relative to comonotone demand, which would order more. It is stated here rather than hidden, and the
service level is swept so the conclusion does not rest on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecaster.chronos_arm import QUANTILES

NEWSVENDOR_SERVICE = 1.0 / (1.0 + 0.2)  # cost ratio from the competition rules
DEFAULT_DRAWS = 256


def sample_paths(quantiles: np.ndarray, draws: int = DEFAULT_DRAWS, seed: int = 0,
                 tail_slope: float = 2.0) -> np.ndarray:
    """Invert the nine quantiles into `draws` demand paths, shape (draws, series, horizon).

    The quantile function is extended below q0.1 to zero (demand cannot be negative) and above q0.9
    by a linear tail, so the top of the distribution is not truncated at 0.9 — which would
    understate the level the newsvendor rule asks for.
    """
    values = np.asarray(quantiles, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"expected (series, quantiles, horizon), got {values.shape}")
    series, n_q, horizon = values.shape
    grid = np.asarray(QUANTILES)

    nodes = values.transpose(1, 0, 2)                       # (n_q, series, horizon)
    top = nodes[-1] + (nodes[-1] - nodes[-2]) * tail_slope  # linear tail above q0.9
    points = np.concatenate(
        [np.zeros((1, series, horizon)), nodes, top[None, :, :]], axis=0
    )  # (n_q + 2, series, horizon)
    positions = np.concatenate([[0.0], grid, [1.0]])

    rng = np.random.default_rng(seed)
    paths = np.empty((draws, series, horizon))
    for draw in range(draws):
        u = rng.random((series, horizon))
        slot = np.clip(np.searchsorted(positions, u, side="right") - 1, 0, len(positions) - 2)
        low = np.take_along_axis(points, slot[None], axis=0)[0]
        high = np.take_along_axis(points, (slot + 1)[None], axis=0)[0]
        x0, x1 = positions[slot], positions[slot + 1]
        paths[draw] = low + (u - x0) / (x1 - x0) * (high - low)
    return paths


def newsvendor_levels(paths: np.ndarray, coverage: float, service: float = NEWSVENDOR_SERVICE):
    """The stock level that minimises expected cost, per series.

    `coverage` weeks of demand are summed per path (fractional weeks take a fraction of the next
    week), then the `service` quantile of those totals is the level to order up to.
    """
    whole = int(np.floor(coverage))
    fraction = coverage - whole
    totals = paths[:, :, :whole].sum(axis=2)
    if fraction:
        totals = totals + fraction * paths[:, :, whole]
    return np.quantile(totals, service, axis=0)


def newsvendor_policy(
    quantiles_fn,
    coverage: float,
    service: float = NEWSVENDOR_SERVICE,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
):
    """A policy with the order-up-to signature, ordering to the newsvendor level."""

    def policy(sales, in_stock_df, master_df, state):
        quantiles = quantiles_fn(sales, in_stock_df, master_df, state)
        level = newsvendor_levels(sample_paths(quantiles, draws=draws, seed=seed), coverage, service)
        net = (
            state[["In Transit W+1", "In Transit W+2", "End Inventory"]].sum(axis=1).to_numpy()
        )
        order = np.clip(np.round(level - net, 0), 0.0, None).astype(int)
        return pd.Series(order, index=sales.index)

    return policy
