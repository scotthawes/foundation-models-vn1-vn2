"""Chronos-2 arm for VN2: the newer checkpoint, with optional cross-learning.

The Bolt arm lost VN2 because the ordering policy wants a per-series mean and a median is biased
low, concentrated in the high-volume series. Chronos-2 offers two things that address that:

- `cross_learning` forecasts all 599 series in one joint pass, so information is shared across
  series instead of each being forecast alone — the tabular property inside a sequence model;
- 21 quantiles rather than 9, so the level can be set nearer the newsvendor fractile the cost ratio
  implies (1 EUR lost against 0.2 EUR held gives 0.833).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import demand_estimate, week_offset
from ..features import HORIZONS
from ..play import ForecasterFn
from .chronos_arm import _load, causal_fill

DEFAULT_CONTEXT_WEEKS = 152


def chronos2(
    context_weeks: int = DEFAULT_CONTEXT_WEEKS,
    quantile: float = 0.5,
    cross_learning: bool = False,
    batch_size: int = 256,
    normalise: bool = False,
) -> ForecasterFn:
    """Chronos-2, optionally forecasting the ratio to each series' own recent level.

    `normalise=True` divides every series by its trailing 13-week mean, forecasts that ratio, and
    multiplies back. The motivation is measured: on the played weeks the raw arm forecast 0.80 of
    realised demand in the top volume decile, while a 13-week mean was at 0.97 — the model's error is
    a per-series level bias, and a scale-free target gives it a level the data supplies rather than
    one it has to infer.
    """
    pipeline = None  # loaded on first use
    LEVEL_WEEKS = 13
    LEVEL_FLOOR = 0.5

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        nonlocal pipeline
        if pipeline is None:
            pipeline = _load("amazon/chronos-2")

        demand = causal_fill(demand_estimate(sales, in_stock_df))
        if normalise:
            level = demand.iloc[:, -LEVEL_WEEKS:].mean(axis=1).clip(lower=LEVEL_FLOOR)
            series = (demand.iloc[:, -context_weeks:] / level.to_numpy()[:, None]).to_numpy(
                dtype=np.float32
            )
        else:
            level = None
            series = demand.iloc[:, -context_weeks:].to_numpy(dtype=np.float32)

        inputs = series[:, None, :]  # (series, 1, weeks)
        forecast = pipeline.predict(
            inputs,
            prediction_length=len(HORIZONS),
            batch_size=batch_size,
            cross_learning=cross_learning,
        )
        stacked = np.stack([np.asarray(t, dtype=float).squeeze(0) for t in forecast])

        grid = np.asarray(pipeline.quantiles, dtype=float)
        q = float(np.clip(quantile, grid[0], grid[-1]))
        high = min(max(int(np.searchsorted(grid, q)), 1), len(grid) - 1)
        weight = (q - grid[high - 1]) / (grid[high] - grid[high - 1])
        point = stacked[:, high - 1, :] * (1 - weight) + stacked[:, high, :] * weight

        if normalise:
            point = point * level.to_numpy()[:, None]

        future = pd.DatetimeIndex(
            [week_offset(sales, len(sales.columns) - 1 + h) for h in HORIZONS]
        )
        return pd.DataFrame(np.clip(point, 0.0, None), index=sales.index, columns=future)

    return forecaster
