"""Two deliberately dumb forecasters, to give the foundation models something to be measured against.

A model that cannot beat these has not earned its inference cost. Both are censoring-aware in the
same way the benchmark is: a week with `in_stock == False` records the shelf, not the demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import week_offset
from ..play import ForecasterFn

SEASON_LENGTH = 52  # weeks in the yearly cycle these series run on


def _censored_to_nan(sales: pd.DataFrame, in_stock_df: pd.DataFrame) -> pd.DataFrame:
    out = sales.copy()
    out[~in_stock_df.reindex(columns=sales.columns)] = np.nan
    return out


def _future(sales: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """The `horizon` Mondays after the last week of `sales`."""
    last = len(sales.columns) - 1
    return pd.DatetimeIndex([week_offset(sales, last + h) for h in range(1, horizon + 1)])


def naive_mean(moving_average: int = 13, horizon: int = 8) -> ForecasterFn:
    """The same number every week ahead: the mean of the last `moving_average` uncensored weeks.

    This is the benchmark with its seasonality removed, which is what makes it a useful control:
    any gain a foundation model shows over it has to come from something other than knowing that
    December sells more than August.
    """

    def forecaster(sales, in_stock_df, master_df, state) -> pd.DataFrame:
        demand = _censored_to_nan(sales, in_stock_df)
        base = demand.iloc[:, -moving_average:].mean(axis=1)
        future = _future(sales, horizon)
        return pd.DataFrame(
            np.repeat(base.values.reshape(-1, 1), horizon, axis=1),
            index=sales.index,
            columns=future,
        )

    return forecaster


def seasonal_naive(horizon: int = 8) -> ForecasterFn:
    """The value of the same week last year, repeated.

    Falls back to the recent mean where last year's week is censored or the history is short, so
    the forecaster never returns NaN and never invents a number from a shelf that ran dry.
    """

    def forecaster(sales, in_stock_df, master_df, state) -> pd.DataFrame:
        demand = _censored_to_nan(sales, in_stock_df)
        future = _future(sales, horizon)
        values = np.empty((len(sales), horizon), dtype=float)
        for h in range(horizon):
            lag = demand.iloc[:, len(demand.columns) - (SEASON_LENGTH - h) - 1]
            fallback = demand.iloc[:, -13:].mean(axis=1)
            values[:, h] = lag.fillna(fallback).to_numpy()
        return pd.DataFrame(values, index=sales.index, columns=future)

    return forecaster
