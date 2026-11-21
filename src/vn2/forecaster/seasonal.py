"""The competition's benchmark forecaster: a deseasonalized moving average.

This is `vn2_benchmark.py`'s arithmetic written out as a forecaster rather than a policy, so it can
be wrapped in the shared order-up-to policy with any coverage. With coverage 4 it must reproduce the
published 4,334.0 EUR exactly; that is the second pin in `tests/test_play.py`, and it is what makes
the shared policy a fair frame for the other arms.

Three steps, as the original:

1. Weeks with a shortage are set aside: their sales are not the demand, so they go to NaN.
2. Seasonality: the mean sales of each ISO week number over all 599 rows, normalised to a mean of 1.
3. Forecast: the mean of the last `moving_average` weeks of deseasonalized sales, reseasonalized
   for each future week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import week_offset
from ..play import ForecasterFn

DEFAULT_MOVING_AVERAGE = 13
HORIZON = 8  # weeks produced; an order placed in round k needs weeks k .. k+7 at most


def seasonal_factors(sales: pd.DataFrame) -> pd.Series:
    """One factor per ISO week number, from the mean sales of that week across all rows, mean 1."""
    season = sales.mean().rename("Demand").to_frame()
    season["Week Number"] = season.index.isocalendar().week
    season = season.groupby("Week Number").mean()
    return season / season.mean()


def seasonal_moving_average(moving_average: int = DEFAULT_MOVING_AVERAGE,
                            horizon: int = HORIZON) -> ForecasterFn:
    """A forecaster that forecasts with a deseasonalized moving average."""
    if moving_average < 1:
        raise ValueError(f"moving_average must be at least 1, got {moving_average}")

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        censored = sales.copy()
        censored[~in_stock_df.reindex(columns=sales.columns)] = np.nan

        season = seasonal_factors(censored)
        weeks = censored.columns.isocalendar().week.values
        deseasonalized = censored / season.loc[weeks].values.reshape(-1)
        base = deseasonalized.iloc[:, -moving_average:].mean(axis=1)

        future = pd.DatetimeIndex(
            [week_offset(sales, len(sales.columns) - 1 + h) for h in range(1, horizon + 1)]
        )
        factors = season.loc[future.isocalendar().week.values].values.reshape(-1)
        values = base.values.reshape(-1, 1) * factors.reshape(1, -1)
        return pd.DataFrame(values, index=sales.index, columns=future)

    return forecaster
