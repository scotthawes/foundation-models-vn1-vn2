"""Folds stay inside the history, and the forecast cache is exact rather than approximate.

Both are silent-failure risks: a fold that reaches into the competition's eight weeks would
contaminate the one result that is meant to be untouched, and a cache keyed on the wrong thing would
serve a forecast made from a different history.
"""
from __future__ import annotations

import pandas as pd
import pytest

from vn2 import backtest, data as vn2data


def panel(weeks: int = 6, rows: int = 2) -> pd.DataFrame:
    columns = pd.date_range("2023-01-02", periods=weeks, freq="W-MON")
    index = pd.MultiIndex.from_product(
        [["S1"], [f"P{i}" for i in range(rows)]], names=["Store", "Product"]
    )
    return pd.DataFrame(0.0, index=index, columns=columns)


def test_the_last_evaluation_fold_ends_on_the_last_week_of_history():
    fold = backtest.make_folds((148,))[0]
    assert fold.window_ends == vn2data.COMPETITION_ANCHOR


def test_a_fold_that_would_touch_the_competition_is_refused():
    with pytest.raises(ValueError, match="competition"):
        backtest.make_folds((152,))


def test_warm_up_is_deterministic_so_arms_share_their_starting_state():
    sales, in_stock, master = vn2data.sales_full(), vn2data.in_stock(), vn2data.master()
    first = backtest.make_folds((104,), sales=sales, in_stock_df=in_stock, master_df=master)[0]
    second = backtest.make_folds((104,), sales=sales, in_stock_df=in_stock, master_df=master)[0]
    pd.testing.assert_frame_equal(first.state, second.state)


def test_the_cache_reuses_a_forecast_only_for_the_same_history_length():
    calls = []

    def fake(sales, in_stock_df, master_df, state):
        calls.append(len(sales.columns))
        return pd.DataFrame(1.0, index=sales.index, columns=[sales.columns[-1]])

    cached = backtest.cache_forecasts(fake)
    history = panel()

    first = cached(history.iloc[:, :4], None, None, None)
    again = cached(history.iloc[:, :4], None, None, None)
    longer = cached(history.iloc[:, :5], None, None, None)

    assert calls == [4, 5]
    assert first is again
    assert longer is not first
