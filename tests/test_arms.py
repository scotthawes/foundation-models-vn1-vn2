"""Tests for the two hosted-API arms, run without credentials or network.

The parts that can be wrong quietly are the frame in and the frame out: a lost Store x Product, a
censored week that leaks into the fill, a forecast column that lands in the wrong order. Those are
pure functions, so they are pinned here; the API call itself is exercised by the smoke script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn2 import keys
from vn2 import forecaster
from vn2.forecaster import chronos_arm, timegpt_arm


def panel(rows: int = 3, weeks: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small panel: `rows` Store x Product series, `weeks` contiguous Mondays."""
    columns = pd.date_range("2023-01-02", periods=weeks, freq="W-MON")
    index = pd.MultiIndex.from_product(
        [["S1"], [f"P{i}" for i in range(rows)]], names=["Store", "Product"]
    )
    sales = pd.DataFrame(
        np.arange(rows * weeks, dtype=float).reshape(rows, weeks), index=index, columns=columns
    )
    in_stock = pd.DataFrame(True, index=index, columns=columns)
    return sales, in_stock


def test_causal_fill_uses_only_weeks_before_the_gap():
    series = pd.DataFrame([[1.0, 3.0, np.nan, 5.0]])
    filled = chronos_arm.causal_fill(series)
    assert filled.iloc[0, 2] == pytest.approx(2.0)  # median of the two weeks before it


def test_causal_fill_does_not_move_when_a_later_week_changes():
    before = pd.DataFrame([[1.0, 3.0, np.nan, 5.0]])
    after = pd.DataFrame([[1.0, 3.0, np.nan, 9_999.0]])
    assert chronos_arm.causal_fill(before).iloc[0, 2] == chronos_arm.causal_fill(after).iloc[0, 2]


def test_long_frame_is_complete_and_censored_weeks_are_filled():
    sales, in_stock = panel()
    sales.iloc[0, 4] = 0.0
    in_stock.iloc[0, 4] = False  # shelf ran dry: not demand

    long = timegpt_arm.long_frame(sales, in_stock)
    assert len(long) == len(sales) * len(sales.columns)
    assert long["unique_id"].nunique() == len(sales)
    assert not long["y"].isna().any(), "TimeGPT refuses missing values"
    assert long["ds"].dtype.kind == "M"


def test_forecast_wide_maps_ids_back_to_rows_and_names_future_weeks():
    sales, _in_stock = panel()
    ids = timegpt_arm.series_ids(sales)
    future = [sales.columns[-1] + pd.Timedelta(weeks=h) for h in range(1, 9)]
    fake = pd.DataFrame(
        [
            {"unique_id": series, "ds": week, "TimeGPT": (n + 1) * 10.0}
            for n, series in enumerate(ids)
            for week in future
        ]
    )
    fake.loc[fake["unique_id"] == ids[1], "TimeGPT"] = -3.0  # demand cannot be negative
    fake = fake[fake["unique_id"] != ids[0]]  # one series missing: no order, not a missing cell

    wide = timegpt_arm.forecast_wide(fake, sales)
    assert wide.shape == (len(sales), 8)
    assert list(wide.index) == list(sales.index)
    assert list(wide.columns) == future
    assert (wide.to_numpy() >= 0).all()
    assert wide.iloc[0].sum() == 0.0
    assert wide.iloc[1].sum() == 0.0


def test_api_arms_fail_fast_to_a_helpful_message_without_a_key(monkeypatch, tmp_path):
    monkeypatch.setattr(keys, "CONFIG", tmp_path / "absent.env")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.delenv("NIXTLA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TABPFN_TOKEN"):
        forecaster.tabpfn_api()(None, None, None, None)
    with pytest.raises(RuntimeError, match="NIXTLA_API_KEY"):
        forecaster.timegpt()(None, None, None, None)


def test_an_unknown_target_transform_is_refused_at_construction():
    with pytest.raises(ValueError, match="target transform"):
        forecaster.tabpfn_api(target="sqrt")


def test_every_arm_is_a_factory_returning_a_callable():
    for factory in (
        forecaster.naive_mean,
        forecaster.seasonal_naive,
        forecaster.seasonal_moving_average,
        forecaster.tabpfn,
        forecaster.chronos,
        forecaster.tabpfn_api,
        forecaster.timegpt,
    ):
        assert callable(factory())
