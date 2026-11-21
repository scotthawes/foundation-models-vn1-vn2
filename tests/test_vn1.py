"""The VN1 score, the organisers' baseline, and the feature builder's no-leakage rule.

`test_ma12_reproduces_the_organisers_baseline` is the important one: it pins the loader and the
scorer to a number that came from outside this repo (0.6597 on all 15,053 series).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn1 import data as vn1data
from vn1 import forecaster as vn1f
from vn1.features import HORIZON, _snapshot, train_anchors
from vn1.score import score


def toy(cells: list[list[float]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([(0, 1, 367)], names=["Client", "Warehouse", "Product"])
    columns = pd.date_range("2024-01-01", periods=len(cells[0]), freq="W-MON")
    return pd.DataFrame(cells, index=index, columns=columns)


def test_a_perfect_forecast_scores_zero():
    actual = toy([[1.0, 2.0]])
    assert score(actual, actual) == 0.0


def test_a_zero_forecast_scores_two():
    actual = toy([[1.0, 2.0]])
    assert score(toy([[0.0, 0.0]]), actual) == 2.0


def test_unbiased_error_scores_less_than_a_biased_one_with_the_same_absolute_error():
    actual = toy([[1.0, 1.0]])
    unbiased = toy([[0.0, 2.0]])  # same absolute error, no net bias
    biased = toy([[2.0, 2.0]])    # same absolute error, +2 net
    assert score(unbiased, actual) == pytest.approx(1.0)
    assert score(biased, actual) == pytest.approx(2.0)


def test_ma12_reproduces_the_organisers_baseline():
    prediction = vn1f.ma12(vn1data.phase0_sales())
    assert score(prediction, vn1data.phase1_sales()) == pytest.approx(0.6597, abs=1e-3)


def test_training_anchors_leave_the_whole_horizon_inside_the_history():
    weeks = vn1data.phase0_sales().shape[1]
    anchors = train_anchors(weeks)
    assert anchors
    assert max(anchors) + HORIZON < weeks


def test_the_feature_snapshot_ignores_weeks_after_the_anchor():
    sales = vn1data.phase0_sales().iloc[:3, :80]
    price = vn1data.phase0_price().iloc[:3, :80]
    anchor = 60

    before = _snapshot(sales, price, anchor)
    scrambled = sales.copy()
    scrambled.iloc[:, anchor + 1 :] = 999.0
    scrambled_price = price.copy()
    scrambled_price.iloc[:, anchor + 1 :] = 999.0
    after = _snapshot(scrambled, scrambled_price, anchor)

    pd.testing.assert_frame_equal(before, after)
