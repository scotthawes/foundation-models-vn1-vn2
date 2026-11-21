"""The harness is pinned to the competition's published numbers before any model is run.

Two independent pins, both of which must hold:

1. `vn2_simulate` called on the benchmark's own order files scores 4,334.0 EUR. This is the
   competition's arithmetic on the competition's data and is the ground truth.
2. This repo's `run_window`, driving the benchmark forecaster through this repo's order-up-to
   policy, scores the same number and produces the same orders. That is what licenses
   `run_window` to stand in for the official loop at the other 140-odd anchors.

If a change makes these disagree, the back-test is measuring something other than the competition.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from vn2 import data as vn2data
from vn2 import play

OFFICIAL_BENCHMARK_SCORE = 4334.0
OFFICIAL_SETUP = 913.8


@pytest.fixture(scope="module")
def benchmark_policy():
    """The competition's benchmark, driven the official way, so its orders are the reference."""
    sys.path.insert(0, str(vn2data.OFFICIAL))
    import vn2_benchmark

    return vn2_benchmark.policy


@pytest.fixture(scope="module")
def our_benchmark_policy():
    """The same forecaster and the same ordering rule, assembled from this repo's parts."""
    from vn2.forecaster import seasonal_moving_average

    return play.order_up_to_policy(seasonal_moving_average(13), coverage=4.0)


def test_official_benchmark_scores_the_published_number(benchmark_policy):
    weekly, score, orders = play.competition_run(benchmark_policy)
    assert play.setup(weekly) == OFFICIAL_SETUP
    assert score == OFFICIAL_BENCHMARK_SCORE
    assert len(orders) == 6


def test_our_harness_matches_the_official_loop(benchmark_policy, our_benchmark_policy):
    """The pin: same weekly costs, same orders, same score, from a different code path."""
    official_weekly, official_score, official_orders = play.competition_run(benchmark_policy)
    our_weekly, our_score, our_orders = play.competition_run(our_benchmark_policy)

    assert our_score == official_score == OFFICIAL_BENCHMARK_SCORE
    pd.testing.assert_frame_equal(our_weekly, official_weekly)
    for week, (mine, theirs) in enumerate(zip(our_orders, official_orders), start=1):
        pd.testing.assert_series_equal(mine, theirs, check_names=False, obj=f"order {week}")


def test_order_up_to_never_orders_below_zero_and_is_whole():
    import numpy as np

    forecast = pd.DataFrame(
        np.array([[5.0, 5.0, 5.0, 5.0]] * 2),
        index=pd.MultiIndex.from_tuples([(0, 1), (0, 2)], names=["Store", "Product"]),
        columns=pd.date_range("2024-04-15", periods=4, freq="W-MON"),
    )
    state = pd.DataFrame(
        {"End Inventory": [100, 0], "In Transit W+1": [100, 0], "In Transit W+2": [0, 0]},
        index=forecast.index,
    )
    orders = play.order_up_to(forecast, state, coverage=4.0)
    assert (orders >= 0).all()
    assert (orders == orders.round()).all()
    assert orders.iloc[0] == 0  # already over-stocked: nothing to order
    assert orders.iloc[1] == 20  # 4 weeks of 5, nothing on the shelf


def test_order_up_to_accepts_fractional_coverage():
    forecast = pd.DataFrame(
        [[10.0, 10.0, 10.0, 10.0]],
        index=pd.MultiIndex.from_tuples([(0, 1)], names=["Store", "Product"]),
        columns=pd.date_range("2024-04-15", periods=4, freq="W-MON"),
    )
    state = pd.DataFrame(
        {"End Inventory": [0], "In Transit W+1": [0], "In Transit W+2": [0]}, index=forecast.index
    )
    assert play.order_up_to(forecast, state, coverage=3.5).iloc[0] == 35
    with pytest.raises(ValueError, match="exceeds"):
        play.order_up_to(forecast, state, coverage=5.0)


def test_warmup_state_is_a_realistic_state():
    """The back-test starts every arm from the same warmed-up shelf; check it is not absurd.

    Measured against the real initial state on the quantities that drive the cost. The level on the
    shelf has to be close, and the goods on the road have to be the same order of magnitude, or
    every back-test number would be measuring a start-up rather than a policy.

    The one thing that is deliberately not matched: the competition's initial state has
    `In Transit W+1 == 0` for all 599 rows, so nothing arrives in week 1 of the real game. A
    warmed-up state has a steady flow instead. That is why back-test costs are compared between
    arms at a shared anchor and never read as a forecast of the competition's own score.
    """
    sales, in_stock_df, master_df = vn2data.sales_full(), vn2data.in_stock(), vn2data.master()
    from vn2.forecaster import seasonal_moving_average

    policy = play.order_up_to_policy(seasonal_moving_average(13), coverage=4.0)
    warmed = play.warmup_state(anchor=vn2data.COMPETITION_ANCHOR, sales=sales,
                               in_stock_df=in_stock_df, master_df=master_df, policy_fn=policy)
    real = vn2data.initial_state()

    shelf_ratio = warmed["End Inventory"].mean() / real["End Inventory"].mean()
    assert 0.75 <= shelf_ratio <= 1.5, (
        f"shelf: warmed-up mean {warmed['End Inventory'].mean():.1f} vs real "
        f"{real['End Inventory'].mean():.1f}"
    )

    pipeline = ["End Inventory", "In Transit W+1", "In Transit W+2"]
    warm_total = warmed[pipeline].sum(axis=1).mean()
    real_total = real[pipeline].sum(axis=1).mean()
    assert 0.5 <= warm_total / real_total <= 2.0, (
        f"pipeline: warmed-up mean {warm_total:.1f} vs real {real_total:.1f}"
    )


@pytest.mark.parametrize("coverage", [3.0, 3.5, 4.0])
def test_no_forecaster_sees_past_the_anchor(coverage):
    """A back-test that leaked the played weeks would flatter every arm equally and prove nothing.

    Order `k` is placed before window week `k`, having seen weeks `0 .. anchor+k-1`. So for each
    round: scramble every week from `anchor+k` onwards, replay, and the first `k` orders must be
    byte-identical while the cost must move. Scrambling only what round `k` is not yet entitled to
    is the point — round 2 is *supposed* to react to week 1's realised demand, so scrambling week 1
    as well would test nothing.

    A test that cannot fail is no test: the cost assertion is the control that proves the scramble
    reached the simulation at all.
    """
    from vn2.forecaster import seasonal_moving_average

    sales = vn2data.sales_full()
    anchor = vn2data.COMPETITION_ANCHOR
    policy = play.order_up_to_policy(seasonal_moving_average(13), coverage=coverage)
    common = dict(
        anchor=anchor,
        in_stock_df=vn2data.in_stock(),
        master_df=vn2data.master(),
        initial=vn2data.initial_state(),
    )
    clean_weekly, clean_orders = play.run_window(policy, sales=sales, **common)

    for round_number in range(1, play.Rounds + 1):
        scrambled = sales.copy()
        untouched = scrambled.iloc[:, anchor + round_number :]
        scrambled.iloc[:, anchor + round_number :] = untouched.to_numpy()[:, ::-1] * 7 + 3
        dirty_weekly, dirty_orders = play.run_window(policy, sales=scrambled, **common)

        for week in range(round_number):
            pd.testing.assert_series_equal(
                clean_orders[week], dirty_orders[week], check_names=False,
                obj=f"order {week + 1} with weeks from {anchor + round_number} scrambled",
            )
        assert clean_weekly["Cost"].sum() != dirty_weekly["Cost"].sum(), (
            f"scrambling from week {anchor + round_number} changed nothing, so the test never "
            "exercised the cut"
        )
