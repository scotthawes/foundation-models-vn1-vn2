"""Play the VN2 game over any window of the history, with the official arithmetic.

`vn2_simulate.play_week` and `vn2_simulate.check_order` are the competition's own code and are
called, never copied: this module supplies only the loop and the date bookkeeping that the official
scripts hard-code to the competition's single anchor. `tests/test_play.py` pins the result to the
published score, so a change that breaks the rules fails before any model is run.

Two ways to start a window:

- the competition itself, from `vn2.data.initial_state()`, at `anchor = COMPETITION_ANCHOR`;
- a back-test, from a state warmed up by `warmup_state`, at an earlier anchor.

The warm-up is shared across arms: every forecaster is handed the identical starting state, so a
difference in score is a difference in forecasting and not in where the shelf happened to be.
"""
from __future__ import annotations

import sys
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import data as vn2data

if str(vn2data.OFFICIAL) not in sys.path:
    sys.path.insert(0, str(vn2data.OFFICIAL))

import vn2_simulate  # noqa: E402  (the competition's rules; imported after its folder is on the path)

#: The policy signature the competition defined: sales, in-stock flags, master, state -> orders.
PolicyFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], Iterable[float]]

#: A forecaster has the same four arguments but returns a table of predicted demand, one column per
#: week ahead. Every arm in this repo is such a forecaster behind one shared order-up-to policy.
ForecasterFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]

Rounds = 6  # orders 1 to 6; weeks 7 and 8 are played with no order
Window = 8  # weeks 1 to 8
ScoredWeeks = (3, 4, 5, 6, 7, 8)  # weeks 1 and 2 are the setup, identical for every player
MIN_WARMUP_HISTORY = 52  # a policy needs a year of context before its first warm-up order


def zero_state(index: pd.Index) -> pd.DataFrame:
    """A state with nothing on the shelf and nothing on the road, shaped like the real file."""
    money = ["Cumulative Holding Cost", "Cumulative Shortage Cost"]
    columns = [
        "Start Inventory",
        "Sales",
        "Missed Sales",
        "End Inventory",
        "In Transit W+1",
        "In Transit W+2",
        "Holding Cost",
        "Shortage Cost",
        *money,
    ]
    state = pd.DataFrame(0, index=index, columns=columns)
    return state.astype({c: "float64" for c in money})


def warmup_state(
    anchor: int,
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    master_df: pd.DataFrame,
    policy_fn: PolicyFn,
    warmup_weeks: int = 52,
) -> pd.DataFrame:
    """The state at the end of week `anchor`, reached by playing the official rules from a zero state.

    An order-up-to policy settles within a few weeks of its lead time, so after a year the shelf is
    wherever that policy keeps it, whatever it started at. The first warm-up step must still have a
    year of history to look at, hence `MIN_WARMUP_HISTORY`.
    """
    first_week = anchor - warmup_weeks
    if first_week < MIN_WARMUP_HISTORY:
        raise ValueError(
            f"anchor {anchor} with {warmup_weeks} warm-up weeks starts at week {first_week}, "
            f"which leaves less than the {MIN_WARMUP_HISTORY} weeks of history a policy needs"
        )
    state = zero_state(sales.index)
    for week in range(first_week, anchor + 1):
        history = sales.iloc[:, :week].copy()  # the policy knows weeks 0 .. week-1
        flags = in_stock_df[history.columns].copy()
        order = policy_fn(history, flags, master_df.copy(), state.copy())
        order = vn2_simulate.check_order(order, sales.index, f"Warm-up order before week {week}")
        state = vn2_simulate.play_week(state, sales.iloc[:, week], order)
    return state


def run_window(
    policy_fn: PolicyFn,
    anchor: int,
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    master_df: pd.DataFrame,
    initial: pd.DataFrame,
    rounds: int = Rounds,
    window: int = Window,
) -> tuple[pd.DataFrame, list[pd.Series]]:
    """Play `window` weeks from `anchor` under the official rules.

    `anchor` is the index of the last week the policy knows. Round `k` sees weeks `0 .. anchor+k-1`
    and orders before week `anchor+k`; that order lands at the start of week `anchor+k+2`. Returns
    the per-week cost table and the orders placed.
    """
    if anchor + window > len(sales.columns) - 1:
        raise ValueError(
            f"anchor {anchor} + window {window} runs past the last of {len(sales.columns)} weeks of sales"
        )
    state = initial.copy()
    states = [state]
    orders: list[pd.Series] = []
    for week in range(1, window + 1):
        if week <= rounds:
            history = sales.iloc[:, : anchor + week].copy()  # knows weeks 0 .. anchor+week-1
            flags = in_stock_df[history.columns].copy()
            order = policy_fn(history, flags, master_df.copy(), state.copy())
            order = vn2_simulate.check_order(order, sales.index, f"Order {week} (before week {week})")
            orders.append(order)  # weeks 7 and 8 are played with no order, so they are not orders
        else:
            order = pd.Series(0, index=sales.index, name="Orders")
        state = vn2_simulate.play_week(state, sales.iloc[:, anchor + week], order)
        states.append(state)
    return summarize(sales, anchor, states), orders


def summarize(sales: pd.DataFrame, anchor: int, states: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per played week: the totals of the 599 rows and the cost of the week.

    The arithmetic is `vn2_simulate.summarize`'s; only the date label differs, because a back-test
    window is not the competition's calendar.
    """
    dates = sales.columns[anchor + 1 : anchor + len(states)]
    rows = []
    for number, (state, date) in enumerate(zip(states[1:], dates), start=1):
        total = state.sum()
        rows.append(
            {
                "Week": number,
                "Date": date,
                "End Inventory": total["End Inventory"],
                "Lost Sales": total["Missed Sales"],
                "Holding Cost": total["Holding Cost"],
                "Shortage Cost": total["Shortage Cost"],
            }
        )
    weekly = pd.DataFrame(rows).set_index("Week")
    weekly["Cost"] = weekly["Holding Cost"] + weekly["Shortage Cost"]
    return weekly


def setup(weekly: pd.DataFrame) -> float:
    """Weeks 1 and 2: no order reaches the shelf in time, so this part is the same for every arm."""
    return round(float(weekly.loc[[1, 2], "Cost"].sum()), 1)


def score(weekly: pd.DataFrame) -> float:
    """The competition's score: the cost of weeks 3 to 8."""
    return round(float(weekly.loc[list(ScoredWeeks), "Cost"].sum()), 1)


def order_up_to(forecast: pd.DataFrame, state: pd.DataFrame, coverage: float) -> pd.Series:
    """The competition's ordering arithmetic, with the coverage made a parameter.

    The level is the forecast demand of the next `coverage` weeks (fractional weeks allowed) and the
    order tops the shelf and the road back up to it. `vn2_benchmark.py` hard-codes coverage 4; the
    competition's own write-up reports that 3.5 would have scored about 3,900 EUR, so the number is
    worth tuning rather than inheriting.
    """
    if coverage <= 0:
        raise ValueError(f"coverage must be positive, got {coverage}")
    if coverage > forecast.shape[1]:
        raise ValueError(
            f"coverage {coverage} exceeds the {forecast.shape[1]} weeks the forecaster produced"
        )
    whole = int(np.floor(coverage))
    fraction = coverage - whole
    level = forecast.iloc[:, :whole].sum(axis=1)
    if fraction:
        level = level + fraction * forecast.iloc[:, whole]
    net = state[["In Transit W+1", "In Transit W+2", "End Inventory"]].sum(axis=1)
    return (level - net).clip(lower=0).round(0).astype(int)


def order_up_to_policy(forecaster: ForecasterFn, coverage: float) -> PolicyFn:
    """Wrap any forecaster in the shared order-up-to policy, so arms differ only in their forecast."""

    def policy(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.Series:
        forecast = forecaster(sales, in_stock_df, master_df, state)
        return order_up_to(forecast, state, coverage)

    policy.forecaster = forecaster  # type: ignore[attr-defined]
    policy.coverage = coverage  # type: ignore[attr-defined]
    return policy


def competition_run(policy_fn: PolicyFn) -> tuple[pd.DataFrame, float, list[pd.Series]]:
    """The six rounds as the competition played them: the real initial state, the real anchor."""
    weekly, orders = run_window(
        policy_fn,
        anchor=vn2data.COMPETITION_ANCHOR,
        sales=vn2data.sales_full(),
        in_stock_df=vn2data.in_stock(),
        master_df=vn2data.master(),
        initial=vn2data.initial_state(),
    )
    return weekly, score(weekly), orders
