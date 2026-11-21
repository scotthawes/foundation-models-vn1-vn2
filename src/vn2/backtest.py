"""Back-test the six-round game across many anchors, with every arm sharing the same folds.

The competition gives one score, at one anchor, from one initial state. That is enough to rank a
submission but not to choose between forecasters: a single window is one draw of demand, and the
winner of VN2 beat the benchmark by 571 EUR over six weeks. So the arms are replayed over a set of
anchors inside the history, and the comparison is made on those.

Three rules keep the comparison honest:

- **Shared folds.** Every arm is scored on the same anchors, from the same warmed-up state. An arm
  evaluated on its own anchors is an anecdote, not a cell in a table.
- **Tuning and evaluation are separate anchors.** `TUNING_ANCHORS` is where a coverage or a window
  is chosen; `EVALUATION_ANCHORS` is where that choice is reported. Choosing and reporting on the
  same anchors measures how well the choice fits the noise.
- **Nothing from the competition's eight weeks.** The last evaluation anchor is 148, whose window
  ends at week 156, the last week of history. The eight weeks the competition played are held back
  for the final run, which is the only place they are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import data as vn2data
from . import play

WARMUP_WEEKS = 52  # a year of settling before the window opens

#: Where a free parameter is chosen. Early anchors, so the choice cannot see the later demand.
TUNING_ANCHORS = tuple(range(104, 132, 4))

#: Where a chosen parameter is reported. Held out from tuning, and still short of the competition.
EVALUATION_ANCHORS = tuple(range(132, 149, 4))


@dataclass(frozen=True)
class Fold:
    """One anchor and the shared state every arm starts from."""

    anchor: int
    state: pd.DataFrame

    @property
    def window_ends(self) -> int:
        return self.anchor + play.Window

    @property
    def first_week(self) -> pd.Timestamp:
        return vn2data.sales_full().columns[self.anchor + 1]


def warmup_policy() -> play.PolicyFn:
    """The policy that settles the shelf before a window opens: the competition's own benchmark.

    Fixed for every arm on purpose. If each arm warmed its own shelf it would be scored partly on
    where it had put the stock, which is a different question from how well it forecasts.
    """
    from .forecaster import seasonal_moving_average

    return play.order_up_to_policy(seasonal_moving_average(13), coverage=4.0)


def make_folds(
    anchors: tuple[int, ...] | list[int] = EVALUATION_ANCHORS,
    warmup_weeks: int = WARMUP_WEEKS,
    warmup: play.PolicyFn | None = None,
    sales: pd.DataFrame | None = None,
    in_stock_df: pd.DataFrame | None = None,
    master_df: pd.DataFrame | None = None,
) -> list[Fold]:
    """Warm one state per anchor, to be shared by every arm scored on it."""
    sales = vn2data.sales_full() if sales is None else sales
    in_stock_df = vn2data.in_stock() if in_stock_df is None else in_stock_df
    master_df = vn2data.master() if master_df is None else master_df
    warmup = warmup_policy() if warmup is None else warmup

    folds = []
    for anchor in anchors:
        if anchor + play.Window > vn2data.COMPETITION_ANCHOR:
            raise ValueError(
                f"anchor {anchor} runs its window past week {vn2data.COMPETITION_ANCHOR}, the last "
                "week of history: a back-test fold must not touch the competition's eight weeks"
            )
        state = play.warmup_state(
            anchor=anchor,
            sales=sales,
            in_stock_df=in_stock_df,
            master_df=master_df,
            policy_fn=warmup,
            warmup_weeks=warmup_weeks,
        )
        folds.append(Fold(anchor=anchor, state=state))
    return folds


def evaluate(folds: list[Fold], policy_fn: play.PolicyFn) -> pd.DataFrame:
    """Score one arm on every fold. One row per anchor, indexed by it."""
    sales = vn2data.sales_full()
    in_stock_df = vn2data.in_stock()
    master_df = vn2data.master()

    rows = []
    for fold in folds:
        weekly, _orders = play.run_window(
            policy_fn,
            anchor=fold.anchor,
            sales=sales,
            in_stock_df=in_stock_df,
            master_df=master_df,
            initial=fold.state,
        )
        scored = weekly.loc[list(play.ScoredWeeks)]
        rows.append(
            {
                "anchor": fold.anchor,
                "window_start": weekly.loc[1, "Date"],
                "score": play.score(weekly),
                "holding": round(float(scored["Holding Cost"].sum()), 1),
                "shortage": round(float(scored["Shortage Cost"].sum()), 1),
                "lost_sales": int(scored["Lost Sales"].sum()),
                "end_inventory": int(scored["End Inventory"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("anchor")


def cache_forecasts(forecaster_fn: play.ForecasterFn) -> play.ForecasterFn:
    """Remember each forecast by the length of the history it was made from.

    Exact rather than approximate: what a fold shows its arm is always a prefix of the same weekly
    table, so the length identifies it — anchor 104 at round 5 and anchor 108 at round 1 both see
    109 columns. No arm reads `state` or `master`, so neither needs a place in the key; an arm that
    did would.
    """
    store: dict[int, pd.DataFrame] = {}

    def cached(sales, in_stock_df, master_df, state):
        key = len(sales.columns)
        if key not in store:
            store[key] = forecaster_fn(sales, in_stock_df, master_df, state)
        return store[key]

    return cached


def summarise(results: pd.DataFrame) -> dict[str, float]:
    """The mean cost across folds, with the spread that decides whether a gap means anything.

    The standard error is the fold-to-fold one. The folds overlap by construction — consecutive
    eight-week windows share weeks — so it understates the true uncertainty; it is reported as the
    noise floor to beat, not as a confidence interval.
    """
    scores = results["score"]
    n = len(scores)
    spread = float(scores.std(ddof=1)) if n > 1 else 0.0
    return {
        "folds": float(n),
        "mean_score": round(float(scores.mean()), 1),
        "std_score": round(spread, 1),
        "se_score": round(spread / (n ** 0.5), 1) if n else 0.0,
        "min_score": round(float(scores.min()), 1),
        "max_score": round(float(scores.max()), 1),
        "mean_shortage": round(float(results["shortage"].mean()), 1),
        "mean_holding": round(float(results["holding"].mean()), 1),
    }
