"""Run every arm over the back-test folds and tabulate the score.

The forecast is cached per history prefix, so a coverage sweep costs one forecast per round instead
of one per (round x coverage): coverage changes the order, not the forecast. Two folds also share
prefixes — anchor 104 at round 5 sees weeks 0..108, exactly what anchor 108 sees at round 1 — so the
cache is shared across folds, which is exact, not an approximation. The arms take their history from
`sales` alone (none of them looks at `state`), which is what makes that reuse sound; an arm that
read the state would need `state` in the key.

Everything else is the competition's arithmetic: each fold starts from the warm-up state built by
`vn2.backtest.make_folds`, and the score is `vn2.play.score`.

Usage:
    python scripts/backtest.py --anchors tuning
    python scripts/backtest.py --anchors evaluation --arms benchmark,chronos,tabpfn_api
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vn2 import backtest  # noqa: E402
from vn2 import data as vn2data  # noqa: E402
from vn2 import forecaster, play  # noqa: E402
ARMS = {
    "benchmark": lambda: forecaster.seasonal_moving_average(13),
    "seasonal_ma": lambda: forecaster.seasonal_moving_average(13),
    "naive_mean": forecaster.naive_mean,
    "seasonal_naive": forecaster.seasonal_naive,
    "chronos": forecaster.chronos,
    "tabpfn_api": forecaster.tabpfn_api,
    "timegpt": forecaster.timegpt,
}

ANCHOR_SETS = {
    "tuning": backtest.TUNING_ANCHORS,
    "evaluation": backtest.EVALUATION_ANCHORS,
    "both": backtest.TUNING_ANCHORS + backtest.EVALUATION_ANCHORS,
}


def cache_forecasts(forecaster_fn: play.ForecasterFn) -> play.ForecasterFn:
    """A thin alias, so the script and the tests share one implementation."""
    return backtest.cache_forecasts(forecaster_fn)


def score_fold(fold: backtest.Fold, policy_fn: play.PolicyFn) -> dict[str, float]:
    weekly, _orders = play.run_window(
        policy_fn,
        anchor=fold.anchor,
        sales=vn2data.sales_full(),
        in_stock_df=vn2data.in_stock(),
        master_df=vn2data.master(),
        initial=fold.state,
    )
    scored = weekly.loc[list(play.ScoredWeeks)]
    return {
        "score": play.score(weekly),
        "holding": round(float(scored["Holding Cost"].sum()), 1),
        "shortage": round(float(scored["Shortage Cost"].sum()), 1),
        "lost_sales": int(scored["Lost Sales"].sum()),
        "end_inventory": int(scored["End Inventory"].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", default="tuning", choices=sorted(ANCHOR_SETS))
    parser.add_argument("--arms", default="benchmark,seasonal_ma,chronos,tabpfn_api")
    parser.add_argument("--coverages", default="3,3.5,4,4.5,5")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    coverages = [float(c) for c in args.coverages.split(",") if c.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")

    folds = backtest.make_folds(ANCHOR_SETS[args.anchors])
    print(f"{len(folds)} folds at anchors {[f.anchor for f in folds]}")
    print(f"arms {arm_names} · coverages {coverages}\n")

    rows = []
    for name in arm_names:
        started = time.time()
        forecaster_fn = cache_forecasts(ARMS[name]())  # one model load per arm, not per fold
        # The benchmark is the published 4,334.0 at coverage 4; sweeping it would only relabel it.
        arm_coverages = [4.0] if name == "benchmark" else coverages
        for fold in folds:
            for coverage in arm_coverages:
                policy = play.order_up_to_policy(forecaster_fn, coverage=coverage)
                rows.append(
                    {"arm": name, "coverage": coverage, "anchor": fold.anchor,
                     **score_fold(fold, policy)}
                )
        print(f"{name}: {time.time() - started:,.0f}s", flush=True)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["arm", "coverage"])["score"]
        .agg(["mean", "std", "min", "max"])
        .round(1)
    )
    print("\nmean score by arm and coverage (lower is better)")
    print(summary.to_string())

    best = summary["mean"].groupby("arm").idxmin()
    print("\nbest coverage per arm")
    for arm, label in best.items():
        coverage = label[1] if isinstance(label, tuple) else label
        print(f"  {arm:12s} coverage {coverage:<4} mean {summary.loc[(arm, coverage), 'mean']:,.1f}")

    out = Path("results") / f"backtest-{args.tag or args.anchors}.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
