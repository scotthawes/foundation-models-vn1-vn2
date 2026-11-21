"""Order to the newsvendor quantile of the summed demand, and see if it beats the moving average.

The earlier quantile attempt summed per-week quantiles, which is not the quantile of the total.
Here the paths are sampled from Chronos's nine per-week quantiles, summed over the coverage window,
and the level is the `service` quantile of those totals. VN2's cost ratio fixes the theory at
1/(1+0.2) = 0.833; the sweep around it is a check, not a search for a winner.

Usage:
    python scripts/newsvendor_grid.py --anchors tuning
    python scripts/newsvendor_grid.py --anchors evaluation --services 0.833 --coverages 3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn2 import backtest  # noqa: E402
from vn2 import data as vn2data  # noqa: E402
from vn2 import forecaster, play  # noqa: E402
from vn2.level_policy import newsvendor_levels, sample_paths  # noqa: E402


def make_policy(quantiles_fn, coverage: float, service: float, draws: int, paths_cache: dict):
    """The order-up-to policy with the newsvendor level, sampling each history prefix once."""
    net_columns = ["In Transit W+1", "In Transit W+2", "End Inventory"]

    def policy(sales, in_stock_df, master_df, state):
        key = len(sales.columns)
        if key not in paths_cache:
            paths_cache[key] = sample_paths(
                quantiles_fn(sales, in_stock_df, master_df, state), draws=draws
            )
        level = newsvendor_levels(paths_cache[key], coverage, service)
        net = state[net_columns].sum(axis=1).to_numpy()
        order = np.clip(np.round(level - net, 0), 0.0, None).astype(int)
        return pd.Series(order, index=sales.index)

    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", default="0.7,0.833,0.9")
    parser.add_argument("--coverages", default="2,2.5,3,3.5")
    parser.add_argument("--anchors", default="tuning", choices=["tuning", "evaluation"])
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    services = [float(s) for s in args.services.split(",") if s.strip()]
    coverages = [float(c) for c in args.coverages.split(",") if c.strip()]

    sales, in_stock, master = vn2data.sales_full(), vn2data.in_stock(), vn2data.master()
    anchors = {
        "tuning": backtest.TUNING_ANCHORS,
        "evaluation": backtest.EVALUATION_ANCHORS,
    }[args.anchors]
    folds = backtest.make_folds(anchors)

    warnings.filterwarnings("ignore")
    quantiles_fn = backtest.cache_forecasts(forecaster.chronos_quantiles())
    paths_cache: dict = {}
    rows = []

    for coverage in coverages:
        for service in services:
            policy = make_policy(quantiles_fn, coverage, service, args.draws, paths_cache)
            scores = []
            for fold in folds:
                weekly, _orders = play.run_window(
                    policy,
                    anchor=fold.anchor,
                    sales=sales,
                    in_stock_df=in_stock,
                    master_df=master,
                    initial=fold.state,
                )
                scores.append(play.score(weekly))
            rows.append(
                {
                    "coverage": coverage,
                    "service": service,
                    "mean": round(float(np.mean(scores)), 1),
                    "std": round(float(np.std(scores, ddof=1)), 1),
                }
            )
            print(f"coverage {coverage:<4} service {service:<6} mean {rows[-1]['mean']:>8,.1f}  "
                  f"std {rows[-1]['std']:>7,.1f}", flush=True)

    results = pd.DataFrame(rows).sort_values("mean")
    base = Path("results") / (
        f"newsvendor-{args.anchors}{'-' + args.tag if args.tag else ''}.csv"
    )
    base.parent.mkdir(exist_ok=True)
    results.to_csv(base, index=False)
    print("\nbest configurations (lower is better)")
    print(results.head(8).to_string(index=False))
    print(f"\nwrote {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
