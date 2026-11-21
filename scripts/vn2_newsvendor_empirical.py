"""Order to the 0.833 fractile of each series' own historical demand, by bootstrap.

This is the textbook answer to VN2's cost structure and the one policy we never tested on its own:
1 EUR per lost sale against 0.2 EUR per unit held implies ordering to the 0.833 quantile of demand
over the coverage window. Rather than take that quantile from a model's forecast, this bootstraps
each series' own history: resample weeks (censored weeks excluded), sum `coverage` of them, and take
the service quantile of the sums.

It is deliberately model-free, so it isolates the question the models kept failing: is the level the
problem, or the per-series allocation?

Usage:
    python scripts/vn2_newsvendor_empirical.py --anchors tuning
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn2 import backtest  # noqa: E402
from vn2 import data as vn2data  # noqa: E402
from vn2 import play  # noqa: E402

NET_COLUMNS = ["In Transit W+1", "In Transit W+2", "End Inventory"]


def empirical_policy(coverage: float, service: float = 0.833, draws: int = 200):
    """Order up to the bootstrap `service` quantile of summed demand, net of stock and transit."""
    whole_weeks = int(np.floor(coverage))
    fraction = coverage - whole_weeks
    width = whole_weeks + (1 if fraction else 0)

    def policy(sales, in_stock_df, master_df, state):
        demand = vn2data.demand_estimate(sales, in_stock_df).to_numpy(dtype=float)
        rng = np.random.default_rng(len(sales.columns))  # causal: depends only on how much history
        levels = np.zeros(len(sales))
        for i in range(len(sales)):
            observed = demand[i][~np.isnan(demand[i])]
            if observed.size == 0:
                continue
            picks = rng.integers(0, observed.size, size=(draws, width))
            sampled = observed[picks]
            if fraction:
                sums = sampled[:, :whole_weeks].sum(axis=1) + fraction * sampled[:, whole_weeks]
            else:
                sums = sampled.sum(axis=1)
            levels[i] = np.quantile(sums, service)

        net = state[NET_COLUMNS].sum(axis=1).to_numpy()
        order = np.clip(np.round(levels - net, 0), 0.0, None).astype(int)
        return pd.Series(order, index=sales.index)

    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", default="tuning", choices=["tuning", "evaluation"])
    parser.add_argument("--services", default="0.7,0.833,0.9")
    parser.add_argument("--coverages", default="2.5,3,3.5")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    anchors = {"tuning": backtest.TUNING_ANCHORS, "evaluation": backtest.EVALUATION_ANCHORS}
    folds = backtest.make_folds(anchors[args.anchors])
    sales, in_stock, master = vn2data.sales_full(), vn2data.in_stock(), vn2data.master()

    rows = []
    for coverage in [float(c) for c in args.coverages.split(",")]:
        for service in [float(s) for s in args.services.split(",")]:
            started = time.time()
            policy = empirical_policy(coverage, service)
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
            rows.append({"coverage": coverage, "service": service,
                         "mean": round(float(np.mean(scores)), 1),
                         "std": round(float(np.std(scores, ddof=1)), 1)})
            print(f"coverage {coverage:<4} service {service:<6} mean {rows[-1]['mean']:>8,.1f} "
                  f"std {rows[-1]['std']:>7,.1f}  ({time.time()-started:,.0f}s)", flush=True)

    results = pd.DataFrame(rows).sort_values("mean")
    out = Path("results") / (
        f"vn2-empirical-{args.anchors}{'-' + args.tag if args.tag else ''}.csv"
    )
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print("\nbest empirical policies (lower is better)")
    print(results.head(6).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
