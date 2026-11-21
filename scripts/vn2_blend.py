"""Blend the moving average with Chronos-2 for VN2, and see if offsetting errors add up.

The two arms fail in opposite directions on the played weeks — the average carries 2,434 EUR of
holding against 1,481 of lost sales, Chronos-2 the reverse — and their calibration differs by volume
decile. A weighted blend of their forecasts is the cheapest way to exploit that.

The weight is chosen on the tuning anchors; the evaluation anchors and the competition weeks are
scored once for the chosen weight, in `--anchors` order.

Usage:
    python scripts/vn2_blend.py --anchors tuning
    python scripts/vn2_blend.py --anchors evaluation --weights 0.5
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
from vn2 import forecaster, play  # noqa: E402


def blend_policy(ma_forecast, model_forecast, weight: float, coverage: float) -> play.PolicyFn:
    """Order up to `coverage` weeks of `weight*MA + (1-weight)*model`."""

    def policy(sales, in_stock_df, master_df, state):
        forecast = weight * ma_forecast(sales, in_stock_df, master_df, state) + (
            1 - weight
        ) * model_forecast(sales, in_stock_df, master_df, state)
        return play.order_up_to(forecast, state, coverage)

    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", default="tuning", choices=["tuning", "evaluation"])
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--coverages", default="3,3.5")
    parser.add_argument("--quantile", type=float, default=0.6)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    coverages = [float(c) for c in args.coverages.split(",")]
    anchors = {"tuning": backtest.TUNING_ANCHORS, "evaluation": backtest.EVALUATION_ANCHORS}
    folds = backtest.make_folds(anchors[args.anchors])

    sales, in_stock, master = vn2data.sales_full(), vn2data.in_stock(), vn2data.master()
    ma = backtest.cache_forecasts(forecaster.seasonal_moving_average(13))
    model = backtest.cache_forecasts(
        forecaster.chronos2(quantile=args.quantile, cross_learning=True)
    )

    rows = []
    for weight in weights:
        for coverage in coverages:
            policy = blend_policy(ma, model, weight, coverage)
            scores = []
            started = time.time()
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
                    "weight_ma": weight,
                    "coverage": coverage,
                    "mean": round(float(np.mean(scores)), 1),
                    "std": round(float(np.std(scores, ddof=1)), 1),
                }
            )
            print(f"MA weight {weight:<5} coverage {coverage:<4} mean {rows[-1]['mean']:>8,.1f} "
                  f"std {rows[-1]['std']:>7,.1f}  ({time.time()-started:,.0f}s)", flush=True)

    results = pd.DataFrame(rows).sort_values("mean")
    out = Path("results") / (
        f"vn2-blend-{args.anchors}{'-' + args.tag if args.tag else ''}.csv"
    )
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print("\nbest blends (lower is better)")
    print(results.head(6).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
