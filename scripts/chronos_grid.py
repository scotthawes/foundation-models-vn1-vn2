"""Can Chronos beat the moving average? Give it the levers the average does not have.

VN2 charges 1 EUR per lost sale and 0.2 EUR per unit held, so the stock level that minimises expected
cost is a **high quantile** of demand, not the median: the newsvendor critical fractile is
1 / (1 + 0.2) = 0.833. Chronos predicts nine quantiles and the comparison fed it the median, then a
coverage that was tuned separately. This sweeps the quantile, a calibration scale and the coverage
together, on the tuning anchors only.

VN1 has no cost asymmetry — absolute error plus a bias penalty — so there the median is already the
right choice and the only question is calibration: the same sweep, tuned on the validation origin
and checked once on the final one.

Nothing here adds an arm: the folds, the policy and the score are the ones already pinned.

Usage:
    python scripts/chronos_grid.py --task vn2
    python scripts/chronos_grid.py --task vn1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn1 import data as vn1data  # noqa: E402
from vn1 import forecaster as vn1f  # noqa: E402
from vn1.score import score as vn1score  # noqa: E402
from vn2 import backtest  # noqa: E402
from vn2 import data as vn2data  # noqa: E402
from vn2 import forecaster, play  # noqa: E402


def float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def vn2_grid(quantiles: list[float], scales: list[float], coverages: list[float],
             anchors=backtest.TUNING_ANCHORS, calibrated: bool = False,
             model: str = "bolt", cross_learning: bool = False):
    sales, in_stock, master = vn2data.sales_full(), vn2data.in_stock(), vn2data.master()
    folds = backtest.make_folds(anchors)
    rows, fold_rows = [], []

    for quantile in quantiles:
        for scale in scales:
            # One forecast per history prefix, shared by every coverage: coverage reorders, it does
            # not re-forecast.
            if calibrated:
                forecaster_fn = backtest.cache_forecasts(forecaster.calibrated_chronos())
            elif model == "chronos2":
                forecaster_fn = backtest.cache_forecasts(
                    forecaster.chronos2(quantile=quantile, cross_learning=cross_learning)
                )
            else:
                forecaster_fn = backtest.cache_forecasts(
                    forecaster.chronos(quantile=quantile, scale=scale)
                )
            for coverage in coverages:
                policy = play.order_up_to_policy(forecaster_fn, coverage=coverage)
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
                    score = play.score(weekly)
                    scores.append(score)
                    fold_rows.append(
                        {"quantile": quantile, "scale": scale, "coverage": coverage,
                         "anchor": fold.anchor, "score": score}
                    )
                rows.append(
                    {
                        "quantile": quantile,
                        "scale": scale,
                        "coverage": coverage,
                        "mean": round(float(np.mean(scores)), 1),
                        "std": round(float(np.std(scores, ddof=1)), 1),
                    }
                )
                print(
                    f"q={quantile:<4} scale={scale:<4} coverage={coverage:<4} "
                    f"mean {rows[-1]['mean']:>8,.1f}  std {rows[-1]['std']:>7,.1f}",
                    flush=True,
                )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def vn1_grid(quantiles: list[float], scales: list[float]):
    sales, price, phase1 = vn1data.phase0_sales(), vn1data.phase0_price(), vn1data.phase1_sales()
    validation_history = sales.iloc[:, :-13]
    validation_actual = sales.iloc[:, -13:]

    rows = []
    best = None
    for quantile in quantiles:
        for scale in scales:
            forecaster_fn = vn1f.chronos(quantile=quantile, scale=scale)
            started = time.time()
            validation = vn1score(forecaster_fn(validation_history, price), validation_actual)
            final = None
            if best is None or validation < best[0]:
                final = vn1score(forecaster_fn(sales, price), phase1)
                best = (validation, quantile, scale, final)
            rows.append(
                {
                    "quantile": quantile,
                    "scale": scale,
                    "validation": round(validation, 4),
                    "final": round(final, 4) if final is not None else None,
                }
            )
            print(
                f"q={quantile:<4} scale={scale:<4} validation {validation:.4f}"
                + (f"  final {final:.4f}" if final is not None else "")
                + f"  ({time.time() - started:,.0f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["vn2", "vn1"])
    parser.add_argument("--quantiles", default="0.5,0.7,0.8,0.9")
    parser.add_argument("--scales", default="1,1.1")
    parser.add_argument("--coverages", default="3,3.5,4")
    parser.add_argument("--anchors", default="tuning", choices=["tuning", "evaluation"])
    parser.add_argument("--calibrated", action="store_true",
                        help="per-series causal bias correction instead of a global scale")
    parser.add_argument("--model", default="bolt", choices=["bolt", "chronos2"],
                        help="which Chronos checkpoint the VN2 grid uses")
    parser.add_argument("--cross-learning", action="store_true",
                        help="forecast all series in one joint pass (Chronos-2 only)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    quantiles, scales, coverages = (
        float_list(args.quantiles), float_list(args.scales), float_list(args.coverages)
    )
    if args.calibrated:  # calibration replaces the global levers; one configuration only
        quantiles, scales = [0.5], [1.0]
    anchor_sets = {"tuning": backtest.TUNING_ANCHORS, "evaluation": backtest.EVALUATION_ANCHORS}
    if args.task == "vn2":
        results, folds = vn2_grid(quantiles, scales, coverages, anchor_sets[args.anchors],
                                  calibrated=args.calibrated, model=args.model,
                                  cross_learning=args.cross_learning)
    else:
        results, folds = vn1_grid(quantiles, scales), None

    base = Path("results") / f"chronos-grid-{args.task}{'-' + args.tag if args.tag else ''}"
    base.parent.mkdir(exist_ok=True)
    results.to_csv(f"{base}.csv", index=False)
    if folds is not None:
        folds.to_csv(f"{base}-folds.csv", index=False)

    sort_column = "mean" if args.task == "vn2" else "validation"
    print("\nbest configurations (lower is better)")
    print(results.sort_values(sort_column).head(8).to_string(index=False))
    print(f"\nwrote {base}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
