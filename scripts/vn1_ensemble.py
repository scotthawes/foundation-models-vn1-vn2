"""Blend Chronos-2 and TimesFM-3, choosing the weight on Phase 1 and scoring Phase 2 once.

The two models are the 3rd- and 1st-ranked zero-shot foundation models on GIFT-Eval, they come from
different architectures, and on VN1 both show the same failure — accurate but biased low. Averaging
two differently-wrong forecasts usually reduces both terms of `MAE% + |Bias%|`; the weight is chosen
on the validation window, not on the scored one.

Chronos-2 uses cross-learning at quantile 0.55 (selected on Phase 1 in the Chronos-2 run). TimesFM-3's
quantile is selected on Phase 1 here.

Usage:
    python scripts/vn1_ensemble.py --series 2000
    python scripts/vn1_ensemble.py --series 0
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn1 import data as vn1data  # noqa: E402
from vn1 import forecaster as vn1f  # noqa: E402
from vn1.score import score  # noqa: E402

INDEX = ["Client", "Warehouse", "Product"]
BUNDLE = vn1data.REPO / "data" / "raw" / "vn1-official" / "vn1-forecasting-competition"
HORIZON = 13


def point_from_tensor(tensor: np.ndarray, quantile: float, index, columns) -> pd.DataFrame:
    levels = np.linspace(0.1, 0.9, tensor.shape[1])
    i = int(np.argmin(np.abs(levels - quantile)))
    return pd.DataFrame(tensor[:, i, :], index=index, columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=2_000)
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--tf-quantiles", default="0.5,0.55,0.6,0.65")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    sales0, sales1 = vn1data.phase0_sales(), vn1data.phase1_sales()
    price0 = vn1data.phase0_price()
    sales2 = pd.read_csv(BUNDLE / "Phase 2 - Sales.csv").set_index(INDEX)
    sales2.columns = pd.to_datetime(sales2.columns)
    if args.series:
        keep = sales0.index[: args.series]
        sales0, sales1, price0, sales2 = (f.loc[keep] for f in (sales0, sales1, price0, sales2))
    history2 = pd.concat([sales0, sales1], axis=1)
    weights = [float(w) for w in args.weights.split(",")]

    started = time.time()
    c2 = vn1f.chronos2(cross_learning=True, quantile=0.55)
    c2_phase1, c2_phase2 = c2(sales0, price0), c2(history2, price0)
    print(f"chronos-2 in {time.time()-started:,.0f}s", flush=True)

    started = time.time()
    tf_fn = vn1f.timesfm3_quantiles()
    tf1 = tf_fn(sales0, price0)
    tf2 = tf_fn(history2, price0)
    print(f"timesfm-3 tensors in {time.time()-started:,.0f}s", flush=True)

    best = None
    for q in [float(q) for q in args.tf_quantiles.split(",")]:
        tf_phase1 = point_from_tensor(tf1, q, sales0.index, sales1.columns)
        for w in weights:
            blended = w * c2_phase1 + (1 - w) * tf_phase1
            value = score(blended, sales1)
            if best is None or value < best[0]:
                best = (value, q, w)
            print(f"  tf q={q:<5} c2 weight={w:<5} phase1 {value:.4f}", flush=True)

    value, q, w = best
    tf_phase2 = point_from_tensor(tf2, q, sales2.index, sales2.columns)
    blended = w * c2_phase2 + (1 - w) * tf_phase2
    error = blended.to_numpy() - sales2.to_numpy(float)
    total = sales2.to_numpy(float).sum()
    print(f"\nchosen tf q={q}, chronos-2 weight {w} (phase1 {value:.4f})")
    print(f"phase 2 {score(blended, sales2):.4f}  "
          f"(MAE {np.abs(error).sum()/total:.4f}, bias {error.sum()/total:+.4f})")
    print("reference: winner 0.4637, 2nd 0.4657, 3rd 0.4758, chronos-2 alone 0.4698")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
