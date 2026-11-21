"""Score Chronos-2 on VN1 — the newer checkpoint, with cross-learning and price covariates.

Bolt-small already matches the winner's MAE and loses on bias. Chronos-2 is a different model, not
a bigger one, and it exposes two things Bolt cannot: `cross_learning` (share information across all
series in one pass) and price as a past covariate. Both are aimed at the same defect.

Protocol: Phase 1 is the validation window (forecast Phase 0, score Phase 1). Phase 2 is scored for
the variant that wins on Phase 1, and reported as exploratory — Phase 2 has been looked at in this
project already, so no result here is a clean blind test.

Usage:
    python scripts/vn1_chronos2.py --series 2000 --variants base,cross,price
    python scripts/vn1_chronos2.py --series 0 --variants base
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

VARIANTS = {
    "base": dict(),
    "cross": dict(cross_learning=True),
    "price": dict(use_price=True),
    "price_cross": dict(use_price=True, cross_learning=True),
}


def phase2_actuals() -> pd.DataFrame:
    table = pd.read_csv(BUNDLE / "Phase 2 - Sales.csv").set_index(INDEX)
    table.columns = pd.to_datetime(table.columns)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=2_000, help="0 means all series")
    parser.add_argument("--variants", default="base,cross,price")
    parser.add_argument("--quantiles", default="0.5")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    sales0, sales1 = vn1data.phase0_sales(), vn1data.phase1_sales()
    price0 = vn1data.phase0_price()
    sales2 = phase2_actuals()

    if args.series:
        keep = sales0.index[: args.series]
        sales0, sales1 = sales0.loc[keep], sales1.loc[keep]
        price0 = price0.loc[keep]
        sales2 = sales2.loc[keep]
    history2 = pd.concat([sales0, sales1], axis=1)
    price2 = pd.concat([price0, price0.iloc[:, -1:].reindex(columns=sales1.columns,
                                                           fill_value=0.0)], axis=1)

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    quantiles = [float(q) for q in args.quantiles.split(",")]
    rows = []

    for name in names:
        for q in quantiles:
            forecaster = vn1f.chronos2(quantile=q, batch_size=args.batch_size, **VARIANTS[name])
            started = time.time()
            validation = score(forecaster(sales0, price0), sales1)
            elapsed = time.time() - started
            rows.append({"variant": name, "quantile": q, "phase1": round(validation, 4),
                         "seconds": round(elapsed, 1)})
            print(f"{name:12s} q={q:<5} phase1 {validation:.4f}  ({elapsed:,.0f}s)", flush=True)

    results = pd.DataFrame(rows).sort_values("phase1")
    print(results.to_string(index=False))

    best = results.iloc[0]
    variant = VARIANTS[best["variant"]]
    forecaster = vn1f.chronos2(quantile=float(best["quantile"]),
                               batch_size=args.batch_size, **variant)
    started = time.time()
    prediction = forecaster(history2, price2)
    final = score(prediction, sales2)
    error = prediction.to_numpy() - sales2.to_numpy(float)
    total = sales2.to_numpy(float).sum()
    print(f"\nbest-on-phase1: {best['variant']} q={best['quantile']}  "
          f"phase2 {final:.4f} (MAE {np.abs(error).sum()/total:.4f}, "
          f"bias {error.sum()/total:+.4f}) in {time.time()-started:,.0f}s")
    print("reference: winner 0.4637, 5th 0.4808, naive 0.5070, bolt median 0.5158")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
