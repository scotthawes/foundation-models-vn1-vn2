"""Find a foundation-model configuration that wins VN1, selected on rolling windows only.

The competition score is `MAE% + |Bias%|`. Chronos's median is MAE-optimal per series but biased low
in aggregate, and a global scale trades the two almost one for one. The distribution offers a
sharper instrument: choosing a quantile `q > 0.5` shifts each series by its own skew rather than by
a common factor, and blending with the naive forecast cancels part of whatever bias remains.

Every choice here is made on **rolling windows inside Phase 0** — eight origins, each forecasting 13
weeks from the history before it. Phase 2 is scored once, at the end, and only for the configuration
the rolling windows selected. The quantile tensor is computed once per origin, so sweeping q costs
nothing beyond the model run.

Usage:
    python scripts/vn1_strategy.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn1 import data as vn1data  # noqa: E402
from vn1 import forecaster as vn1f  # noqa: E402
from vn1.score import score  # noqa: E402
from vn2.forecaster.chronos_arm import pick_quantile  # noqa: E402

INDEX = ["Client", "Warehouse", "Product"]
BUNDLE = vn1data.REPO / "data" / "raw" / "vn1-official" / "vn1-forecasting-competition"
HORIZON = 13


def flat(history: pd.DataFrame) -> pd.DataFrame:
    values = history.iloc[:, -1].to_numpy().reshape(-1, 1)
    weeks = [history.columns[-1] + pd.Timedelta(weeks=k) for k in range(1, HORIZON + 1)]
    return pd.DataFrame(np.repeat(values, HORIZON, axis=1), index=history.index, columns=weeks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantiles", default="0.5,0.55,0.6,0.65")
    parser.add_argument("--weights", default="0.5,0.7,1.0")
    parser.add_argument("--step", type=int, default=8)
    args = parser.parse_args()

    if not BUNDLE.exists():
        raise SystemExit(f"{BUNDLE} is missing; run scripts/fetch_vn1_official.sh")

    warnings.filterwarnings("ignore")
    qs = [float(q) for q in args.quantiles.split(",")]
    ws = [float(w) for w in args.weights.split(",")]

    sales0, sales1 = vn1data.phase0_sales(), vn1data.phase1_sales()
    price0 = vn1data.phase0_price()
    sales2 = pd.read_csv(BUNDLE / "Phase 2 - Sales.csv").set_index(INDEX)
    sales2.columns = pd.to_datetime(sales2.columns)

    quantiles_fn = vn1f.chronos_quantiles()
    origins = list(range(100, len(sales0.columns) - HORIZON + 1, args.step))
    results = {(_q, _w): [] for _q in qs for _w in ws}

    for origin in origins:
        history = sales0.iloc[:, : origin + 1]
        tensor = quantiles_fn(history, price0.iloc[:, : origin + 1])
        naive = flat(history).to_numpy()
        actual = sales0.iloc[:, origin + 1 : origin + HORIZON + 1]

        for q in qs:
            chronos = pd.DataFrame(pick_quantile(tensor, q), index=history.index,
                                   columns=actual.columns).to_numpy()
            for w in ws:
                blended = pd.DataFrame(w * chronos + (1 - w) * naive, index=history.index,
                                       columns=actual.columns)
                results[(q, w)].append(score(blended, actual))
        print(f"origin {origin}: done", flush=True)

    ranked = sorted((float(np.mean(v)), k) for k, v in results.items())
    print("\nrolling-window mean score (this selects the configuration)")
    for mean, (q, w) in ranked[:8]:
        print(f"  quantile {q:<5} chronos share {w:<5} {mean:.4f}")

    best_q, best_w = ranked[0][1]
    history = pd.concat([sales0, sales1], axis=1)
    tensor = quantiles_fn(history, pd.concat([price0, price0.iloc[:, -1:]], axis=1))
    naive = flat(history).to_numpy()
    chronos = pd.DataFrame(pick_quantile(tensor, best_q), index=history.index,
                           columns=sales2.columns).to_numpy()
    chosen = pd.DataFrame(best_w * chronos + (1 - best_w) * naive, index=history.index,
                          columns=sales2.columns)

    error = chosen.to_numpy() - sales2.to_numpy(float)
    total = sales2.to_numpy(float).sum()
    print(f"\nCHOSEN quantile {best_q}, chronos share {best_w}")
    print(f"Phase 2 score {score(chosen, sales2):.4f}  "
          f"(MAE {np.abs(error).sum() / total:.4f}, bias {error.sum() / total:+.4f})")
    print(f"reference: winner 0.4637, 5th 0.4808, naive 0.5070, moving average 0.8046")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
