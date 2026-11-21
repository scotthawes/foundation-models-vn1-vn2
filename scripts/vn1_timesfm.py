"""Score TimesFM-3 on VN1 at full population, with the quantile chosen on Phase 1.

TimesFM-3 is the GIFT-Eval leader among zero-shot foundation models and had the best MAE of the
three tested on a 500-series subset, but a worse bias than Chronos-2 — the same problem Chronos-2
solved by taking q=0.55 instead of the median. The nine quantiles come back from one model pass, so
this script computes the tensor once per phase and sweeps q for free.

Protocol: q is chosen on Phase 1 (the validation window), Phase 2 is scored once for that q, and the
result is reported as exploratory because Phase 2 has been looked at in this project.

Usage:
    python scripts/vn1_timesfm.py --series 2000
    python scripts/vn1_timesfm.py --series 0
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


def point(tensor: np.ndarray, quantile: float, index: pd.Index, columns) -> pd.DataFrame:
    levels = np.linspace(0.1, 0.9, tensor.shape[1])
    i = int(np.argmin(np.abs(levels - quantile)))
    return pd.DataFrame(tensor[:, i, :], index=index, columns=columns)


def load_or_build(fn, path: Path, history: pd.DataFrame, price: pd.DataFrame, label: str):
    """Reuse a saved quantile tensor if it exists — a full-population pass is ~11 minutes."""
    if path.exists():
        print(f"loading {label} from {path.name}", flush=True)
        return np.load(path)
    started = time.time()
    tensor = fn(history, price)
    print(f"{label} in {time.time()-started:,.0f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor)
    return tensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=2_000, help="0 means all series")
    parser.add_argument("--quantiles", default="0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--scales", default="1,1.05,1.1,1.15,1.2")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    sales0, sales1 = vn1data.phase0_sales(), vn1data.phase1_sales()
    price0 = vn1data.phase0_price()
    sales2 = pd.read_csv(BUNDLE / "Phase 2 - Sales.csv").set_index(INDEX)
    sales2.columns = pd.to_datetime(sales2.columns)

    if args.series:
        keep = sales0.index[: args.series]
        sales0, sales1, price0, sales2 = (f.loc[keep] for f in (sales0, sales1, price0, sales2))
    quantiles = [float(q) for q in args.quantiles.split(",")]
    tensor_fn = vn1f.timesfm3_quantiles()
    cache = vn1data.REPO / "data" / "raw" / "vn1-official" / "mine"
    tag = "all" if args.series == 0 else str(args.series)

    tensor1 = load_or_build(tensor_fn, cache / f"timesfm_phase1_{tag}.npy", sales0, price0,
                            "phase-1 tensor")

    rows = []
    for q in quantiles:
        prediction = point(tensor1, q, sales0.index, sales1.columns)
        for scale in [float(s) for s in args.scales.split(",")]:
            rows.append((q, scale, score(prediction * scale, sales1)))
    rows.sort(key=lambda r: r[2])
    print("phase-1 scores (this selects q and scale):")
    for q, scale, value in rows[:6]:
        print(f"  q={q:<5} scale={scale:<5} {value:.4f}")

    best_q, best_scale, best_value = rows[0]
    tensor2 = load_or_build(tensor_fn, cache / f"timesfm_phase2_{tag}.npy",
                            pd.concat([sales0, sales1], axis=1), price0, "phase-2 tensor")
    prediction = point(tensor2, best_q, sales2.index, sales2.columns) * best_scale
    error = prediction.to_numpy() - sales2.to_numpy(float)
    total = sales2.to_numpy(float).sum()
    print(f"\nchosen q={best_q} scale={best_scale} (phase1 {best_value:.4f})")
    print(f"phase 2 {score(prediction, sales2):.4f}  "
          f"(MAE {np.abs(error).sum()/total:.4f}, bias {error.sum()/total:+.4f})")
    print("reference: winner 0.4637, 2nd 0.4657, 3rd 0.4758, 5th 0.4808; "
          "chronos-2 q0.55 0.4698")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
