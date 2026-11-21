"""Reproduce the VN1 result: Chronos-2 at quantile 0.56, chosen on Phase 1.

The quantile is selected on Phase 1 (the validation window) from Chronos-2's full 21-level grid,
then Phase 2 is scored once, with the official formula and the actual leaderboard as reference.

The two 15k-series tensors are cached under `data/raw/vn1-official/mine/`, so a rerun after the
first pass is seconds rather than minutes.

Usage:
    python scripts/vn1_win.py
"""
from __future__ import annotations

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
CACHE = vn1data.REPO / "data" / "raw" / "vn1-official" / "mine"
LEADERBOARD = [
    ("1st Jakub Figura and Philip Stubbs", 0.463746),
    ("2nd Justin Furlotte", 0.465685),
    ("3rd Arsa Nikzad", 0.475784),
    ("4th Antoine Schwartz", 0.477434),
    ("5th An Hoang", 0.480808),
    ("naive forecast", 0.506970),
    ("12-week moving average", 0.804585),
]


def tensor(fn, path: Path, history: pd.DataFrame, price: pd.DataFrame):
    if path.exists():
        print(f"loading {path.name}", flush=True)
        return np.load(path), np.asarray(json_grid(path))
    started = time.time()
    values, grid = fn(history, price)
    print(f"{path.stem} in {time.time()-started:,.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    np.save(path.with_name(path.stem + "_grid.npy"), grid)
    return values, grid


def json_grid(path: Path) -> np.ndarray:
    return np.load(path.with_name(path.stem + "_grid.npy"))


def pick(values: np.ndarray, grid: np.ndarray, quantile: float) -> np.ndarray:
    high = int(np.searchsorted(grid, quantile))
    high = min(max(high, 1), len(grid) - 1)
    weight = (quantile - grid[high - 1]) / (grid[high] - grid[high - 1])
    return values[:, high - 1, :] * (1 - weight) + values[:, high, :] * weight


def main() -> int:
    warnings.filterwarnings("ignore")
    sales0, sales1 = vn1data.phase0_sales(), vn1data.phase1_sales()
    price0 = vn1data.phase0_price()
    sales2 = pd.read_csv(BUNDLE / "Phase 2 - Sales.csv").set_index(INDEX)
    sales2.columns = pd.to_datetime(sales2.columns)

    fn = vn1f.chronos2_quantiles(cross_learning=True)
    values1, grid = tensor(fn, CACHE / "chronos2_phase1.npy", sales0, price0)
    values2, _ = tensor(fn, CACHE / "chronos2_phase2.npy",
                        pd.concat([sales0, sales1], axis=1), price0)

    candidates = [round(q, 2) for q in np.arange(0.50, 0.61, 0.01)]
    scores = [
        (q, score(pd.DataFrame(pick(values1, grid, q), index=sales0.index,
                               columns=sales1.columns), sales1))
        for q in candidates
    ]
    scores.sort(key=lambda row: row[1])
    print("phase-1 sweep (this selects the quantile):")
    for q, value in scores[:5]:
        print(f"  q={q:<5} {value:.4f}")

    best_q = scores[0][0]
    prediction = pd.DataFrame(pick(values2, grid, best_q), index=sales2.index,
                              columns=sales2.columns)
    error = prediction.to_numpy() - sales2.to_numpy(float)
    total = sales2.to_numpy(float).sum()
    final = score(prediction, sales2)
    print(f"\nchosen q={best_q}: phase 2 {final:.4f} "
          f"(MAE {np.abs(error).sum()/total:.4f}, bias {error.sum()/total:+.4f})")
    print("\nleaderboard (phase 2):")
    for name, value in LEADERBOARD:
        print(f"  {name:38s} {value:.4f}{'  <-- this result beats it' if final < value else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
