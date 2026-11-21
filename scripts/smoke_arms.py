"""Smoke-test the two hosted arms and quote the TabPFN bill before any back-test is launched.

Three things happen, in order, and each is skipped if its key is absent:

1. **Quote.** `estimate_cost` prices one TabPFN fit+predict from dataset dimensions alone. It does
   not upload data and spends no tokens, so this is the cheap way to learn whether the free tier
   (5M tokens/day, 20M/month) can carry a back-test of ~78 fits.
2. **TabPFN.** One real fit and predict at the shape the back-test uses: 2,000 training rows and
   4,792 prediction rows. Prints the server-side timings.
3. **TimeGPT.** One real forecast of the 599 series, eight weeks ahead, from the competition
   history. Prints wall time and the shape that came back.

Run this before `scripts/backtest.py`: it fails in seconds on a bad key, instead of an hour into a
run.

Usage:
    python scripts/smoke_arms.py [--skip tabpfn] [--skip timegpt]
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

from vn2 import data as vn2data  # noqa: E402
from vn2 import keys  # noqa: E402
from vn2.features import HORIZONS, build_rows, prediction_rows, train_anchors  # noqa: E402
from vn2.forecaster import timegpt_arm  # noqa: E402
from vn2.forecaster.tabpfn_arm import META_COLUMNS  # noqa: E402

MAX_TRAIN_ROWS = 2_000
ESTIMATORS = 4


def shapes():
    """The one training table and one prediction table every arm is asked about."""
    sales = vn2data.sales_full()
    in_stock = vn2data.in_stock()
    master = vn2data.master()
    history = sales.iloc[:, : vn2data.COMPETITION_ANCHOR + 1]

    anchors = train_anchors(len(history.columns), lookback=104, step=4)
    rows = build_rows(sales, in_stock, master, anchors).dropna(subset=["target"])
    rows = rows.sample(n=min(MAX_TRAIN_ROWS, len(rows)), random_state=0)
    asked = prediction_rows(history, in_stock, master, horizons=HORIZONS)
    features = [c for c in rows.columns if c not in META_COLUMNS]
    return history, in_stock, master, rows[features], rows["target"], asked[features]


def quote(X_train, X_test) -> None:
    from tabpfn_client import estimate_cost

    estimate = estimate_cost(
        X_train, X_test, model_version="v3.5", n_estimators=ESTIMATORS
    )
    print(f"  quote: {estimate.estimated_cost:,.0f} tokens for one fit+predict "
          f"(pricing {estimate.pricing_version})")
    print(f"  free tier: 5,000,000 tokens/day → "
          f"~{5_000_000 / max(float(estimate.estimated_cost), 1):.0f} rounds per day")


def tabpfn_smoke(history, in_stock, master, X_train, y_train, X_test) -> None:
    from tabpfn_client import TabPFNRegressor

    model = TabPFNRegressor(model_path="v3.5_default", n_estimators=ESTIMATORS, random_state=0)
    started = time.time()
    model.fit(X_train, y_train)
    fitted = time.time() - started
    started = time.time()
    predicted = np.concatenate(
        [
            np.asarray(model.predict(X_test.iloc[start : start + 1_000]), dtype=float)
            for start in range(0, len(X_test), 1_000)
        ]
    )
    predicted_at = time.time() - started
    print(f"  fit {fitted:,.1f}s  predict {predicted_at:,.1f}s  rows={len(predicted):,}  "
          f"mean={predicted.mean():.2f} max={predicted.max():.1f}")
    if hasattr(model, "get_timings"):
        print(f"  server timings: {model.get_timings()}")


def timegpt_smoke(history, in_stock, master) -> None:
    forecaster = timegpt_arm.timegpt()
    state = vn2data.initial_state()
    started = time.time()
    forecast = forecaster(history, in_stock[history.columns], master, state)
    elapsed = time.time() - started
    print(f"  {elapsed:,.1f}s  shape={forecast.shape}  "
          f"{forecast.columns[0].date()}..{forecast.columns[-1].date()}  "
          f"mean={forecast.to_numpy().mean():.2f} max={forecast.to_numpy().max():.1f}")
    assert forecast.shape == (len(history), len(HORIZONS)), "TimeGPT returned the wrong shape"
    assert not forecast.isna().any().any(), "TimeGPT returned a missing cell"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", action="append", default=[], choices=["tabpfn", "timegpt"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    keys.load()
    history, in_stock, master, X_train, y_train, X_test = shapes()
    print(f"training rows {X_train.shape} · prediction rows {X_test.shape}")
    ran = 0

    if "tabpfn" not in args.skip:
        print("\nTabPFN (Prior Labs API)")
        try:
            keys.require("TABPFN_TOKEN", "this smoke test")
            quote(X_train, X_test)
            tabpfn_smoke(history, in_stock, master, X_train, y_train, X_test)
            ran += 1
        except RuntimeError as exc:
            print(f"  skipped: {exc}")

    if "timegpt" not in args.skip:
        print("\nTimeGPT (Nixtla API)")
        try:
            keys.require("NIXTLA_API_KEY", "this smoke test")
            timegpt_smoke(history, in_stock, master)
            ran += 1
        except RuntimeError as exc:
            print(f"  skipped: {exc}")

    print("\ndone." if ran else "\nno arm ran: every one was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
