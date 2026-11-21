"""Calibrate a foundation model's forecast conditionally, then score it on VN1.

A global scale trades bias for MAE almost one for one, because it moves well-calibrated series as
much as badly calibrated ones. This fits a small model on rolling out-of-sample forecasts:

    actual ~ f(forecast, horizon, series level, zero-share, price, week-of-year)

so the correction can depend on *which kind* of series and week is being forecast. The foundation
model still produces the forecast; the calibrator only removes its systematic error.

Protocol: the calibrator and the model are fitted on rolling windows inside Phase 0, checked on
Phase 1, and scored on Phase 2 once. Phase 2 has been looked at in this project, so the result is
reported as exploratory, not blind.

Usage:
    python scripts/vn1_calibrate.py --model bolt --series 0
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

INDEX = ["Client", "Warehouse", "Product"]
HORIZON = 13
BUNDLE = vn1data.REPO / "data" / "raw" / "vn1-official" / "vn1-forecasting-competition"


def features(history: pd.DataFrame, forecast: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """One row per (series, horizon): the forecast plus the things that explain its bias."""
    level = history.iloc[:, -13:].mean(axis=1)
    zero_share = (history.iloc[:, -52:] == 0).mean(axis=1)
    last_price = price.iloc[:, -1]
    rows = []
    for h in range(1, HORIZON + 1):
        week = forecast.columns[h - 1]
        iso = float(pd.Timestamp(week).isocalendar().week)
        block = pd.DataFrame(
            {
                "forecast": forecast.iloc[:, h - 1].to_numpy(),
                "horizon": h,
                "level": level.to_numpy(),
                "zero_share": zero_share.to_numpy(),
                "price": last_price.to_numpy(),
                "iso_sin": np.sin(2 * np.pi * iso / 52),
                "iso_cos": np.cos(2 * np.pi * iso / 52),
            },
            index=history.index,
        )
        rows.append(block)
    out = pd.concat(rows)
    out["series"] = np.tile(np.arange(len(history)), HORIZON)
    return out.reset_index()


def fit_and_score(model, sales0, price0, sales1, history2, price2, sales2, origins):
    """Train on rolling Phase-0 windows, report Phase 1, then score Phase 2."""
    import lightgbm as lgb

    train_X, train_y = [], []
    for origin in origins:
        history = sales0.iloc[:, : origin + 1]
        forecast = model(history, price0.iloc[:, : origin + 1])
        frame = features(history, forecast, price0.iloc[:, : origin + 1])
        # features() is horizon-major (all series for h=1, then h=2, ...), so the target must be
        # flattened the same way; a naive reshape would pair every horizon with the wrong series.
        actual = sales0.iloc[:, origin + 1 : origin + HORIZON + 1].to_numpy().T.reshape(-1)
        frame["target"] = actual
        train_X.append(frame)
    train = pd.concat(train_X, ignore_index=True)
    feature_columns = ["forecast", "horizon", "level", "zero_share", "price", "iso_sin", "iso_cos"]

    learner = lgb.LGBMRegressor(
        objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, random_state=0, n_jobs=1,
    )
    learner.fit(train[feature_columns], train["target"])

    def predict(history, price):
        forecast = model(history, price)
        frame = features(history, forecast, price)
        values = np.clip(learner.predict(frame[feature_columns]), 0.0, None)
        # Undo the horizon-major order the features were built in.
        return pd.DataFrame(values.reshape(HORIZON, len(history)).T, index=history.index,
                            columns=forecast.columns)

    phase1 = predict(sales0, price0)
    phase2 = predict(history2, price2)
    return phase1, phase2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=0)
    parser.add_argument("--model", default="bolt", choices=["bolt", "chronos2"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    sales0, sales1 = vn1data.phase0_sales(), vn1data.phase1_sales()
    price0 = vn1data.phase0_price()
    sales2 = pd.read_csv(BUNDLE / "Phase 2 - Sales.csv").set_index(INDEX)
    sales2.columns = pd.to_datetime(sales2.columns)

    if args.series:
        keep = sales0.index[: args.series]
        sales0, sales1, price0, sales2 = (frame.loc[keep] for frame in (sales0, sales1, price0, sales2))
    history2 = pd.concat([sales0, sales1], axis=1)
    price2 = pd.concat([price0, price0.iloc[:, -1:].reindex(columns=sales1.columns, fill_value=0.0)],
                       axis=1)

    model = (
        vn1f.chronos() if args.model == "bolt"
        else vn1f.chronos2(cross_learning=True)
    )
    origins = list(range(100, len(sales0.columns) - HORIZON + 1, 8))
    phase1, phase2 = fit_and_score(model, sales0, price0, sales1, history2, price2, sales2, origins)

    for name, prediction, actual in (("phase 1", phase1, sales1), ("phase 2", phase2, sales2)):
        error = prediction.to_numpy() - actual.to_numpy(float)
        total = actual.to_numpy(float).sum()
        print(f"{args.model} + calibration, {name}: {score(prediction, actual):.4f}  "
              f"(MAE {np.abs(error).sum() / total:.4f}, bias {error.sum() / total:+.4f})")
    print("reference: winner 0.4637, bolt median phase2 0.5158, naive 0.5070")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
