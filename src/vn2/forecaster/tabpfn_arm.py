"""TabPFN arm: demand forecasting as tabular regression.

At each round the history becomes a table of (Store x Product, week ahead) rows and TabPFN predicts
the eight weeks ahead. The model is refit every round, because every round adds a week of history
and the training rows are drawn from that history.

Two decisions worth naming:

- **The training set is a sample, not the pool.** A year of pseudo-anchors at 599 rows and eight
  horizons is a few hundred thousand rows, far past the context TabPFN was trained for. A seeded
  sample of `max_train_rows` keeps the fit inside the model's regime and makes a run repeatable.
- **Censored targets are dropped, not filled.** See `vn2.features`. Filling them would hand the
  model a target known to be below demand and would bias every forecast down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import week_offset
from ..features import HORIZONS, build_rows, prediction_rows, train_anchors
from ..play import ForecasterFn

#: Columns that identify a row or record where it came from; none of them is a feature.
META_COLUMNS = ("Store", "Product", "anchor", "target")

DEFAULT_MAX_TRAIN_ROWS = 10_000  # TabPFN's context limit without ignoring the pretraining limits
DEFAULT_LOOKBACK = 104           # weeks of pseudo-anchors to draw training rows from
DEFAULT_STEP = 4                 # weeks between pseudo-anchors
DEFAULT_PREDICT_BATCH = 1_000    # rows per predict() call; 4,792 at once exhausts MPS on this machine


def tabpfn(
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    lookback: int = DEFAULT_LOOKBACK,
    step: int = DEFAULT_STEP,
    seed: int = 0,
    n_estimators: int = 4,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH,
) -> ForecasterFn:
    """A forecaster that fits TabPFN on the history and predicts the eight weeks ahead.

    `predict_batch_size` exists because of a measured failure: one call over all 4,792 test rows
    exhausts the 3.73 GB MPS budget on this machine. Batching bounds the peak without touching the
    numbers — each row is scored against the same fitted model either way.
    """

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        from tabpfn import TabPFNRegressor

        anchors = train_anchors(len(sales.columns), lookback=lookback, step=step)
        rows = build_rows(sales, in_stock_df, master_df, anchors)
        rows = rows.dropna(subset=["target"])  # a censored week is not a target
        if len(rows) > max_train_rows:
            rows = rows.sample(n=max_train_rows, random_state=seed)

        features = [c for c in rows.columns if c not in META_COLUMNS]
        model = TabPFNRegressor(n_estimators=n_estimators, random_state=seed)
        model.fit(rows[features].to_numpy(), rows["target"].to_numpy())

        asked = prediction_rows(sales, in_stock_df, master_df, horizons=HORIZONS)[features].to_numpy()
        predicted = np.concatenate(
            [
                np.asarray(model.predict(asked[start : start + predict_batch_size]), dtype=float)
                for start in range(0, len(asked), predict_batch_size)
            ]
        )
        predicted = np.clip(predicted, 0.0, None)  # demand cannot be negative

        values = predicted.reshape(len(HORIZONS), len(sales)).T  # (row, horizon)
        future = pd.DatetimeIndex(
            [week_offset(sales, len(sales.columns) - 1 + h) for h in HORIZONS]
        )
        return pd.DataFrame(values, index=sales.index, columns=future)

    return forecaster
