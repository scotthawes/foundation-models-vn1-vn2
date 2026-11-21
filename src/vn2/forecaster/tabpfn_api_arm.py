"""TabPFN arm, run through Prior Labs' hosted API instead of on this machine.

Why the API: measured here, one 2,000-row fit with 4 estimators takes 531.7 s, and 4,000 rows
exhausts the 3.73 GB MPS budget outright; the fit itself takes 0.3 s and the rest is inference over
the 4,792 prediction rows, so the cost is the model's attention over the test set, not training.
The API runs the same in-context model on server hardware, and the binding constraint becomes the
free tier (5M tokens/day) rather than the MPS allocator.

Two differences from the local arm are recorded rather than hidden:

- **The model is the server's.** `model_path="v3.5_default"` pins the version by name; it is
  TabPFN-3.5-Plus, not the open weights the local `tabpfn` package ships. Predictions are not
  byte-identical to the local arm's, which is why `docs/method.md` names the version.
- **Fits are billable.** `estimate_cost` quotes a fit without uploading data; the arms should be
  quoted before a full back-test is launched, not discovered halfway through one.

Everything above the estimator is shared with `tabpfn_arm`: the same feature table, the same
censoring rule, the same eight horizons, so the two differ only in where the arithmetic runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import keys
from ..data import week_offset
from ..features import HORIZONS, build_rows, prediction_rows, train_anchors
from ..play import ForecasterFn
from .tabpfn_arm import META_COLUMNS

DEFAULT_MODEL = "v3.5_default"  # the server's named model; "auto" follows whatever ships next
DEFAULT_MAX_TRAIN_ROWS = 2_000  # the largest context measured to fit the local MPS budget
DEFAULT_LOOKBACK = 104
DEFAULT_STEP = 4
DEFAULT_PREDICT_BATCH = 1_000  # rows per predict() call; keeps each request small
DEFAULT_ESTIMATORS = 4


def tabpfn_api(
    model_path: str = DEFAULT_MODEL,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    lookback: int = DEFAULT_LOOKBACK,
    step: int = DEFAULT_STEP,
    seed: int = 0,
    n_estimators: int = DEFAULT_ESTIMATORS,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH,
) -> ForecasterFn:
    """A forecaster that fits TabPFN on the history through the Prior Labs API."""

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        from tabpfn_client import TabPFNRegressor

        keys.require("TABPFN_TOKEN", "the TabPFN API arm")

        anchors = train_anchors(len(sales.columns), lookback=lookback, step=step)
        rows = build_rows(sales, in_stock_df, master_df, anchors)
        rows = rows.dropna(subset=["target"])  # a censored week is not a target
        if len(rows) > max_train_rows:
            rows = rows.sample(n=max_train_rows, random_state=seed)

        features = [c for c in rows.columns if c not in META_COLUMNS]
        model = TabPFNRegressor(
            model_path=model_path, n_estimators=n_estimators, random_state=seed
        )
        model.fit(rows[features], rows["target"])

        asked = prediction_rows(sales, in_stock_df, master_df, horizons=HORIZONS)[features]
        predicted = np.concatenate(
            [
                np.asarray(
                    model.predict(asked.iloc[start : start + predict_batch_size]), dtype=float
                )
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
