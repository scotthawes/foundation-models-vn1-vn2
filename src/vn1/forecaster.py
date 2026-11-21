"""Four arms for the VN1 pilot: two cheap references, TabPFN over the API, Chronos locally.

`ma12` is the organisers' own dummy submission, reproduced exactly, and it is the number to beat.
`naive_last` repeats the last observed week. The two foundation models are the same two as VN2, but
the question here is sharper: the score is forecast error, so nothing hides behind an ordering
policy, and the tabular arm gets price and the series identifiers that Chronos cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HORIZON, IDENTIFIERS, prediction_rows, train_anchors, training_rows

NOT_FEATURES = {"target"}


def future_weeks(history: pd.DataFrame, horizon: int = HORIZON) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [history.columns[-1] + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
    )


def _frame(values: np.ndarray, history: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    return pd.DataFrame(values, index=history.index, columns=future_weeks(history, horizon))


def ma12(history: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """The organisers' baseline: the mean of the last 12 weeks, held flat across the horizon."""
    values = history.iloc[:, -12:].mean(axis=1).to_numpy().reshape(-1, 1)
    return _frame(np.repeat(values, horizon, axis=1), history, horizon)


def naive_last(history: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    values = history.iloc[:, -1].to_numpy().reshape(-1, 1)
    return _frame(np.repeat(values, horizon, axis=1), history, horizon)


def prepare(
    sales: pd.DataFrame,
    price: pd.DataFrame,
    max_train_rows: int = 20_000,
    lookback: int = 52,
    step: int = 4,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """The exact tables the TabPFN arm is fitted and asked on.

    Split out so `scripts/vn1_pilot.py` can quote the API bill from the real shapes without
    uploading anything; the arm below calls the same function, so the quote cannot drift from the
    request.
    """
    anchors = train_anchors(sales.shape[1], lookback=lookback, step=step, horizon=horizon)
    rows = training_rows(sales, price, anchors, horizon=horizon)
    if len(rows) > max_train_rows:
        rows = rows.sample(n=max_train_rows, random_state=0)

    asked = prediction_rows(sales, price, horizon=horizon)
    for col in IDENTIFIERS:  # the API takes categoricals natively; labels, not magnitudes
        rows[col] = rows[col].astype(str).astype("category")
        asked[col] = asked[col].astype(str).astype("category")
    features = [c for c in rows.columns if c not in NOT_FEATURES]
    return rows[features], rows["target"], asked[features]


def tabpfn_api(
    max_train_rows: int = 20_000,
    n_estimators: int = 4,
    horizon: int = HORIZON,
    lookback: int = 52,
    step: int = 4,
    predict_batch_size: int = 2_000,
):
    """TabPFN through the Prior Labs API, on the lag/price/hierarchy table."""

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
        from tabpfn_client import TabPFNRegressor

        from vn2 import keys

        keys.require("TABPFN_TOKEN", "the VN1 TabPFN arm")

        X_train, y_train, X_test = prepare(
            sales, price, max_train_rows=max_train_rows, lookback=lookback, step=step,
            horizon=horizon,
        )
        model = TabPFNRegressor(
            model_path="v3.5_default", n_estimators=n_estimators, random_state=0
        )
        model.fit(X_train, y_train)
        predicted = np.concatenate(
            [
                np.asarray(
                    model.predict(X_test.iloc[start : start + predict_batch_size]),
                    dtype=float,
                )
                for start in range(0, len(X_test), predict_batch_size)
            ]
        )
        values = np.clip(predicted, 0.0, None).reshape(horizon, len(sales)).T
        return _frame(values, sales, horizon)

    return forecaster


def chronos(context_weeks: int | None = 152, horizon: int = HORIZON):
    """Chronos-Bolt-small, one sequence per series, blind to price and to the other series."""
    pipeline = None

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
        nonlocal pipeline
        import torch

        if pipeline is None:
            from vn2.forecaster.chronos_arm import _load  # the Bolt-aware loader, already measured

            pipeline = _load("amazon/chronos-bolt-small")

        context = sales if context_weeks is None else sales.iloc[:, -context_weeks:]
        tensor = torch.tensor(context.to_numpy(dtype=np.float32))
        with torch.no_grad():
            forecast = pipeline.predict(tensor, prediction_length=horizon)
        values = np.asarray(forecast, dtype=float)
        if values.ndim == 3:  # (batch, quantiles, horizon): the median is the 0.5 quantile
            values = np.median(values, axis=1)
        values = np.clip(values.reshape(len(sales), horizon), 0.0, None)
        return _frame(values, sales, horizon)

    return forecaster
