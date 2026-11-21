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
    trend: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """The exact tables the TabPFN arm is fitted and asked on.

    Split out so `scripts/vn1_pilot.py` can quote the API bill from the real shapes without
    uploading anything; the arm below calls the same function, so the quote cannot drift from the
    request.
    """
    anchors = train_anchors(sales.shape[1], lookback=lookback, step=step, horizon=horizon)
    rows = training_rows(sales, price, anchors, horizon=horizon, trend=trend)
    if len(rows) > max_train_rows:
        rows = rows.sample(n=max_train_rows, random_state=0)

    asked = prediction_rows(sales, price, horizon=horizon, trend=trend)
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
    trend: bool = False,
    target: str = "raw",
):
    """TabPFN through the Prior Labs API, on the lag/price/hierarchy table.

    `target` is the untested knob the pilot left open. VN1's targets are 72% zeros and span orders
    of magnitude, so two transforms are offered next to the raw column:

    - `log1p` — fit and predict in log space, invert on the way out;
    - `ratio` — divide each target by its series' 13-week mean (a scale-free multiple), multiply
      back at prediction time; a series with no recent demand is predicted at zero rather than
      given an arbitrary multiple.
    """
    if target not in ("raw", "log1p", "ratio"):
        raise ValueError(f"unknown target transform {target!r}")

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
        from tabpfn_client import TabPFNRegressor

        from vn2 import keys

        keys.require("TABPFN_TOKEN", "the VN1 TabPFN arm")

        X_train, y_train, X_test = prepare(
            sales, price, max_train_rows=max_train_rows, lookback=lookback, step=step,
            horizon=horizon, trend=trend,
        )
        y = y_train.to_numpy(dtype=float)
        base_test = None
        if target == "log1p":
            y = np.log1p(y)
        elif target == "ratio":
            base_train = X_train["mean_13"].to_numpy(dtype=float)
            base_test = X_test["mean_13"].to_numpy(dtype=float)
            alive = base_train > 0.5
            y = np.where(alive, y / np.where(alive, base_train, 1.0), 0.0)

        model = TabPFNRegressor(
            model_path="v3.5_default", n_estimators=n_estimators, random_state=0
        )
        model.fit(X_train, y)
        predicted = np.concatenate(
            [
                np.asarray(
                    model.predict(X_test.iloc[start : start + predict_batch_size]),
                    dtype=float,
                )
                for start in range(0, len(X_test), predict_batch_size)
            ]
        )
        if target == "log1p":
            predicted = np.expm1(predicted)
        elif target == "ratio":
            predicted = predicted * np.where(base_test > 0.5, base_test, 0.0)
        values = np.clip(predicted, 0.0, None).reshape(horizon, len(sales)).T
        return _frame(values, sales, horizon)

    return forecaster


def chronos_quantiles(context_weeks: int | None = 152, horizon: int = HORIZON,
                      batch_size: int = 2_000):
    """The raw nine-quantile forecast, `(series, quantiles, horizon)`, batched across series.

    Computing the tensor once per origin is what makes a quantile sweep cheap: the choice of
    quantile is a selection from this array, not another model run.
    """
    pipeline = None

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> np.ndarray:
        nonlocal pipeline
        import torch

        if pipeline is None:
            from vn2.forecaster.chronos_arm import _load

            pipeline = _load("amazon/chronos-bolt-small")

        context = sales if context_weeks is None else sales.iloc[:, -context_weeks:]
        pieces = []
        for start in range(0, len(context), batch_size):
            chunk = context.iloc[start : start + batch_size]
            tensor = torch.tensor(chunk.to_numpy(dtype=np.float32))
            with torch.no_grad():
                forecast = pipeline.predict(tensor, prediction_length=horizon)
            values = np.asarray(forecast, dtype=float)
            if values.ndim != 3:
                raise ValueError("this checkpoint did not return quantiles; use a Bolt model")
            pieces.append(values)
        return np.clip(np.concatenate(pieces, axis=0), 0.0, None)

    return forecaster


def timesfm3(horizon: int = HORIZON, quantile: float = 0.5, context_weeks: int | None = None):
    """TimesFM-3 — the GIFT-Eval leader among zero-shot foundation models.

    Ranked ahead of Chronos-2 and Toto-2.0 on 97 datasets by mean per-dataset rank
    (`docs/vn1-pilot.md`). Its API takes a list of 1-D contexts and returns a point forecast plus
    nine quantiles; `quantile` picks from those, so the same score-aware shifts can be applied.
    """
    model = None

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
        nonlocal model
        if model is None:
            from timesfm import TimesFM3Forecaster

            model = TimesFM3Forecaster.from_pretrained(device="mps")

        context = sales if context_weeks is None else sales.iloc[:, -context_weeks:]
        outputs = list(
            model.predict_batch(
                [row for row in context.to_numpy(dtype=float)],
                horizon=horizon,
                return_quantiles=True,
                make_positive=True,
            )
        )
        if quantile == 0.5:
            values = np.stack([out.forecast for out in outputs])
        else:
            levels = np.linspace(0.1, 0.9, 9)
            index = int(np.argmin(np.abs(levels - quantile)))
            values = np.stack([np.asarray(out.quantiles)[:, index] for out in outputs])
        weeks = [sales.columns[-1] + pd.Timedelta(weeks=k) for k in range(1, horizon + 1)]
        return pd.DataFrame(np.clip(values, 0.0, None), index=sales.index, columns=weeks)

    return forecaster


def chronos2(context_weeks: int | None = None, horizon: int = HORIZON, batch_size: int = 128,
             cross_learning: bool = False, use_price: bool = False, quantile: float = 0.5):
    """Chronos-2 (478 MB) — the newer checkpoint, with optional price covariates.

    Two things here are not available in Bolt: `cross_learning` lets the model share information
    across the 15,053 series in one pass, and `use_price` passes the weekly price as a past
    covariate. Both attack the same failure the bolt model showed on VN1 — a point forecast that is
    accurate but systematically low.
    """
    pipeline = None

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
        nonlocal pipeline
        import torch

        from vn2.forecaster.chronos_arm import _load, pick_quantile

        if pipeline is None:
            pipeline = _load("amazon/chronos-2")  # the base loader dispatches on the config

        context = sales if context_weeks is None else sales.iloc[:, -context_weeks:]
        values = context.to_numpy(dtype=np.float32)  # (series, weeks)

        if use_price:
            prices = price.reindex(columns=context.columns).to_numpy(dtype=np.float32)
            inputs = [
                {"target": values[i], "past_covariates": {"price": prices[i]}}
                for i in range(len(values))
            ]
        else:
            inputs = values[:, None, :]  # (series, 1, weeks)

        forecast = pipeline.predict(inputs, prediction_length=horizon, batch_size=batch_size,
                                    cross_learning=cross_learning)
        # Each element is (1, n_quantiles, horizon); Chronos-2 emits 21 levels, not Bolt's 9.
        stacked = np.stack([np.asarray(t, dtype=float).squeeze(0) for t in forecast])
        grid = np.asarray(pipeline.quantiles, dtype=float)
        q = float(np.clip(quantile, grid[0], grid[-1]))
        high = int(np.searchsorted(grid, q))
        if grid[high] == q or high == 0:
            point = stacked[:, high, :]
        else:
            weight = (q - grid[high - 1]) / (grid[high] - grid[high - 1])
            point = stacked[:, high - 1, :] * (1 - weight) + stacked[:, high, :] * weight
        weeks = [sales.columns[-1] + pd.Timedelta(weeks=k) for k in range(1, horizon + 1)]
        return pd.DataFrame(np.clip(point, 0.0, None), index=sales.index, columns=weeks)

    return forecaster


def chronos(context_weeks: int | None = 152, horizon: int = HORIZON, batch_size: int = 2_000,
            quantile: float = 0.5, scale: float = 1.0):
    """Chronos-Bolt-small, one sequence per series, blind to price and to the other series.

    The batch is bounded: 15,053 series in one tensor would not fit the 3.73 GB MPS budget, and the
    pilot is meant to scale. Batching changes nothing about the numbers — each series is forecast
    from its own context either way.

    VN1's score has no cost asymmetry (absolute error plus bias), so the median is the right choice
    here; `quantile` and `scale` exist so the same calibration question can be asked as on VN2.
    """
    pipeline = None

    def forecaster(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
        nonlocal pipeline
        import torch

        if pipeline is None:
            from vn2.forecaster.chronos_arm import _load  # the Bolt-aware loader, already measured

            pipeline = _load("amazon/chronos-bolt-small")

        context = sales if context_weeks is None else sales.iloc[:, -context_weeks:]
        pieces = []
        for start in range(0, len(context), batch_size):
            chunk = context.iloc[start : start + batch_size]
            tensor = torch.tensor(chunk.to_numpy(dtype=np.float32))
            with torch.no_grad():
                forecast = pipeline.predict(tensor, prediction_length=horizon)
            values = np.asarray(forecast, dtype=float)
            if values.ndim == 3:  # (batch, quantiles, horizon)
                if hasattr(pipeline, "quantiles"):
                    from vn2.forecaster.chronos_arm import pick_quantile

                    values = pick_quantile(values, quantile)
                else:
                    values = np.median(values, axis=1)
            pieces.append(np.clip(values.reshape(len(chunk), horizon) * scale, 0.0, None))
        values = np.concatenate(pieces, axis=0)
        return _frame(values, sales, horizon)

    return forecaster
