"""Time-series foundation model arm: Chronos, run zero-shot on each series.

Chronos sees a sequence, not a table, so it gets one call per Store x Product with the demand
estimate of the last `context_weeks` weeks as context and is asked for `horizon` weeks ahead. The
hierarchy, the calendar and the other series are invisible to it by construction — that is the
difference being measured against the TabPFN arm, which sees all of them.

Censoring is handled before the model is involved: a week whose shelf ran dry is blanked, then
filled with the trailing median of the uncensored weeks. The fill is causal — it uses only weeks
already in the context — so the model is never handed a number that future weeks helped make.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import demand_estimate, week_offset
from ..features import HORIZONS
from ..play import ForecasterFn

DEFAULT_MODEL = "amazon/chronos-bolt-small"
DEFAULT_CONTEXT_WEEKS = 152  # two years, and the most Chronos takes without truncating
CENSOR_FILL_WINDOW = 13


def causal_fill(series: pd.DataFrame) -> pd.DataFrame:
    """Fill each censored week from the weeks before it, and any leading gap with zero.

    The rolling window is over **weeks**, i.e. along the columns: `rolling` defaults to axis 0,
    which on a (series x weeks) frame would take the median of the neighbouring series in the same
    week — a measured misread, caught by `tests/test_arms.py`, that fills a stock-out with a number
    made from other products.

    The trailing median ignores NaN by construction, so it never looks forward; the leading fill is
    zero because a series with no observed week yet has no level to inherit.
    """
    trailing = series.T.rolling(CENSOR_FILL_WINDOW, min_periods=1).median().T
    return series.fillna(trailing).fillna(0.0)


def chronos(
    model_name: str = DEFAULT_MODEL,
    context_weeks: int = DEFAULT_CONTEXT_WEEKS,
    horizon: int = len(HORIZONS),
) -> ForecasterFn:
    """A zero-shot Chronos forecaster. The model is loaded once and shared by every call."""
    pipeline = None  # loaded on first use: importing this module must not download weights

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        nonlocal pipeline
        if pipeline is None:
            pipeline = _load(model_name)

        demand = causal_fill(demand_estimate(sales, in_stock_df).iloc[:, -context_weeks:])
        context = torch_tensor(demand.to_numpy(dtype=np.float32))
        with torch_no_grad():
            forecast = pipeline.predict(context, prediction_length=horizon)
        values = np.asarray(forecast, dtype=float)
        if values.ndim == 3:  # (batch, samples, horizon): take the median over draws
            values = np.median(values, axis=1)
        values = np.clip(values.reshape(len(sales), horizon), 0.0, None)

        future = pd.DatetimeIndex(
            [week_offset(sales, len(sales.columns) - 1 + h) for h in range(1, horizon + 1)]
        )
        return pd.DataFrame(values, index=sales.index, columns=future)

    return forecaster


def _load(model_name: str):
    """Load through the base class, which dispatches on the config's `chronos_pipeline_class`.

    Calling `ChronosPipeline.from_pretrained` directly is a measured failure: the Bolt checkpoints
    carry a `ChronosBoltPipeline` config (with `input_patch_size` and friends) that the classic
    `ChronosConfig` dataclass rejects with `TypeError`.
    """
    import torch
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return BaseChronosPipeline.from_pretrained(model_name, device_map=device)


def torch_tensor(values: np.ndarray):
    import torch

    return torch.tensor(values)


def torch_no_grad():
    import torch

    return torch.no_grad()
