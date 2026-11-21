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
QUANTILES = tuple(round(0.1 * k, 1) for k in range(1, 10))  # what Chronos-Bolt emits


def pick_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    """Take one quantile out of a (batch, 9, horizon) forecast, interpolating between grid points.

    The ordering policy has no distribution of its own, so which quantile it is fed decides the
    service level. VN2 charges 1 EUR per lost sale and 0.2 EUR per unit held, so the newsvendor
    critical fractile is 1/(1+0.2) = 0.833 — a quantile the median cannot express.
    """
    grid = np.asarray(QUANTILES)
    q = float(np.clip(quantile, grid[0], grid[-1]))
    if q in grid:
        return values[:, int(np.where(grid == q)[0][0]), :]
    high = int(np.searchsorted(grid, q))
    low = high - 1
    weight = (q - grid[low]) / (grid[high] - grid[low])
    return values[:, low, :] * (1.0 - weight) + values[:, high, :] * weight


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
    quantile: float = 0.5,
    scale: float = 1.0,
) -> ForecasterFn:
    """A zero-shot Chronos forecaster. The model is loaded once and shared by every call.

    `quantile` picks which of the nine predicted quantiles becomes the forecast the policy sees,
    and `scale` multiplies it afterwards. Both default to the plain median, which is what the
    original comparison used; the grid in `scripts/chronos_grid.py` is what varies them.
    """
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
        if values.ndim == 3:
            # Bolt emits quantiles; the classic model emits samples. Only the former has a quantile
            # axis to pick from, and `_load` returns the class that matches the checkpoint.
            if hasattr(pipeline, "quantiles"):
                values = pick_quantile(values, quantile)
            else:
                values = np.median(values, axis=1)  # median forecast of the sample paths
        values = np.clip(values.reshape(len(sales), horizon) * scale, 0.0, None)

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


CALIBRATION_WINDOWS = 4   # historical forecast windows used to estimate each series' bias
CALIBRATION_STEP = 4      # weeks between those windows
CALIBRATION_SHRINK = 2.0  # pseudo-units pulling a thin series' ratio back toward 1
CALIBRATION_CLIP = (0.5, 2.5)


def series_ratios(
    forecaster: ForecasterFn,
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    master_df: pd.DataFrame,
    state: pd.DataFrame,
    windows: int = CALIBRATION_WINDOWS,
    step: int = CALIBRATION_STEP,
    shrink: float = CALIBRATION_SHRINK,
    clip: tuple[float, float] = CALIBRATION_CLIP,
) -> pd.Series:
    """Each series' realised demand divided by what Chronos forecast for it, on past windows only.

    Every window ends at or before the last week `sales` knows, so a forecast for week `t` is
    calibrated with information from weeks before `t`. Censored weeks are excluded from both the
    numerator and the denominator: a stock-out is not evidence about forecast bias.

    The ratio is smoothed toward 1 with `shrink` pseudo-units, so a series with one small sale and a
    forecast of 0.1 does not become a 10x multiplier, and clipped to a sane band.
    """
    weeks = len(sales.columns)
    actual_total = np.zeros(len(sales))
    forecast_total = np.zeros(len(sales))

    for k in range(1, windows + 1):
        origin = weeks - 1 - k * step  # last week the calibration forecast may know
        if origin + 8 > weeks:
            continue
        history = sales.iloc[:, : origin + 1]
        flags = in_stock_df[sales.columns[: origin + 1]]
        state_arg = None if state is None else state.copy()
        forecast = forecaster(history, flags, master_df.copy(), state_arg).to_numpy(dtype=float)
        realised = demand_estimate(sales, in_stock_df).iloc[:, origin + 1 : origin + 9].to_numpy()

        observed = ~np.isnan(realised)
        actual_total += np.where(observed, realised, 0.0).sum(axis=1)
        forecast_total += np.where(observed, forecast, 0.0).sum(axis=1)

    ratio = (actual_total + shrink) / (forecast_total + shrink)
    return pd.Series(np.clip(ratio, *clip), index=sales.index)


def calibrated_chronos(
    model_name: str = DEFAULT_MODEL,
    context_weeks: int = DEFAULT_CONTEXT_WEEKS,
    horizon: int = len(HORIZONS),
    windows: int = CALIBRATION_WINDOWS,
    step: int = CALIBRATION_STEP,
    shrink: float = CALIBRATION_SHRINK,
    clip: tuple[float, float] = CALIBRATION_CLIP,
) -> ForecasterFn:
    """The Chronos arm with a per-series bias correction, estimated causally from past windows.

    The median forecast is a median, not a mean, and the policy orders as if it were a mean. A
    single global scale cannot repair that — it lifts series that are already over-forecast as much
    as those that are not (`docs/chronos-vs-baseline.md`). This corrects each series with its own
    past forecast-to-actual ratio instead.
    """
    base = chronos(model_name=model_name, context_weeks=context_weeks, horizon=horizon)

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        ratios = series_ratios(
            base, sales, in_stock_df, master_df, state, windows=windows, step=step,
            shrink=shrink, clip=clip,
        )
        forecast = base(sales, in_stock_df, master_df, state)
        return forecast.mul(ratios, axis=0)

    return forecaster
