"""Time-series foundation model arm: TimeGPT, run zero-shot through Nixtla's hosted API.

Chronos was the plan; Chronos has no hosted API, and this machine's MPS budget is the same wall the
TabPFN arm hits. TimeGPT is the equivalent client-server time-series foundation model, so the two
arms are now both API-based. The consequence is named in `docs/method.md`: the time-series arm is
TimeGPT, not Chronos, so the headline comparison is a tabular foundation model against a
time-series foundation model, not two implementations of one family.

TimeGPT sees one sequence per Store x Product and nothing else: no hierarchy, no calendar feature,
no other series. Censoring is handled before the model is involved, with the same causal fill as
`chronos_arm`: a week whose shelf ran dry is blanked, then filled from the trailing median of the
uncensored weeks, so no number the model sees was made with a future week's help.

The frame is built and unpacked by pure functions so both halves are testable without an API key.
"""
from __future__ import annotations

import pandas as pd

from .. import keys
from ..data import demand_estimate, week_offset
from ..features import HORIZONS
from ..play import ForecasterFn
from .chronos_arm import causal_fill

DEFAULT_MODEL = "timegpt-2.1"
DEFAULT_CONTEXT_WEEKS = None  # the whole history; TimeGPT has no short fixed context window
ID_SEPARATOR = "|"


def series_ids(sales: pd.DataFrame) -> list[str]:
    """One stable id per Store x Product, in the row order of `sales`."""
    frame = sales.index.to_frame(index=False)
    return (frame["Store"].astype(str) + ID_SEPARATOR + frame["Product"].astype(str)).to_list()


def long_frame(
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    context_weeks: int | None = DEFAULT_CONTEXT_WEEKS,
) -> pd.DataFrame:
    """The history as TimeGPT's long format: one row per (unique_id, week), no missing values.

    `demand_estimate` blanks censored weeks; `causal_fill` fills each from the weeks before it.
    TimeGPT refuses missing values outright, so the fill is a data requirement as well as the
    honest treatment.
    """
    demand = causal_fill(demand_estimate(sales, in_stock_df))
    if context_weeks:
        demand = demand.iloc[:, -context_weeks:]
    long = demand.stack(future_stack=True).rename("y").reset_index()
    long.columns = ["Store", "Product", "ds", "y"]
    long["unique_id"] = (
        long["Store"].astype(str) + ID_SEPARATOR + long["Product"].astype(str)
    )
    return long[["unique_id", "ds", "y"]]


def forecast_wide(
    forecast: pd.DataFrame,
    sales: pd.DataFrame,
    horizon: int = len(HORIZONS),
) -> pd.DataFrame:
    """Turn TimeGPT's long answer back into (Store x Product) rows and week columns.

    The eight future weeks are named by offset from the last column of `sales`, the same way every
    other arm names them, so the columns line up without depending on how TimeGPT labels dates.
    A NaN or a negative is floored to zero: demand cannot be negative and a series with no usable
    history gets no order rather than a missing cell.
    """
    values = [c for c in forecast.columns if c not in ("unique_id", "ds")]
    if len(values) != 1:
        raise ValueError(f"expected one forecast column besides unique_id and ds, got {values}")
    wide = forecast.pivot(index="unique_id", columns="ds", values=values[0])
    wide.columns = pd.to_datetime(wide.columns)

    future = pd.DatetimeIndex(
        [week_offset(sales, len(sales.columns) - 1 + h) for h in range(1, horizon + 1)]
    )
    wide = wide.reindex(index=series_ids(sales), columns=future)
    wide = wide.clip(lower=0.0).fillna(0.0)
    wide.index = sales.index
    return wide


def timegpt(
    model: str = DEFAULT_MODEL,
    context_weeks: int | None = DEFAULT_CONTEXT_WEEKS,
    horizon: int = len(HORIZONS),
    finetune_steps: int = 0,
) -> ForecasterFn:
    """A zero-shot TimeGPT forecaster. The client is created once and shared by every call."""
    client = None  # created on first use: importing this module must not need a key

    def forecaster(
        sales: pd.DataFrame,
        in_stock_df: pd.DataFrame,
        master_df: pd.DataFrame,
        state: pd.DataFrame,
    ) -> pd.DataFrame:
        nonlocal client
        if client is None:
            from nixtla import NixtlaClient

            client = NixtlaClient(api_key=keys.require("NIXTLA_API_KEY", "the TimeGPT arm"))

        history = long_frame(sales, in_stock_df, context_weeks=context_weeks)
        forecast = client.forecast(
            history,
            h=horizon,
            freq="W-MON",
            model=model,
            finetune_steps=finetune_steps,
        )
        return forecast_wide(forecast, sales, horizon=horizon)

    return forecaster
