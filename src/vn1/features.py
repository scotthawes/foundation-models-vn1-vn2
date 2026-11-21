"""The tabular arm's table: one row per (series, week ahead), from weeks known at the origin.

Same rule as VN2's feature builder: every feature for a row anchored at week `a` comes from columns
`0 .. a`, and the target is the sale at `a + h`. Nothing later is touched, which is what lets a
fold move the origin backwards without contaminating itself.

The difference is scale and context. Here there are 15,053 series and a price column, so the table
carries the series' own lags and rolling means, the price level, and the series identifiers — the
information a sequence model cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 4, 8, 13, 26, 52)
ROLLING = (4, 13, 26)
HORIZON = 13
IDENTIFIERS = ["Client", "Warehouse", "Product"]


def train_anchors(weeks: int, lookback: int = 52, step: int = 4, horizon: int = HORIZON) -> list[int]:
    """Pseudo-origins inside the history: the last week of each must leave `horizon` weeks of target."""
    first = max(max(LAGS) - 1, weeks - lookback)
    last = weeks - horizon - 1
    if last < first:
        return []
    return list(range(first, last + 1, step))


def _snapshot(sales: pd.DataFrame, price: pd.DataFrame, anchor: int) -> pd.DataFrame:
    """One row per series, from columns `0 .. anchor` and nothing after it."""
    past = sales.iloc[:, : anchor + 1]
    past_price = price.iloc[:, : anchor + 1]

    out = pd.DataFrame(index=sales.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = past.iloc[:, -lag] if lag <= past.shape[1] else np.nan
    for window in ROLLING:
        out[f"mean_{window}"] = past.iloc[:, -window:].mean(axis=1)
    out["mean_52"] = past.iloc[:, -52:].mean(axis=1) if past.shape[1] >= 52 else np.nan
    out["price"] = past_price.iloc[:, -1]
    out["price_mean_13"] = past_price.iloc[:, -13:].mean(axis=1)
    out["weeks_seen"] = float(past.shape[1])
    return out.reset_index()


def _calendar(block: pd.DataFrame, week: pd.Timestamp) -> pd.DataFrame:
    """The week of the horizon as a position in the year, in the same sine/cosine form as VN2."""
    iso = float(pd.Timestamp(week).isocalendar().week)
    block["iso_sin"] = np.sin(2 * np.pi * iso / 52)
    block["iso_cos"] = np.cos(2 * np.pi * iso / 52)
    return block


def training_rows(
    sales: pd.DataFrame,
    price: pd.DataFrame,
    anchors: list[int],
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Every (anchor, horizon, series) row with its target; call the trainer's sampler to shrink it."""
    if not anchors:
        raise ValueError("no training anchors: the history is shorter than a lag plus a horizon")
    frames = []
    for anchor in anchors:
        base = _snapshot(sales, price, anchor)
        for h in range(1, horizon + 1):
            block = _calendar(base.copy(), sales.columns[anchor + h])
            block["h"] = h
            block["target"] = sales.iloc[:, anchor + h].to_numpy()
            frames.append(block)
    return pd.concat(frames, ignore_index=True)


def prediction_rows(
    sales: pd.DataFrame,
    price: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """The same features at the last known week, with no target: what the model is asked now."""
    base = _snapshot(sales, price, sales.shape[1] - 1)
    frames = []
    for h in range(1, horizon + 1):
        week = sales.columns[-1] + pd.Timedelta(weeks=h)
        block = _calendar(base.copy(), week)
        block["h"] = h
        frames.append(block)
    return pd.concat(frames, ignore_index=True)
