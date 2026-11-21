"""Turn the VN2 history into one row per (Store x Product, week ahead) for a tabular model.

The shape of the problem is fixed by the ordering policy: at each round the model must say what
demand will be for the next eight weeks, for 599 combinations. That is a table, not a sequence, so
it can be handed to a tabular foundation model directly.

Two rules make the rows honest:

- **Nothing past the anchor.** Every feature for a row anchored at week `a` is computed from
  columns `0 .. a` only, and its target is the demand at `a + h`. `tests/test_features.py` fails
  the build if a feature column moves when later weeks are scrambled.
- **A censored week is not a target.** Where `in_stock` is False the shelf ran dry, so the sales
  recorded are the shelf and not the demand. Those targets are dropped rather than imputed:
  imputing would teach the model a number that is known to be too low.

Missing lags stay NaN. TabPFN treats a missing feature as missing, which is exactly what "this
series is younger than 52 weeks" means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import demand_estimate, week_offset

INDEX = ["Store", "Product"]
MASTER_COLUMNS = ["ProductGroup", "Division", "Department", "DepartmentGroup", "StoreFormat", "Format"]
LAGS = (1, 2, 3, 4, 8, 13, 26, 52)   # weeks back; 52 is the yearly cycle
ROLLING = (4, 13, 26)                # weeks in the trailing means
CENSOR_WINDOW = 13                   # weeks over which the stock-out rate is measured
HORIZONS = tuple(range(1, 9))        # the eight weeks an order has to cover


def train_anchors(history_weeks: int, lookback: int = 104, step: int = 4) -> list[int]:
    """The pseudo-anchors used to build training rows, all inside the history.

    A row anchored at `a` needs `a + max(HORIZONS)` to exist as a target and `max(LAGS)` weeks of
    context before it, so the usable anchors are a narrow band at the end of the history. The
    bound is `history_weeks - max(HORIZONS) - 1` because the last usable target is the last column.
    """
    first = max(max(LAGS) - 1, history_weeks - lookback)
    last = history_weeks - max(HORIZONS) - 1
    if last < first:
        return []
    return list(range(first, last + 1, step))


def build_rows(
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    master_df: pd.DataFrame,
    anchors: list[int],
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """The training table: one row per (Store x Product, anchor, horizon).

    Every feature comes from columns `0 .. anchor`. `target` is the demand at `anchor + h` and is
    NaN where that week was censored, which is the caller's signal to drop the row.
    """
    if not anchors:
        raise ValueError("no training anchors: not enough history to build a row")
    demand = demand_estimate(sales, in_stock_df)
    frames = []
    for anchor in anchors:
        for h in horizons:
            if anchor + h >= len(sales.columns):
                raise ValueError(
                    f"anchor {anchor} + horizon {h} runs past the {len(sales.columns)} weeks given"
                )
            block = _block(sales, in_stock_df, master_df, anchor, h)
            block["target"] = demand.iloc[:, anchor + h].to_numpy()
            frames.append(block)
    return pd.concat(frames, ignore_index=True)


def prediction_rows(
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    master_df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """The same features for the current anchor, with no target: what the model is asked now."""
    anchor = len(sales.columns) - 1
    return pd.concat(
        [_block(sales, in_stock_df, master_df, anchor, h) for h in horizons], ignore_index=True
    )


def _block(
    sales: pd.DataFrame,
    in_stock_df: pd.DataFrame,
    master_df: pd.DataFrame,
    anchor: int,
    h: int,
) -> pd.DataFrame:
    """The feature rows for one (anchor, horizon), from columns `0 .. anchor` and nothing later."""
    past = demand_estimate(sales, in_stock_df).iloc[:, : anchor + 1]
    past_flags = in_stock_df.reindex(columns=sales.columns).iloc[:, : anchor + 1]

    out = _lag_features(past, past_flags).reset_index()  # Store and Product become columns
    out = pd.concat(
        [out, master_df.loc[past.index, MASTER_COLUMNS].reset_index(drop=True)], axis=1
    )
    iso = float(week_offset(sales, anchor + h).isocalendar().week)
    out["anchor"] = anchor
    out["h"] = h
    out["iso_week"] = iso
    out["iso_sin"] = np.sin(2 * np.pi * iso / 52)
    out["iso_cos"] = np.cos(2 * np.pi * iso / 52)
    return out


def _lag_features(past: pd.DataFrame, past_flags: pd.DataFrame) -> pd.DataFrame:
    """All moving features for one anchor, from columns `0 .. a` and nothing later."""
    weeks = len(past.columns)
    out = pd.DataFrame(index=past.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = past.iloc[:, -lag] if lag <= weeks else np.nan
    for window in ROLLING:
        out[f"mean_{window}"] = past.iloc[:, -window:].mean(axis=1)
    out["std_13"] = past.iloc[:, -CENSOR_WINDOW:].std(axis=1)
    out["censor_rate_13"] = 1.0 - past_flags.iloc[:, -CENSOR_WINDOW:].mean(axis=1)
    out["weeks_seen"] = float(weeks)
    return out
