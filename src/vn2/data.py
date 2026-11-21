"""Load the VN2 competition files.

The files come from `scripts/fetch_data.sh` and live under `data/`, which git ignores. Everything
here is a thin reader: the shapes, the index and the column names are the competition's, and the
tests pin them.

The one piece of real judgement in this module is `demand_estimate`. Observed sales are censored
whenever a shelf ran dry: `sales = min(start inventory, demand)`, so a week with `in_stock == False`
records what the shelf could serve, not what customers wanted. Every forecaster in this repo sees
`demand_estimate`, which sets those weeks to NaN, so a model never trains on a number that is known
to be too low. How each model copes with the NaN is that model's business.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDEX = ["Store", "Product"]
REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "data" / "raw" / "vn2-inventory-competition"
DATA = OFFICIAL / "Data"

#: The eight weeks the competition played, as indices into the full sales table.
#: Index 156 is 2024-04-08, the last week of history (competition "week 0"); index 157 is 2024-04-15,
#: the first week competitors played. The score covers window weeks 3 to 8, indices 159 to 164.
COMPETITION_ANCHOR = 156


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The competition data is not in git; run scripts/fetch_data.sh"
        )
    return pd.read_csv(path).set_index(INDEX)


def _read_weeks(name: str) -> pd.DataFrame:
    """A weekly table with its week columns as datetimes.

    The competition's own files carry the Monday dates as text and each script converts them
    itself. Doing it once here means seasonality, alignment and date labels all see real dates;
    `vn2_benchmark.py` re-converting them is a no-op.

    The columns are also checked to be contiguous Mondays, which is what lets a forecaster name a
    week past the end of the table by offset rather than by lookup.
    """
    table = _read(name)
    table.columns = pd.to_datetime(table.columns)
    gaps = table.columns.to_series().diff().dropna().unique()
    if len(gaps) != 1 or gaps[0] != pd.Timedelta(weeks=1):
        raise ValueError(f"{name}: the week columns are not contiguous Mondays: {gaps}")
    return table


def sales_full() -> pd.DataFrame:
    """Every week of sales, history and the eight played weeks: 599 rows, 165 week columns."""
    return _read_weeks("Week 8 - 2024-06-03 - Sales.csv")


def sales_history() -> pd.DataFrame:
    """The 157 weeks competitors were given, 2021-04-12 to 2024-04-08."""
    return _read_weeks("Week 0 - 2024-04-08 - Sales.csv")


def in_stock() -> pd.DataFrame:
    """The in-stock flag of every week, history and the eight played weeks included."""
    return _read_weeks("Week 0 - In Stock.csv")


def master() -> pd.DataFrame:
    """Product and store hierarchy, one row per Store x Product."""
    return _read("Week 0 - Master.csv")


def initial_state() -> pd.DataFrame:
    """The state every player started from: the same stock and in-transit orders for all."""
    return _read("Week 0 - 2024-04-08 - Initial State.csv")


def template() -> pd.DataFrame:
    """The layout an order file must have: 599 rows, in this order."""
    return _read("Week 0 - Submission Template.csv")


def week_offset(sales: pd.DataFrame, offset: int) -> pd.Timestamp:
    """The Monday `offset` weeks after the first week of `sales`.

    Valid past the last column, which is what a forecaster needs: the eight weeks it is asked about
    are exactly the weeks the table does not contain yet. `_read_weeks` checks the columns are
    contiguous Mondays, so an offset is a date.
    """
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if offset < len(sales.columns):
        return sales.columns[offset]
    return sales.columns[0] + pd.Timedelta(weeks=offset)


def demand_estimate(sales: pd.DataFrame, in_stock_df: pd.DataFrame) -> pd.DataFrame:
    """The sales table with the censored weeks blanked out.

    `sales` and `in_stock_df` must be cut to the same weeks. Weeks after the last flag are left
    alone: `in_stock` covers the played weeks too, so in practice every column is flagged.
    """
    out = sales.copy()
    flags = in_stock_df.reindex(columns=out.columns)
    out[flags.eq(False)] = np.nan
    return out
