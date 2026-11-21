"""Load the VN1 datathon files.

Phase 0 is 170 weeks of history (2020-07-06 to 2023-10-02) with a weekly price per series; Phase 1
is the 13 weeks that were scored (2023-10-09 to 2024-01-01). The organisers' evaluation function
reads exactly these files, so the columns are parsed as dates once, here.

There is no in-stock flag in VN1. Unlike VN2, a censored week cannot be identified, so nothing in
this package blanks one; that limitation is recorded in the pilot's write-up rather than papered
over with a guess.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INDEX = ["Client", "Warehouse", "Product"]
REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "vn1-pilot"


def _read(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Copy the VN1 files into data/raw/vn1-pilot/ (see README)."
        )
    table = pd.read_csv(path).set_index(INDEX)
    table.columns = pd.to_datetime(table.columns)
    return table


def phase0_sales() -> pd.DataFrame:
    """170 weeks of sales, one row per Client x Warehouse x Product."""
    return _read("Phase 0 - Sales.csv")


def phase0_price() -> pd.DataFrame:
    """The same weeks of weekly price, aligned to `phase0_sales`."""
    return _read("Phase 0 - Price.csv")


def phase1_sales() -> pd.DataFrame:
    """The 13 scored weeks: what a Phase 1 submission is compared against."""
    return _read("Phase 1 - Sales.csv")
