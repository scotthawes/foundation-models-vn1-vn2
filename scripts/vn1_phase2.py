"""Score Phase 2 — the window that actually ranked VN1 — with the official formula.

The official bundle (`data/raw/vn1-official/vn1-forecasting-competition/`, downloaded from
<https://supchains.com/vn1-forecasting-competition/>) holds Phase 2's actuals and the five winning
forecasts. This script scores those, our Chronos at two calibrations, and the two benchmarks, so
the comparison is on one window with one formula.

Run `scripts/fetch_vn1_official.sh` first (or download the zip into data/raw/vn1-official/).

Usage:
    python scripts/vn1_phase2.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn1 import data as vn1data  # noqa: E402
from vn1 import forecaster as vn1f  # noqa: E402
from vn1.score import score  # noqa: E402

INDEX = ["Client", "Warehouse", "Product"]
BUNDLE = vn1data.REPO / "data" / "raw" / "vn1-official" / "vn1-forecasting-competition"


def read_weekly(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path).set_index(INDEX)
    table.columns = pd.to_datetime(table.columns)
    return table


def main() -> int:
    if not BUNDLE.exists():
        raise SystemExit(
            f"{BUNDLE} is missing. Download the official bundle:\n"
            "  https://supchains.com/downloads/vn1-forecasting-competition.zip"
        )

    warnings.filterwarnings("ignore")
    actual = read_weekly(BUNDLE / "Phase 2 - Sales.csv")
    rows = []

    for path in sorted((BUNDLE / "Top 5 forecasts").glob("*.csv")):
        rows.append((path.stem, score(read_weekly(path).reindex(index=actual.index), actual)))

    history = pd.concat([vn1data.phase0_sales(), vn1data.phase1_sales()], axis=1)
    price = vn1data.phase0_price()
    forecast = vn1f.chronos()(history, price)

    for label, frame in (
        ("Chronos, median", forecast),
        ("Chronos, scale 1.1", forecast * 1.1),
        ("Chronos, Phase-1 bias scale (1.125)", forecast * 1.125),
    ):
        rows.append((label, score(frame, actual)))

    rows.sort(key=lambda row: row[1])
    print(f"{'entry':44s} {'score':>7s}")
    for name, value in rows:
        print(f"{name:44s} {value:7.4f}")
    print(f"\n{'naive forecast (official)':44s} {0.506970:7.4f}")
    print(f"{'12-week moving average (official)':44s} {0.804585:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
