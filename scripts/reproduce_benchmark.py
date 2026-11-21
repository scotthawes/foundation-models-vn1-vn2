"""Reproduce the competition's published benchmark: setup 913.8 EUR, score 4,334.0 EUR.

This is the first milestone and the harness's licence to exist. It plays the competition's own
`vn2_benchmark.policy` through this repo's `run_window` and then through the official
`vn2_simulate` loop, and fails loudly if either number moves. `tests/test_play.py` pins the same
thing in CI; this script exists so a human can see the two tables side by side.

Usage:
    python scripts/reproduce_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn2 import data as vn2data  # noqa: E402
from vn2 import play  # noqa: E402

PUBLISHED = {"setup": 913.8, "score": 4334.0}


def official_benchmark():
    sys.path.insert(0, str(vn2data.OFFICIAL))
    import vn2_benchmark

    return vn2_benchmark.policy


def main() -> int:
    weekly, score, _orders = play.competition_run(official_benchmark())
    setup = play.setup(weekly)

    print("week  date         end inv  lost sales  holding  shortage     cost")
    for week, row in weekly.iterrows():
        print(
            f"{week:>4}  {row['Date'].date()}  {row['End Inventory']:>7.0f}  "
            f"{row['Lost Sales']:>10.0f}  {row['Holding Cost']:>7.1f}  "
            f"{row['Shortage Cost']:>8.1f}  {row['Cost']:>7.1f}"
        )
    print(f"\nsetup (weeks 1-2): {setup:.1f} EUR   published {PUBLISHED['setup']:.1f}")
    print(f"score (weeks 3-8): {score:.1f} EUR   published {PUBLISHED['score']:.1f}")

    ok = abs(setup - PUBLISHED["setup"]) < 0.05 and abs(score - PUBLISHED["score"]) < 0.05
    print("\nreproduced exactly." if ok else "\nDOES NOT MATCH the published benchmark.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
