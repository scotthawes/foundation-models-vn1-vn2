"""Play the competition's own eight weeks, once, from the real initial state.

This is the only use of weeks 157-164 in the repo. Everything before it was measured on anchors that
end at week 156; the evaluation anchors are what decide the reported comparison, and this is what
the competition would have scored. Running it more than once per arm and picking the best would turn
a held-out result into a tuning set, so the script plays each arm exactly once.

The benchmark arm is also a harness check: it must score the published 4,334.0, exactly as
`scripts/reproduce_benchmark.py` prints.

Usage:
    python scripts/competition_run.py
    python scripts/competition_run.py --arms benchmark,seasonal_ma,tabpfn_api --coverage 3.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vn2 import forecaster, play  # noqa: E402

PUBLISHED = 4334.0

ARMS = {
    "benchmark": lambda: forecaster.seasonal_moving_average(13),
    "seasonal_ma": lambda: forecaster.seasonal_moving_average(13),
    "naive_mean": forecaster.naive_mean,
    "seasonal_naive": forecaster.seasonal_naive,
    "chronos": forecaster.chronos,
    "tabpfn_api": forecaster.tabpfn_api,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="benchmark,seasonal_ma,chronos,tabpfn_api")
    parser.add_argument("--coverage", type=float, default=3.5)
    parser.add_argument("--tag", default="competition")
    args = parser.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")

    rows = []
    for name in arm_names:
        coverage = 4.0 if name == "benchmark" else args.coverage  # the published rule, unaltered
        policy = play.order_up_to_policy(ARMS[name](), coverage=coverage)
        started = time.time()
        weekly, score, _orders = play.competition_run(policy)
        elapsed = time.time() - started
        scored = weekly.loc[list(play.ScoredWeeks)]
        rows.append(
            {
                "arm": name,
                "coverage": coverage,
                "setup": play.setup(weekly),
                "score": score,
                "vs_benchmark": round(score - PUBLISHED, 1),
                "lost_sales": int(scored["Lost Sales"].sum()),
                "end_inventory": int(scored["End Inventory"].sum()),
                "holding": round(float(scored["Holding Cost"].sum()), 1),
                "shortage": round(float(scored["Shortage Cost"].sum()), 1),
                "seconds": round(elapsed, 1),
            }
        )
        print(f"{name:12s} coverage {coverage:<4} score {score:>8,.1f}  ({elapsed:,.0f}s)", flush=True)

    results = pd.DataFrame(rows).sort_values("score")
    print()
    print(results.to_string(index=False))

    benchmark = results.loc[results["arm"] == "benchmark", "score"]
    if not benchmark.empty and abs(float(benchmark.iloc[0]) - PUBLISHED) > 0.05:
        print(f"\nHARNESS CHECK FAILED: benchmark scored {float(benchmark.iloc[0])}, "
              f"published {PUBLISHED}")
        return 1
    if not benchmark.empty:
        print(f"\nharness check: benchmark scored {PUBLISHED} as published.")

    out = Path("results") / f"{args.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
