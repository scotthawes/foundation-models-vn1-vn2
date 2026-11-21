"""Pilot the VN1 comparison on a sample of series before pointing it at all 15,053.

Two origins, and they answer different questions:

- **validation** — the last 13 weeks of Phase 0. Trained on what came before, scored on weeks no arm
  has seen. Use it to choose anything that needs choosing.
- **final** — the real Phase 1 weeks, scored by the organisers' own formula. Reported once.

The cheap arms also run on the full 15,053 series, because they cost nothing: that is the context
the sample has to be read against, and the MA12 figure is the organisers' 0.6597.

Usage:
    python scripts/vn1_pilot.py --series 500 --quote-only
    python scripts/vn1_pilot.py --series 500
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vn1 import data as vn1data  # noqa: E402
from vn1 import forecaster as vn1f  # noqa: E402
from vn1.score import score  # noqa: E402

HORIZON = 13


def sample_index(sales: pd.DataFrame, count: int, seed: int) -> pd.MultiIndex:
    """A fixed, uniformly spread sample of series: the full population includes 30% all-zero ones."""
    positions = np.sort(np.random.default_rng(seed).choice(len(sales), size=min(count, len(sales)),
                                                           replace=False))
    return sales.index[positions]


def evaluate(forecaster, history: pd.DataFrame, price: pd.DataFrame, actual: pd.DataFrame):
    started = time.time()
    prediction = forecaster(history, price)
    elapsed = time.time() - started
    if not prediction.columns.equals(actual.columns):
        raise ValueError(
            f"prediction weeks {prediction.columns[0].date()}..{prediction.columns[-1].date()} "
            f"do not match scored weeks {actual.columns[0].date()}..{actual.columns[-1].date()}"
        )
    return score(prediction, actual), elapsed


def build_arms(names: list[str], train_rows: int, trend: bool = False,
               target: str = "raw") -> dict:
    """Every arm takes `(history, price)`; the cheap ones ignore the price frame."""
    arms = {}
    for name in names:
        if name == "tabpfn_api":
            arms[name] = vn1f.tabpfn_api(max_train_rows=train_rows, trend=trend, target=target)
        elif name == "chronos":
            arms[name] = vn1f.chronos()
        elif hasattr(vn1f, name):
            function = getattr(vn1f, name)
            arms[name] = lambda history, price, fn=function: fn(history)
        else:
            raise SystemExit(f"unknown arm {name}")
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=500)
    parser.add_argument("--all", action="store_true", help="use every series instead of a sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=20_000)
    parser.add_argument("--trend", action="store_true",
                        help="add scale-free trend features to the tabular arm")
    parser.add_argument("--target", default="raw", choices=["raw", "log1p", "ratio"],
                        help="target transform for the tabular arm")
    parser.add_argument("--origins", default="both", choices=["both", "validation", "final"])
    parser.add_argument("--arms", default="ma12,naive_last,tabpfn_api,chronos")
    parser.add_argument("--quote-only", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    from vn2 import keys

    keys.load()  # the quote is authenticated and spends no tokens, but it does need the key
    sales, price, phase1 = vn1data.phase0_sales(), vn1data.phase0_price(), vn1data.phase1_sales()
    count = len(sales) if args.all else args.series
    chosen = sample_index(sales, count, args.seed)
    sampled = sales.loc[chosen]
    sampled_price = price.loc[chosen]

    validation_history = sampled.iloc[:, :-HORIZON]
    validation_actual = sampled.iloc[:, -HORIZON:]
    final_actual = phase1.loc[chosen]

    print(f"{len(sampled):,} of {len(sales):,} series (seed {args.seed}); "
          f"{(final_actual.sum(axis=1) > 0).mean():.0%} with any Phase 1 sales")

    if args.quote_only:
        X_train, _y, X_test = vn1f.prepare(sampled, sampled_price, trend=args.trend)
        from tabpfn_client import estimate_cost

        quote = estimate_cost(X_train, X_test, model_version="v3.5", n_estimators=4)
        print(f"TabPFN quote: {quote.estimated_cost:,.0f} tokens "
              f"(train {X_train.shape[0]:,} x {X_train.shape[1]}, "
              f"predict {X_test.shape[0]:,} x {X_test.shape[1]})")
        return 0

    arms = build_arms([n.strip() for n in args.arms.split(",") if n.strip()], args.train_rows,
                      trend=args.trend, target=args.target)
    rows = []

    # Context: the cheap arms on the whole population, one line each.
    for name in ("ma12", "naive_last"):
        if name not in arms:
            continue
        value, elapsed = evaluate(arms[name], sales, price, phase1)
        rows.append({"origin": "final", "scope": "all 15,053", "arm": name,
                     "score": round(value, 4), "seconds": round(elapsed, 1)})
        print(f"{name:12s} all 15,053  final      {value:.4f}", flush=True)

    origins = {
        "validation": ("validation", validation_history, validation_actual),
        "final": ("final", sampled, final_actual),
    }
    wanted = ["validation", "final"] if args.origins == "both" else [args.origins]
    for origin in wanted:
        _, history, actual = origins[origin]
        for name, forecaster in arms.items():
            value, elapsed = evaluate(forecaster, history, sampled_price, actual)
            rows.append({"origin": origin, "scope": f"{len(sampled):,} sample", "arm": name,
                         "score": round(value, 4), "seconds": round(elapsed, 1)})
            print(f"{name:12s} {len(sampled):>5,} sample  {origin:10s} {value:.4f}  "
                  f"({elapsed:,.1f}s)", flush=True)

    results = pd.DataFrame(rows)
    suffix = f"-rows{args.train_rows}{'-trend' if args.trend else ''}{'-' + args.target if args.target != 'raw' else ''}"
    label = "all" if args.all else str(args.series)
    out = Path("results") / f"vn1-pilot-{label}{suffix}.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(results.pivot_table(index="arm", columns="origin", values="score").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
