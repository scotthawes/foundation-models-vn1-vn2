# VN1 pilot: does the model question survive a pure-forecast task?

VN2 measured inventory cost, where the ordering policy and coverage swamped the forecast difference.
VN1 measures forecast error directly, so it is the cleaner test of the same question — and it is
25× larger, with a price covariate the sequence model cannot see.

## What VN1 is (measured from the local files)

| | |
| --- | --- |
| task | 13-week-ahead forecast, no ordering |
| series | 15,053 (Client x Warehouse x Product) |
| history | 170 weeks, 2020-07-06 to 2023-10-02, with a weekly price per series |
| scored weeks | Phase 1, 2023-10-09 to 2024-01-01 (present, so the pilot self-scores) |
| score | `(sum abs error + abs net error) / sum actual` — absolute error plus a bias penalty |
| organisers' baseline | MA12 = **0.6597** on all 15,053 series (naive 0.6882, zero 2.0) |

There is **no in-stock flag**, so censoring cannot be identified the way `vn2.data.demand_estimate`
identifies it. Nothing in the pilot blanks a week; that limitation is not fixable from these files.

## What was run

500 of 15,053 series (seed 0, 73% with any Phase 1 sales), two origins:

- **validation** — the last 13 weeks of Phase 0, which no arm trained on;
- **final** — the real Phase 1 weeks, scored by the organisers' formula.

Cheap arms also run on all 15,053, because they cost nothing and give the sample its context.

| arm | validation | final | all 15,053 (final) |
| --- | ---: | ---: | ---: |
| `chronos` | **0.4106** | **0.5690** | — |
| `ma12` | 0.4836 | 0.6261 | 0.6597 |
| `naive_last` | 0.4791 | 0.6424 | 0.6882 |
| `tabpfn_api` | 0.7082 | 0.8391 | — |

## Why TabPFN is last, by decomposition

The score is absolute error plus bias, so it splits cleanly. On the final origin:

| arm | score | WAPE | bias | mean forecast | mean actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chronos` | 0.5690 | 0.4484 | -0.1206 | 15.28 | 17.37 |
| `ma12` | 0.6261 | 0.4681 | -0.1579 | 14.63 | 17.37 |
| `tabpfn_api` | 0.8391 | 0.5622 | -0.2770 | 12.56 | 17.37 |

TabPFN is less accurate *and* more biased: it under-forecasts by 28% of total volume where MA12
misses by 16% and Chronos by 12%. Every arm under-forecasts because sales rise over the scored
weeks, but TabPFN under-forecasts roughly twice as hard.

## Configuration check (validation origin)

VN2 taught that configuration matters more than family, so the pilot's first TabPFN result was
challenged on the validation origin before anything was concluded. Sampled 20k context versus the
full 65k, and 4 versus 8 estimators:

| context rows | estimators | score | WAPE | bias | mean forecast |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 (sampled) | 4 | 0.7082 | — | — | — |
| 65,000 (all) | 8 | 0.6294 | 0.4824 | -0.1470 | 12.52 |
| 65,000 (all) | 4 | 0.6223 | 0.4827 | -0.1395 | 12.63 |

Full context buys 0.086; the gap to MA12 (0.4836) is still 0.14 and to Chronos (0.4106) is 0.21.
Estimators make no difference at this size. So the configuration was a real defect, but fixing it
does not change the ordering.

## Trend hypothesis: tested and rejected (validation origin)

The remaining explanation was that the table encodes level but no trend, so every model regresses to
the historic mean while sales rise. Three scale-free features were added — `lag_1` and `lag_4`
relative to the 13-week mean, and the 4-week-to-13-week mean ratio — and the full-context arm rerun:

| configuration (validation) | score |
| --- | ---: |
| full context, no trend features | 0.6223 |
| full context + trend ratios | 0.6824 |

The trend features made it **worse by 0.06**. The hypothesis is rejected: the absolute-value table
already carries the level, and the extra ratios pushed the forecast the wrong way. TabPFN's deficit
on this task is not a missing-feature problem.

## What this does and does not show

- It shows a **signal worth pursuing**: on a pure forecast metric, Chronos beats the organisers'
  baseline by 0.09 on the full population (0.569 vs 0.660), and by 0.07 on the 500-series sample.
- It does not establish why TabPFN lags. Two explanations were tested — context size (helped, not
  enough) and trend information (hurt) — and neither closes the gap. What remains untested is a
  target transform (the targets are 72% zeros) and per-series normalisation.
- The pilot's cost is trivial: **10,000 tokens per origin** (the API minimum) for 20k training rows
  and 6,500 predictions, so a full 15,053-series run is affordable on the free tier.

## Next, if it is continued

1. Scale the winner, not the laggard: run Chronos and MA12 on all 15,053 series and report against
   the organisers' 0.6597. That is the headline the pilot was sizing for.
2. If TabPFN is revisited, the untested knobs are a target transform and per-series normalisation,
   not more context or more estimators.
3. Multiple seeds for the sample; a single 500-series draw is a small experiment.
