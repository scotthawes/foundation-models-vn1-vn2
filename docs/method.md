# Method (pre-registered)

Written before any arm was scored, so that the bars are fixed and the numbers are not chosen after
the fact. Anything that changes later is recorded as a change with the date and the reason.

## The game

VN2 Inventory Planning Challenge (Vandeput, 2025). 599 Store x Product series, eight weeks played,
an order placed at the start of weeks 1-6. Cost is `0.2 EUR x units left on shelf + 1 EUR x lost
sales`, scored over weeks 3-8; weeks 1-2 are setup and are 913.8 EUR for everybody. The published
benchmark is **4,334.0 EUR** and the winner scored **3,763.2 EUR**; only 25 of 180 entries beat the
benchmark.

The harness calls the competition's own `vn2_simulate.play_week` and `check_order` — never a
re-implementation — and `tests/test_play.py` pins both the published score and the equivalence of
this repo's loop to the official one. `scripts/reproduce_benchmark.py` prints the same table for a
human.

## What the arms differ in

One ordering policy for every arm: order up to `coverage` weeks of forecast demand, net of what is
on the shelf and in transit (`vn2.play.order_up_to`). The competition hard-codes coverage 4; it is a
parameter here and is tuned on the tuning anchors only. So a difference in score is a difference in
forecast, not in the rule.

| arm | what it is | context |
| --- | --- | --- |
| `benchmark` | seasonal moving average, 13 weeks, coverage 4 | the competition's own |
| `seasonal_ma` | the same forecaster, coverage tuned | cheap reference |
| `naive_mean`, `seasonal_naive` | two naive baselines | cheap reference |
| `tabpfn_api` | TabPFN through the Prior Labs API, `v3.5_default`, 4 estimators | table of lags, calendar, hierarchy |
| `chronos` | Chronos-Bolt-small, **local** | one sequence per series, nothing else |
| `timegpt` | TimeGPT-2.1 through Nixtla — **written, unavailable** | one sequence per series |

### Why only one arm is called

Measured on this machine (Apple silicon, MPS budget 3.73 GB):

- local TabPFN, 2,000 training rows, 4 estimators: **531.7 s per fit**, and the fit itself is 0.3 s
  — the cost is inference over the 4,792 prediction rows; 4,000 rows runs out of memory, and CPU
  refuses more than 1,000 rows without an override.
- the same fit+predict through the Prior Labs API: **9 s wall, 1.5 s of server compute**, billed at
  the 10,000-token minimum (≈500 rounds against the 5M/day free tier).
- local Chronos-Bolt-small: **11.4 s first call (weights), 0.7 s after**, deterministic, 599 x 8.
  It fits the MPS budget, so there is nothing to move off the machine.

So TabPFN is called and Chronos is local, which restores the original comparison: a tabular
foundation model against a time-series foundation model. TimeGPT was written as an arm
(`timegpt_arm.py`) but the free trial key was not available, so it is not part of the comparison;
the code stays for anyone who has a key.

### What is pinned, and what is not

`model_path="v3.5_default"` pins TabPFN by name, but the API's model is not byte-identical to the
open weights the local `tabpfn` package ships. Two runs on different days can therefore differ in
the last digits; the recorded version and date are part of the result. Chronos is pinned by the
HuggingFace revision in `DEFAULT_MODEL`.

## Censoring

A week with `in_stock == False` records what the shelf could serve, not demand. Every arm sees
`sales` with those weeks blanked. Tabular arms drop them as training targets (never impute a target
known to be too low); sequence arms fill them from the trailing 13-week median of uncensored weeks,
which is causal and required because TimeGPT refuses missing values outright.

## Back-test

- **Shared warm-up.** Each fold starts from a state reached by playing the competition's benchmark
  policy for 52 weeks from a zero state, at the fold's anchor. Every arm starts from that identical
  state, so it is scored on forecasting, not on where the shelf happened to be.
- **Tuning anchors** `104, 108, ..., 128` choose the coverage. **Evaluation anchors**
  `132, 136, 140, 144, 148` report it. The two sets are disjoint; the last evaluation window ends at
  week 156, so no fold touches the eight weeks the competition played.
- **Competition run** happens once, at anchor 156, from the real initial state, after the method is
  frozen. It is the only use of those eight weeks.

## Pre-registered bars

1. **Beats the benchmark** if the mean score over evaluation folds is below the mean score of the
   benchmark arm on the same folds.
2. **One arm beats another** only if the mean difference exceeds the fold-to-fold standard error of
   the difference. Folds overlap by construction, so this bar is generous to a noise win; it is the
   floor, not a confidence interval.
3. **The TabPFN-vs-Chronos result stands** only if it clears bar 2 and holds at the competition
   anchor. A win on evaluation folds that reverses on the played weeks is reported as unstable.

## Tuning results (2026-11-21)

Seven tuning anchors, coverage swept 3.0-5.0 in steps of 0.5. Generated from
`results/backtest-tuning.csv`:

| arm | coverage | mean | std | vs benchmark |
| --- | ---: | ---: | ---: | ---: |
| `benchmark` | 4.0 | 4,806.4 | 817.6 | +0.0 |
| `seasonal_ma` | 3.0 | 4,365.3 | 629.4 | -441.1 |
| `naive_mean` | 3.5 | 4,611.8 | 930.8 | -194.7 |
| `seasonal_naive` | 3.5 | 5,269.2 | 491.3 | +462.8 |
| `chronos` | 3.5 | 4,858.7 | 1,129.0 | +52.3 |
| `tabpfn_api` | 3.5 | 4,581.0 | 1,010.8 | -225.4 |

Paired differences at coverage 3.5 (benchmark at its own 4.0), with the fold-to-fold standard error
of the difference:

| comparison | mean difference | paired SE | t |
| --- | ---: | ---: | ---: |
| `seasonal_ma` - `benchmark` | -419.4 | 111.4 | -3.76 |
| `tabpfn_api` - `benchmark` | -225.4 | 156.2 | -1.44 |
| `tabpfn_api` - `chronos` | -277.7 | 57.5 | -4.83 |
| `tabpfn_api` - `seasonal_ma` | +194.0 | 135.0 | +1.44 |
| `tabpfn_api` - `naive_mean` | -30.7 | 80.6 | -0.38 |
| `chronos` - `benchmark` | +52.3 | 196.7 | +0.27 |

**Decision, recorded before the evaluation anchors were run:** coverage is a single value, **3.5**,
for every arm. Per-arm tuning was rejected because only `seasonal_ma` preferred another value (3.0),
by 21.8 EUR against a paired SE of 88.3 — noise; and one number for all arms keeps the comparison
about forecasts rather than about five separate coverage choices.

**Reading.** `tabpfn_api` beats `chronos` by 277.7 ± 57.5, which clears bar 2. It beats the
benchmark by 225.4 ± 156.2 and the cheap `naive_mean` by 30.7 ± 80.6, which do not. `seasonal_ma`
beats the benchmark by 419.4 ± 111.4, which does. So on tuning anchors the moving average is the
strongest arm, and the result that survives the bar is TabPFN-over-Chronos, not TabPFN-over-baseline.

## Evaluation results (2026-11-21)

Five held-out anchors (132-148), coverage fixed at 3.5 by the decision above; the benchmark keeps
its own 4.0. Generated from `results/backtest-evaluation.csv`:

| arm | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| `benchmark` | 6,083.0 | 2,270.7 | 4,119.2 | 9,011.4 |
| `seasonal_ma` | 5,739.8 | 2,268.6 | 3,630.6 | 8,783.8 |
| `tabpfn_api` | 6,302.1 | 2,806.3 | 3,610.0 | 10,366.8 |
| `chronos` | 6,855.7 | 3,388.6 | 3,609.0 | 11,608.8 |
| `seasonal_naive` | 6,923.2 | 1,879.8 | 4,860.8 | 9,368.2 |
| `naive_mean` | 7,153.2 | 2,350.5 | 4,892.8 | 10,642.2 |

| comparison | mean difference | paired SE | t |
| --- | ---: | ---: | ---: |
| `seasonal_ma` - `benchmark` | -343.2 | 68.5 | -5.01 |
| `tabpfn_api` - `benchmark` | +219.1 | 318.3 | +0.69 |
| `tabpfn_api` - `chronos` | -553.6 | 267.9 | -2.07 |
| `tabpfn_api` - `seasonal_ma` | +562.3 | 279.2 | +2.01 |
| `chronos` - `benchmark` | +772.7 | 531.9 | +1.45 |
| `naive_mean` - `benchmark` | +1,070.2 | 220.1 | +4.86 |

**Verdict, against the bars written before the run.**

- Bar 1: only `seasonal_ma` beats the benchmark, and it does so on both anchor sets.
- Bar 2: TabPFN-over-Chronos clears it on both anchor sets (−277.7 ± 57.5 on tuning,
  −553.6 ± 267.9 on evaluation), and the direction is consistent.
- Bar 3: TabPFN-over-Chronos does **not** hold at the competition anchor (see below), so it is
  recorded as a small, unstable edge rather than a standing result.
- TabPFN vs the benchmark flips sign between the two anchor sets (−225.4 on tuning, +219.1 on
  evaluation): unstable, not a result.

## Competition run (2026-11-21, anchor 156, one pass)

| arm | coverage | score | vs benchmark | lost sales | end inventory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seasonal_ma` | 3.5 | 3,915.0 | -419.0 | 1,481 | 12,170 |
| `naive_mean` | 3.5 | 3,994.2 | -339.8 | 2,024 | 9,851 |
| `chronos` | 3.5 | 4,237.2 | -96.8 | 2,541 | 8,481 |
| `tabpfn_api` | 3.5 | 4,252.2 | -81.8 | 1,706 | 12,731 |
| `benchmark` | 4.0 | 4,334.0 | 0.0 | 1,078 | 16,280 |

The benchmark arm scored the published 4,334.0 exactly, so the harness is intact.

**Conclusion.** The question was whether a tabular foundation model forecasts the demand that pays
better than a time-series one. On twelve back-test folds the answer is yes, by 278-554 EUR a window.
On the eight weeks that were actually played it is no: Chronos edges TabPFN by 15.0 EUR, which is
noise but is the wrong sign. Neither beats the competition's own moving average with a lower
coverage, which is the one result that holds in all three settings. And everything is dominated by
fold-to-fold variance — standard deviations of 2,000+ EUR against mean gaps of a few hundred.

The honest summary: TabPFN and Chronos both price a shelf competently, the moving average at
coverage 3.5 prices it slightly better, and the foundation-model comparison is too small a signal
for eight weeks to settle.

## Amendment 2026-11-21: target representation (queued, not yet run)

The VN1 pilot (`docs/vn1-pilot.md`) found that TabPFN's bad showing there was a target
mis-specification, not a model limit: dividing each sale by its series' 13-week mean and multiplying
back moved it from 0.8391 to 0.6219 on held-out weeks, past the organisers' baseline. VN2's TabPFN
arm above was run on **raw targets**, so its 4,252.2 EUR may rest on the same defect.

Queued for when the API quota resets (2026-12-01; the monthly 5,000,000-token cap is exhausted):

1. `tabpfn_api(target="ratio")` on the **tuning anchors**, coverage 3.5, same folds — one pass, no
   sweep, since the transform is the question and coverage was already settled.
2. If it beats `seasonal_ma` there, the same arm on the **evaluation anchors**, once.
3. If that holds, the **competition anchor**, once. The benchmark arm is unchanged and must still
   score 4,334.0.

The bars do not move: an arm wins only if the paired difference exceeds its fold standard error.
A measured VN2 fit is quoted at the 10,000-token minimum, so the whole sequence costs well under
1M tokens. This amendment was written before any of it was run.

## Costs and quota

TabPFN free tier: 5M tokens/day, 20M/month, 10k-token minimum per operation, reset midnight UTC.
One measured fit+predict at the back-test shape is quoted at exactly that 10,000-token minimum, so
a ~78-fit back-test costs well under 1M tokens. `scripts/smoke_arms.py` re-quotes before a run.

## Reproducing

```bash
scripts/fetch_data.sh                     # competition files into data/raw/
python3 scripts/reproduce_benchmark.py    # must print 913.8 / 4334.0
python3 scripts/smoke_arms.py             # validate the TabPFN key, quote the bill
python3 scripts/backtest.py --anchors tuning
python3 -m pytest -q                      # harness pins and arm tests
```

Keys live in `~/.config/vn2-inventory/keys.env`, never in the repo.
