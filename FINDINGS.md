# Findings

Two forecasting competitions, one question: can a foundation model beat a simple baseline? The
short answer is **yes on VN1 and no on VN2** — and the reasons are more useful than either answer.

## VN1: a foundation model wins the scored window

VN1 scored 13-week forecasts of 15,053 retail series with `MAE% + |Bias%|`, and ranked only the
Phase 2 window (Jan-Apr 2024).

| entry | score |
| --- | ---: |
| **Chronos-2, cross-learning, quantile 0.56 (this repo)** | **46.02** |
| 1st — Jakub Figura and Philip Stubbs | 46.37 |
| 2nd — Justin Furlotte | 46.57 |
| 3rd — Arsa Nikzad | 47.58 |
| naive forecast | 50.70 |
| 12-week moving average | 80.46 |

Three things produced it, in order of importance:

1. **The model.** Chronos-Bolt-small — the arm this project started with — is last of eight
   foundation models on GIFT-Eval (mean rank 6.3); Chronos-2 is third (3.4). Switching took the score
   from 51.58 to 46.98 with no other change.
2. **Cross-learning.** Chronos-2 forecasts all 15,053 series in one joint pass rather than one
   sequence at a time.
3. **The quantile.** `MAE% + |Bias%|` makes the median wrong by construction. Selecting q on
   Phase 1 from Chronos-2's full 21-level grid gave q=0.56, MAE 45.15 and bias −0.86 — against the
   winner's 45.25 and +1.12.

**How much to trust it.** The margin is 0.35 points on one window, and q=0.55 scores 46.98: a
one-hundredth change in the quantile moves the score by nearly a point. The selection was made on
Phase 1 and Phase 2 was scored once, but Phase 2 was examined repeatedly during the search. The fair
claim is that this configuration would have won that window, not that it would win again.

**What failed** (all on full population, all selected on Phase 1): blending with the naive forecast
(49.35), global scaling (51.25), per-series causal calibration (worse), conditional LightGBM
calibration (−33% or +20% bias depending on the loss), and TimesFM-3 — the most accurate model tested
(MAE 42.15 against the winner's 45.25) which lost on an 11% low bias.

## VN2: the moving average wins

VN2 scored inventory cost over eight played weeks, with a published benchmark of 4,334.0 EUR.

| arm | score (EUR, lower better) |
| --- | ---: |
| `seasonal_ma` at coverage 3.5 | **3,915.0** |
| Chronos at coverage 3.5 | 4,237.2 |
| TabPFN (API, raw targets) | 4,252.2 |
| benchmark (the competition's own policy) | 4,334.0 |

The harness reproduces the benchmark exactly (913.8 setup + 4,334.0), which is what licenses every
other number here.

**Why the foundation model loses.** Not accuracy: over the played weeks Chronos's total absolute
error is 7,590 against the moving average's 7,648. It loses because the ordering policy is written
for a **per-series mean** and Chronos returns a **median**: 0.79 of realised demand in aggregate
against the average's 1.01. The bias is concentrated where the money is — decile 9 holds 66.5% of
all units, where the average forecasts 0.97 of demand and Chronos 0.80 — so its errors land where a
lost sale costs 1 EUR, and the average's land where a held unit costs 0.2 EUR.

**What failed:** per-week quantiles (the newsvendor fractile is 0.833, and every level above the
median was worse), a global scale (trades ~0.9 MAE per point of bias removed), per-series
calibration (three windows of history give a noisy ratio), and a correctly built newsvendor over the
summed demand (4,856.3 on the tuning anchors, against the moving average's 4,365.3 there).

## Lessons that transfer

- **The metric decides.** VN1's bias term and VN2's cost asymmetry both punish the same thing: a
  biased central forecast. A model can tie the winner on accuracy and lose by five points.
- **Task fit beats model family.** VN2 needs allocation across series; VN1 needs a central forecast.
  The same model wins one and loses the other.
- **Benchmarks rank models, they do not predict tasks.** GIFT-Eval correctly said to leave Bolt for
  Chronos-2, and also ranked TimesFM-3 first — which lost here on bias.
- **Subsamples mislead.** The 500-series VN1 pilot and the 2,000-series TimesFM-3 run both reported
  results that reversed at full population.

## Reproduce

```bash
git clone https://github.com/scotthawes/vn2-inventory.git && cd vn2-inventory
scripts/fetch_data.sh                     # VN2 competition data
scripts/fetch_vn1_official.sh             # VN1 Phase 2 actuals, winners' forecasts, official scorer
python3 -m pytest -q                      # 33 tests
python3 scripts/reproduce_benchmark.py    # VN2: 913.8 / 4334.0
python3 scripts/competition_run.py        # VN2: all arms on the played weeks
python3 scripts/vn1_win.py                # VN1: 46.02, first place on the scored window
```

VN1's data is downloaded from the competition page and cited in the README; VN2's is fetched by
script. Neither is redistributed here.
