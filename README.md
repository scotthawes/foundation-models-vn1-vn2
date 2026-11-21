# Foundation models on two forecasting competitions

Four pretrained models — one tabular, three time-series — tested against simple baselines on two
real competitions. The question is deliberately narrow: **when does a foundation model beat a
moving average, and why?**

| competition | task | scale | question | result |
|---|---|---|---|---|
| **VN1** | 13-week sales forecasts, scored `MAE% + \|Bias%\|` | 15,053 series, 170 weeks | pure forecast accuracy | **won**: Chronos-2 q=0.56 → **46.02%** vs the winning entry's 46.37% |
| **VN2** | 8-week inventory orders, scored `0.2·held + 1·lost` | 599 series, 157 weeks | cost-optimal stocking | **lost**: moving average **3,915 €**, best foundation arm 4,077 € |

Full analysis: [`FINDINGS.md`](FINDINGS.md). Detailed records: [`docs/method.md`](docs/method.md)
(VN2), [`docs/vn1-pilot.md`](docs/vn1-pilot.md) (VN1), [`docs/chronos-vs-baseline.md`](docs/chronos-vs-baseline.md)
(model-vs-baseline comparison).

## The models

| model | family | what it sees | tested on |
|---|---|---|---|
| **TabPFN 3.5** (API) | **tabular** | a row per series-week: lags, rolling means, calendar, hierarchy, price | VN2 (raw targets); VN1 (500-series sample, ratio target) |
| **Chronos-Bolt-small** | time-series | one sequence per series | VN1, VN2 |
| **Chronos-2** | time-series | one sequence per series, optionally all series in one pass (cross-learning), 21 quantiles | VN1 (won), VN2 |
| **TimesFM-3** | time-series | one sequence per series, 9 quantiles | VN1 (best MAE, worst bias) |
| Toto-2.0, Moirai2, Sundial | time-series | — | not run (ranked below or untested here) |

The distinction that matters: **tabular models see absolute levels; time-series models normalise
each series internally and are therefore scale-invariant.** Measured directly on Chronos-2 — a 3×
input produces a 3× output — so a time-series model cannot tell you how much to stock, only the shape
of the next few weeks.

## What was tested

| task | arms tried | best | outcome |
|---|---|---|---|
| VN1 | Bolt median/scale/quantiles; Chronos-2 (cross-learning, 21-level quantiles); TimesFM-3; TabPFN ratio (sample); blends | Chronos-2 q=0.56, chosen on Phase 1 | beats the actual winner by 0.35 points |
| VN2 | benchmark; moving averages; naive; TabPFN raw; Bolt (quantiles, scale, per-series calibration); Chronos-2 (cross-learning, quantiles, newsvendor); model-free bootstrap newsvendor; MA×Chronos-2 blends | `seasonal_ma` @3.5 | wins by 162 € over the best foundation arm |

Every VN2 arm shares one ordering policy, so differences are differences in forecast (or in level).

## Findings

1. **The task decides more than the model.** The same Chronos-2 that won VN1 lost VN2 by 162 €.
   VN1 pays for a central forecast; VN2 pays for a per-series level and allocation.
2. **The metric's second term is where wins are lost.** VN1 is `MAE% + |Bias%|`: Chronos's median
   matched the winner's MAE (45.36 vs 45.25) and still lost by five points on bias until a quantile
   was selected on the validation window. VN2 is 1 € lost against 0.2 € held: an under-stocked
   median costs five times what an over-stocked mean does.
3. **Scale invariance blocks the inventory use case.** A time-series foundation model cannot supply
   the level that VN2's cost function prices. Normalising its input is a measured no-op; the level
   must come from a model that sees absolute values (tabular) or from a baseline.
4. **Benchmarks rank models, they do not predict tasks.** GIFT-Eval correctly said to leave
   Bolt-small (last of eight, mean rank 6.3) for Chronos-2 (3.4) — and also ranked TimesFM-3 first
   (2.2), which lost VN1 on bias despite the best MAE of anything tested.
5. **Subsamples mislead.** A 500-series VN1 pilot and a 2,000-series TimesFM-3 run both reported
   orderings that reversed at full population.

## Recommendations

- **For accuracy-scored forecasting:** a high-ranked time-series model plus a quantile chosen on a
  held-out window. The median is not the right point forecast when the metric charges bias.
- **For inventory or any level-priced decision:** a tabular model with a normalised target, or a
  tuned moving average. Check scale invariance first; if the model returns 3× for a 3× input, it
  cannot set the level.
- **For judging any of it:** full population, several windows, paired standard errors, and the
  selection rule written down before the result is seen. Two of this project's apparent results
  reversed once that discipline was applied.

## What is still open

- **TabPFN with a ratio target on VN2** — implemented and pre-registered
  ([`docs/method.md`](docs/method.md)) but not yet run: the Prior Labs API's monthly quota is spent
  until 2026-12-01. It is the one foundation-model route that can address VN2's level defect.
- **A second scored window for VN1** — the win is one window and 0.35 points; Phase 2 data for a
  second window is not public.

## Reproduce

```bash
git clone https://github.com/scotthawes/foundation-models-vn1-vn2.git && cd foundation-models-vn1-vn2

scripts/fetch_data.sh              # VN2 data
scripts/fetch_vn1_official.sh      # VN1 Phase 2 actuals, winners' forecasts, official scorer

python3 -m pytest -q               # 33 tests: benchmark pins, leakage pins, arm tests
python3 scripts/reproduce_benchmark.py   # VN2 benchmark: 913.8 / 4334.0
python3 scripts/competition_run.py       # VN2 arms on the played weeks
python3 scripts/vn1_win.py               # VN1: 46.02%, first on the scored window
```

Keys (only needed for the TabPFN arm) live outside the repo:

```bash
# ~/.config/vn2-inventory/keys.env   (chmod 600)
export TABPFN_TOKEN="..."
```

## Layout

```
src/vn1/            VN1 package: loader, score, features, forecasters
  data.py, score.py, features.py
  forecaster.py     ma12, naive, chronos (Bolt), chronos2, timesfm3
src/vn2/            VN2 package: harness, back-test, policy, forecasters
  data.py           competition shapes and the censoring rule
  play.py           the official loop and the shared order-up-to policy
  backtest.py       folds, warm-up states, tuning vs evaluation anchors
  features.py       the tabular feature table
  level_policy.py   newsvendor level from summed demand
  forecaster/       one module per arm (seasonal, naive, tabpfn, tabpfn_api, chronos, chronos2,
                    calibrated, timegpt)
scripts/            fetch, reproduce, smoke, grids, competition run, reproductions
tests/              33 tests: benchmark pins, leakage pins, arm frame and arithmetic tests
docs/               detailed method records and the model-vs-baseline analysis
FINDINGS.md         what was tested, found, and recommended
```

## Status

- VN2 benchmark reproduced exactly (913.8 / 4,334.0) and pinned by tests.
- VN1 scorer reproduces the organisers' reference scores (MA12 0.659707, naive 0.688207).
- 33 tests pass; the numbers in `FINDINGS.md` come from the CSVs under `results/`
  (git-ignored, regenerated by the scripts in `scripts/`).
- No competition data or API keys are stored in the repo.

## Citation

Both competitions ask that their data be cited:

> Vandeput, N. (2025). VN2 Inventory Planning Challenge.
> https://supchains.com/vn2-inventory-competition/

> Vandeput, N. (2024). VN1 Forecasting Competition.
> https://supchains.com/vn1-forecasting-competition/
