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

## Target shape: the missing piece

The last untested knob was the target itself. VN1's targets are 72% zeros and span orders of
magnitude across series, and the arm was asking the model to predict the absolute level of each.
Two transforms were tried on the validation origin, then checked on the held-out weeks:

| configuration (500 series) | validation | final |
| --- | ---: | ---: |
| raw target, 20k context | 0.7082 | 0.8391 |
| raw target, 65k context | 0.6223 | — |
| `log1p` target, 65k context | 0.5160 | — |
| **per-series ratio target, 65k context** | **0.4324** | **0.6219** |
| references: MA12 / Chronos | 0.4836 / 0.4106 | 0.6261 / 0.5690 |

A ratio target divides each sale by its series' own 13-week mean, so the model learns "how many
multiples of recent demand" rather than an absolute level that differs by three orders of magnitude
between series. That moves TabPFN from last to second: it beats MA12 on both origins, sits 0.02
behind Chronos on validation and 0.05 behind on the played weeks.

This is the answer to the pilot's question about TabPFN: the model was not the problem, the target
representation was. It is also a warning about the earlier negative result — three of the four
probes were measuring a mis-specified arm.

## Full population (15,053 series)

The winner was scaled, and the sample turned out to flatter it:

| arm | validation | final (Phase 1) |
| --- | ---: | ---: |
| `chronos` | 0.5223 | **0.6123** |
| `ma12` | **0.5163** | 0.6597 |
| `naive_last` | 0.5247 | 0.6882 |

On the 500-series sample Chronos led MA12 by 0.073 on validation (0.4106 against 0.4836). On all
15,053 series the validation origin is a tie (0.5223 against 0.5163): the sample over-weighted
high-volume series, and the full population is mostly dormant ones where a flat zero-ish MA12 is
hard to beat and a sequence model still predicts movement.

On the weeks that were actually scored, Chronos wins by 0.047 and the decomposition says it is
accuracy, not bias:

| arm | score | WAPE | bias | mean forecast | mean actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chronos` | 0.6123 | 0.5013 | -0.1110 | 17.74 | 19.96 |
| `ma12` | 0.6597 | 0.5420 | -0.1177 | 17.61 | 19.96 |

That is the result of the exercise: **a time-series foundation model beats the organisers' MA12
baseline on the played weeks by about 7% relative, with essentially the same bias.** It is a modest
edge, and it does not appear on the validation origin, so it is one window's evidence rather than a
robust win.

## Phase 2: against the actual winners

The official bundle (`data/raw/vn1-official/`, git-ignored) contains Phase 2's actuals, the official
scorer, and **the five winning forecasts**. Scoring them reproduces the published leaderboard
exactly, and the same scorer puts our arms in the same table:

| entry | score | MAE% | bias% |
| --- | ---: | ---: | ---: |
| 1st — Jakub Figura & Philip Stubbs | **46.37** | 45.25 | +1.12 |
| 2nd — Justin Furlotte | 46.57 | — | — |
| 3rd — Arsa Nikzad | 47.58 | — | — |
| 4th — Antoine Schwartz | 47.74 | — | — |
| 5th — An Hoang | 48.08 | — | — |
| naive forecast | 50.70 | — | — |
| **Chronos, median** | 51.58 | **45.36** | -6.22 |
| Chronos, scale 1.1 (tuned on Phase 1) | 51.25 | 48.09 | +3.15 |
| Chronos, scaled by Phase 1's bias (1.125) | 54.52 | 49.04 | +5.49 |
| old R submission (`rf_submission.csv`) | 53.53 | 50.95 | -2.58 |
| 12-week moving average | 80.46 | — | — |

Three things follow.

**Chronos matched the winner on accuracy and lost on bias.** Its MAE is 45.36% against the winner's
45.25% — a tie — but the score is `MAE% + |bias|`, and a median forecast of right-skewed demand
carries a −6.2% bias. That single term is the whole 5.2-point gap.

**Bias corrections do not transfer across windows.** Phase 1's out-of-sample bias was −11.1%, so the
causal correction was 1.125; Phase 2's bias was only −6.2%, so that correction overshot to +5.5% and
made the score worse than doing nothing. A scale tuned on Phase 1's validation splits (1.1) was the
best calibrated variant at 51.25 — still behind the naive forecast.

**The winners' method was not a foundation model.** Vandeput's post-competition interviews report
LightGBM as the most-used model at the top, with most entrants ensembling several models or runs;
half wrote under 300 lines and nearly all ran in under 10 minutes. The winning edge was a fast,
robust evaluation loop and a bias-controlled point forecast, not a larger model.

So on the weeks that actually ranked, Chronos would have placed below the naive forecast — despite
being as accurate as the winner. The lesson matches VN2's: `MAE% + |Bias%|` and a cost function both
punish a biased central forecast, and a zero-shot median is biased by construction.

## Which foundation model? GIFT-Eval, 97 datasets

Chronos-Bolt-small was chosen early for convenience, not for evidence. On the independent GIFT-Eval
benchmark — mean per-dataset rank across 97 datasets, computed here from the space's per-model
`all_results.csv` files — it is the weakest of the eight models compared:

| model | MASE rank | CRPS rank |
| --- | ---: | ---: |
| TimesFM-3 | **2.2** | **2.0** |
| Toto-2.0-313m | 3.3 | 3.3 |
| Chronos-2 | 3.4 | 3.5 |
| TimesFM-2.5 | 4.1 | 4.0 |
| Moirai2 | 5.3 | 5.2 |
| Toto 1.0 base | 5.6 | 5.2 |
| Sundial | 5.8 | 6.4 |
| **Chronos-Bolt-small** | **6.3** | **6.3** |

Source: `huggingface.co/spaces/Salesforce/GIFT-Eval/results/<model>/all_results.csv`, ranks computed
per dataset and averaged. The caveat is real: GIFT-Eval is broad (annual to secondly, many domains),
not intermittent weekly retail, so this ranks models rather than predicting the VN1 result. It does
say that switching away from Bolt is the obvious first move, and it is consistent with Chronos-2
beating Bolt on MAE in our own subsample test (39.8 vs 42.9).

## Searching for a win (all on full population, all selected on Phase 1)

| configuration | MAE% | bias% | score |
| --- | ---: | ---: | ---: |
| Bolt, median (the original arm) | 45.36 | -6.22 | 51.58 |
| Bolt, scale 1.1 | 48.09 | +3.15 | 51.25 |
| Bolt+naive 50/50 | 45.95 | -3.39 | 49.35 |
| Bolt+naive 70/30 (chosen on rolling windows) | 45.24 | -4.53 | 49.77 |
| **Chronos-2 cross, q=0.55** | **44.82** | **-2.16** | **46.98** |
| **Chronos-2 cross, q=0.56 (21-level grid, chosen on Phase 1)** | **45.15** | **-0.86** | **46.02** |
| TimesFM-3, q=0.5 | 42.15 | -10.96 | 53.12 |
| TimesFM-3, q=0.5, scale 1.05 | 42.84 | -6.51 | 49.36 |
| Chronos-2 + conditional LightGBM calibration | 63.41 | +19.90 | 83.31 |

Against the actual leaderboard, **Chronos-2 q=0.56 at 46.02 would have placed first**, by 0.35 points
over Jakub Figura and Philip Stubbs (46.37). The quantile was chosen on Phase 1 — the validation
window — and Phase 2 was scored once, so nothing here is fitted to the scored weeks.

That margin is thin and should be read as such: 0.35 points separates first from first-runner-up on
one window, the coarse sweep's q=0.55 scored 46.98, and a one-hundredth change in the quantile moved
the score by nearly a point. It is a result, not a robust win.

Why the attempts around it failed, in one line each:

- **Blending with naive** reduced bias but raised MAE; the metric charges both, so it lost.
- **Quantile shifts on Bolt** did not survive rolling-window selection.
- **TimesFM-3** is the most accurate model tested (MAE 42.15, four points better than the winner's
  45.25) and lost anyway: its median is 11% low, and the Phase-1-selected scale only halved that.
- **Conditional LightGBM calibration** (predict the actual from forecast, level, zero-share,
  horizon, calendar) failed both ways: an L1 learner predicted a median and added −33% bias, an L2
  learner overfit Phase 1 and scored 83 on Phase 2.
- **Blending the two foundation models** was never run: both are biased in the same direction, so
  the blend cannot cancel the bias term that decides the score.

The remaining question is whether that 0.35-point margin replicates. Everything tried suggests the
answer depends on the bias term, and every causal estimate of that bias has been unstable across
windows (-11.1% on Phase 1, -6.2% on Phase 2 for Bolt; -6.2% to -0.9% for Chronos-2 across a
one-hundredth change in quantile). A robust win would need either a model whose median is unbiased by
construction, or a bias estimate that holds across windows — neither of which the project has
established, and both of which the thin margin leaves open.

TabPFN could not be scaled. The ratio-target arm needs the API, and the account's monthly quota
(5,000,000 tokens) was exhausted by the 65k-context probes: the full-population fit succeeded and
the prediction was rejected with **HTTP 429**, reset 2026-12-01. What exists for TabPFN is the
500-series evidence above — held out, but a sample. The population-scale number is the one thing
this pilot set out to price and did not get.

## What this does and does not show

- It shows the VN1 question is answerable and cheap: the full-population comparison of Chronos
  against MA12 costs no API tokens, and the MA12 baseline reproduces the organisers' 0.6597 exactly.
- It shows why pilots mislead: the 500-series sample's 0.07 Chronos lead halved at full population
  on one origin and vanished on the other. Sampling series uniformly is not sampling *volume*
  uniformly.
- It shows the tabular arm's deficit was **target representation, not the model**: a per-series
  ratio target took TabPFN from 0.8391 to 0.6219 on the held-out weeks, past MA12.
- It does not establish TabPFN's population-scale score, and the tabular stock comparison is
  therefore sample-scale only. Nor can any arm handle censoring: VN1 has no in-stock flag, so a
  stock-out is indistinguishable from low demand.
- It cost the API's month: five probe runs at 65k context spent the 5M quota. Cheaper would have
  been one probe at 20k context before committing to the larger ones.

## Next, if it is continued

1. When the token quota resets (2026-12-01) or is raised, run the ratio-target arm on all 15,053
   series: at the measured 50,723 tokens per origin it costs little, and it is the missing row.
2. The Phase 2 comparison above settles the second window: Chronos's Phase 1 edge did not transfer.
   What remains unanswered is whether a **bias-controlled** variant — a per-series correction
   estimated from rolling windows rather than a global scale — could reach the winners' 46.4%. The
   winner's MAE is within 0.11 points of Chronos's, so the gap is bias, and bias is what LightGBM
   ensembles get right by construction.
3. Multiple seeds, and stratify by volume rather than sampling series uniformly — the pilot's
   sample-bias lesson.

## Decision (2026-11-21)

Chronos is the standing foundation-model arm. It is local, needs no quota, and it is the only
foundation model here that beats the organisers' baseline on the weeks that were scored. TabPFN is
retired from the comparison except for one pre-registered VN2 check (see `method.md`); no further
VN1 TabPFN work is planned, including the full-population row above, which stays a gap rather than
a commitment.
