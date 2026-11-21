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
2. A second scored window would settle whether Chronos's 0.047 edge is real. The files here contain
   only one (Phase 1), so that needs Phase 2 data.
3. Multiple seeds, and stratify by volume rather than sampling series uniformly — the pilot's
   sample-bias lesson.
