# Can Chronos beat the moving average?

The question, asked on both tasks:

| task | baseline | best Chronos | verdict |
| --- | --- | --- | --- |
| VN2 (inventory cost, EUR/window) | `seasonal_ma` @3.5, **5,739.8** on evaluation folds | 6,828.4 | **no** — loses by 1,088.6 ± 497.0 (t 2.19) |
| VN1 (forecast error, lower better) | MA12, **0.6597** on the scored weeks | **0.5502** | **yes** — wins by 0.1095 (16% relative) |

## VN2: no, after giving it every lever the average lacks

VN2's costs are asymmetric — 1 EUR per lost sale against 0.2 EUR per unit held — so the
cost-minimising stock level is the 0.833 quantile of demand. Chronos predicts nine quantiles and
the original comparison fed it the median. The arm now exposes `quantile` and `scale`, and a
pre-registered grid swept `quantile ∈ {0.5, 0.7, 0.8, 0.9}` × `scale ∈ {1.0, 1.1}` ×
`coverage ∈ {3, 3.5, 4}` on the tuning anchors.

The best cell was the configuration already in use: median, scale 1, coverage 3.5 — 4,858.7. Every
quantile above the median was worse. A post-hoc second grid tested the actual newsvendor shape,
a high quantile with short coverage (`quantile ∈ {0.6..0.9}` × `coverage ∈ {1.5, 2, 2.5, 3}`); its
best was quantile 0.6 at coverage 3.0, **4,809.9** on tuning — a 49 EUR gain over the original and
still 445 behind `seasonal_ma` at 4,365.3. On the evaluation anchors that best cell scores 6,828.4
against `seasonal_ma`'s 5,739.8.

Per fold (evaluation anchors):

| anchor | `seasonal_ma` @3.5 | Chronos q0.6 @3.0 |
| ---: | ---: | ---: |
| 132 | 8,784 | 11,472 |
| 136 | 7,483 | 9,061 |
| 140 | 4,753 | 5,804 |
| 144 | 4,048 | 4,251 |
| 148 | 3,631 | 3,554 |

Chronos wins only the last window and loses every other. The pre-registered rule said the
competition anchor is played only if the evaluation result beats `seasonal_ma`; it does not, so it
was not played.

There is a plausible mechanism rather than a mystery: at equal coverage Chronos stocks less
(2,541 lost sales against 1,481) and holds less, and VN2 prices a lost sale five times a held unit.
A moving average with coverage 3.5 is a cost-aware policy by construction; Chronos's forecast is not
calibrated to that asymmetry, and its quantiles are wide enough that buying service level with them
costs more in holding than it recovers in shortages.

## Why Chronos loses on VN2: measured

**It is not accuracy.** Over the played weeks, total absolute forecast error is 7,590 for Chronos
and 7,648 for the moving average — Chronos is marginally the better forecaster in the unweighted
sense. It loses because the errors are in the wrong place and have the wrong sign.

The ordering policy is written for a **mean** demand level. The moving average estimates the mean
directly. Chronos returns the **median**, and for right-skewed intermittent demand the median sits
below the mean by a factor that varies with each series' skew. Over the played weeks Chronos
forecasts 0.79 of realised demand in aggregate; the moving average forecasts 1.01.

That bias is not spread evenly, and this is the part that decides the score — forecast/actual ratio
by volume decile:

| volume decile | share of units | MA ratio | Chronos ratio |
| ---: | ---: | ---: | ---: |
| 7 | 5.8% | 0.82 | 0.59 |
| 8 | 8.9% | 0.92 | 0.74 |
| **9** | **66.5%** | **0.97** | **0.80** |

Two-thirds of the units live in decile 9, where the moving average is calibrated (0.97) and Chronos
is 20% low. So the moving average's errors are concentrated on low-volume series where a held unit
costs 0.2 EUR, and Chronos's are concentrated where a lost unit costs 1 EUR.

A **single global scale cannot fix a per-series bias**, which is why the wider grid failed
monotonically. What scaling buys, on the played weeks:

| Chronos configuration | score | lost sales | holding cost |
| --- | ---: | ---: | ---: |
| median, coverage 3.5 | 4,237.2 | 2,541 | 1,696.2 |
| scale 1.3, coverage 3.0 | 4,166.6 | 2,005 | 2,161.6 |
| scale 1.3, coverage 3.5 | 4,423.8 | 1,461 | 2,962.8 |

Lifting the forecast by 30% removes 536 lost sales for 466 EUR of extra holding — near break-even,
because it adds stock to the low-volume series that were already over-forecast while only partly
fixing the high-volume ones. The remaining lever is therefore **per-series calibration**: estimate
each series' forecast bias on earlier windows and correct that series only. That is untested here,
and it is the only lever the diagnosis leaves open.

**It was tested, and it fails too.** `calibrated_chronos` estimates each series' realised/forecast
ratio on the four most recent 8-week windows that end before the round (censored weeks excluded,
ratio shrunk toward 1 by 2 pseudo-units, clipped to 0.5-2.5). The factors are noisy: median 1.16,
mean 1.33, 10% of series pinned at the 2.5 cap and 37% below 1 — because three windows of
eight-week-ahead forecasts are a thin basis for a per-series ratio.

| configuration | tuning mean | played weeks | lost sales | holding |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated, coverage 3.5 | 4,858.7 | 4,237.2 | 2,541 | 1,696.2 |
| calibrated, coverage 3.0 | 5,433.4 | 4,907.0 | 3,093 | 1,814.0 |
| calibrated, coverage 3.5 | 5,761.3 | 4,821.0 | 2,461 | 2,360.0 |

Calibration adds stock and does not buy service: at coverage 3.5 it costs 664 EUR more in holding
and removes only 80 lost sales, because the uplift goes to series whose ratio was estimated from
noise while the high-volume series are corrected slightly or, for the 37% with a ratio below one,
downward.

**The correct newsvendor level does not help either.** The per-week quantile attempt summed
quantiles, which is not the quantile of the total. `vn2.level_policy` does it properly: sample 256
demand paths from the nine per-week quantiles, sum each path over the coverage window, order the
0.833 quantile of the totals. On the tuning anchors:

| coverage | service 0.7 | service 0.833 | service 0.9 |
| ---: | ---: | ---: | ---: |
| 2.0 | 5,285.3 | 5,064.3 | 5,096.7 |
| 2.5 | **4,856.3** | 4,972.9 | 5,263.8 |
| 3.0 | 4,929.1 | — | — |
| 3.5 | 5,377.5 | 6,228.3 | 7,042.1 |

The best cell, 4,856.3, is within 2.4 EUR of the plain median configuration (4,858.7) and 491
behind `seasonal_ma` (4,365.3). The theoretically implied service level is *worse* than a lower one.

**So the aggregate level was never the binding constraint.** Every level lever — quantile, global
scale, per-series calibration, a correctly built newsvendor — lands Chronos at roughly 4,850-4,950
on tuning, close to the benchmark's 4,806 and consistently below the moving average. What differs is
the *allocation across series*: the moving average is a per-series mean, so its errors are spread
proportionally and land in cheap places, while Chronos's median carries a skew-dependent per-series
bias that no global factor removes and that a three-window per-series estimate is too noisy to
correct.

The mismatch is structural, not a tuning failure: the policy needs a calibrated per-series mean, and
a zero-shot median is not one. A model fitted to this task — fine-tuned on ordering outcomes, or a
per-series model with enough history to estimate its own bias — is the only route left, and it is
outside what a zero-shot comparison can claim.

## VN1: yes, and already at the median

VN1's score is absolute error plus a bias penalty, so the median is the right forecast and the only
free parameter is calibration. On all 15,053 series:

| Chronos scale | validation | final (scored) |
| ---: | ---: | ---: |
| 1.0 | 0.5222 | 0.6123 |
| **1.1** | **0.4981** | **0.5502** |
| 1.2 | 0.6356 | 0.6355 |
| 1.3 | 0.7840 | — |
| MA12 (organisers') | 0.5163 | 0.6597 |

Scale 1.1 was chosen on validation and beaten nothing on the way out: it wins on both origins,
against MA12 and against the uncalibrated arm. The optimum is narrow — 1.2 gives back the whole
gain and 1.3 is worse than the median — which is consistent with VN1's forecast bias being a
level error that a single global factor corrects.

## Why the answers differ

The two tasks reward different things. VN1 rewards an accurate central forecast, and Chronos's is
better than a 12-week mean plus a global level correction. VN2 rewards a *stock level* under
asymmetric costs, where the moving average is already tuned and the extra machinery of a foundation
model is spent on quantiles that cost more than they earn.

One caveat on the VN1 win: it is a single scored window (Phase 1). The scale was selected on
validation, but there is no second window to check that 0.55 replicates.
