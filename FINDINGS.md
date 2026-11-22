# Findings

Four foundation models — one tabular, three time-series — against simple baselines on two
competitions. The short version: **the time-series model won the accuracy task and lost the
inventory task, and the reason is structural rather than a tuning failure.**

- VN1 (forecast accuracy): **won** — Chronos-2, quantile 0.56, 46.02% against the winning entry's
  46.37%.
- VN2 (inventory cost): **lost** — `seasonal_ma` 3,915 € against Chronos-2's 4,077 €, with the best
  tabular arm at 4,252 €.

## The models and what they can see

| model | family | unit of input | absolute level? |
|---|---|---|---|
| TabPFN 3.5 | tabular | one row per series-week: lags, means, calendar, hierarchy, price | **yes** |
| Chronos-Bolt-small | time-series | one sequence per series | no |
| Chronos-2 | time-series | sequence, optionally all series jointly; 21 quantiles | no |
| TimesFM-3 | time-series | sequence; 9 quantiles | no |

The "absolute level" column is measured, not assumed. Feeding Chronos-2 an input multiplied by
three returns a forecast multiplied by exactly 3.0000. Time-series foundation models normalise each
series internally — that is how one model serves a bakery and a refinery — and the consequence is
that they forecast shape, never magnitude.

That single fact explains VN2. Its cost function prices the level: 1 € per lost sale against 0.2 €
per unit held. A moving average of 13 weeks already estimates that level well (0.97 of realised
demand in the top volume decile, which carries two-thirds of all units). A model that only knows
shape can add trajectory information and nothing else — and blending 25% of it into the average
moved the score by 34 €, inside a ±670 € fold spread.

## VN1: how the win was found

| step | score |
| --- | ---: |
| Bolt-small, median | 51.58 |
| Bolt + global scale 1.1 | 51.25 |
| Bolt × naive blend | 49.35 |
| **Chronos-2, cross-learning, q=0.55** | 46.98 |
| **Chronos-2, cross-learning, q=0.56 (chosen on Phase 1)** | **46.02** |
| winner — Figura & Stubbs | 46.37 |
| TimesFM-3, q selected on Phase 1 | 53.12 |

Three changes, in order of importance:

1. **The model.** Bolt-small is last of eight zero-shot models on GIFT-Eval (mean rank 6.3);
   Chronos-2 is third (3.4). Switching took 51.58 → 46.98 with nothing else altered.
2. **Cross-learning.** All 15,053 series forecast in one joint pass rather than one at a time.
3. **The quantile.** `MAE% + |Bias%|` makes the median wrong by construction. Choosing q on Phase 1
   from Chronos-2's 21-level grid gave q=0.56: MAE 45.15, bias −0.86, against the winner's 45.25 and
   +1.12. The model's *accuracy* already matched the winner's; the bias term was the gap.

**Caveats.** The margin is 0.35 points on one window, and q=0.55 scores 46.98 — one hundredth of a
quantile moves the score by nearly a point. The selection was made on Phase 1 and Phase 2 was scored
once, but Phase 2 has been examined repeatedly during the search. The claim is that this
configuration would have won that window, not that it would win again.

## VN2: why the foundation models lost

| arm | tuning anchors | evaluation anchors | competition weeks |
| --- | ---: | ---: | ---: |
| `seasonal_ma` @3.5 | 4,365.3 | 5,739.8 | **3,915.0** |
| Chronos-2, cross-learning, q=0.6, cov 3.0 | 4,626.2 | 6,013.2 | 4,077.2 |
| Chronos-Bolt, best of sweeps | 4,858.7 | 6,855.7 | 4,237.2 |
| TabPFN, raw targets | 4,581.0 | 6,302.1 | 4,252.2 |
| benchmark (the competition's own) | 4,806.4 | 6,083.0 | 4,334.0 |

The ordering is identical on all three boards, which is stronger evidence than the single scored
window. The competition-week split shows why:

| arm | lost sales | holding | total |
| --- | ---: | ---: | ---: |
| `seasonal_ma` | 1,481 | 2,434 | 3,915 |
| Chronos-2 | 2,491 | 1,586 | 4,077 |

The average over-stocks and the model under-stocks, and the cost ratio prices those errors five to
one. By volume decile the pattern is sharp: in decile 9 (66.5% of units) the average forecasts 0.97
of demand and Chronos 0.80; in the bottom decile the average is at 1.82 and Chronos 1.15. The
average's errors land where a held unit costs 0.2 €; the model's land where a lost sale costs 1 €.

### Every VN2 lever tried

| strategy | result | verdict |
| --- | --- | --- |
| TabPFN, raw targets | 4,252 € competition | loses by 337 |
| TabPFN, ratio target | 4,581 tuning | no change from raw (0.1 €); loses by 194 |
| Bolt: per-week quantiles, global scale | best 4,859 tuning | loses by 494 |
| Bolt: per-series causal calibration | 5,433 tuning | worse by 1,068 |
| Chronos-2: quantiles 0.5–0.7, coverage 3–3.5 | best 4,626 tuning | loses by 261 |
| Newsvendor over summed demand (0.833 fractile) | 4,856 tuning | loses by 491 |
| Model-free bootstrap newsvendor | 4,533 tuning | loses by 168 |
| Blend: moving average × Chronos-2 | 4,332 tuning at 75% average | 34 better, inside noise |

Nothing moved the ordering. The blend wanting three-quarters of the average, and the model-free
newsvendor losing to the average, are the two strongest signals that the missing piece is *not* the
level-setting mechanism and *not* the forecast shape — it is the per-series allocation that a
tabular model with absolute values could in principle learn.

## Cross-cutting lessons

1. **Task fit beats model family.** The same model won one competition and lost the other. Before
   choosing a model, ask what the metric prices: accuracy (a central forecast) or a decision (a
   level, a stock, a threshold).
2. **The bias term decides.** Both competitions charge for systematic error at full weight. A model
   can tie the winner on MAE and lose by five points, or beat the benchmark and still lose to a
   moving average.
3. **Scale invariance is a hard constraint.** Check it in one call: multiply the input by three. If
   the output triples, the model cannot set a level, and no calibration or quantile will change
   that.
4. **Benchmarks rank models, they do not predict tasks.** GIFT-Eval got the Bolt → Chronos-2 swap
   right and also ranked TimesFM-3 first — which had the best MAE of anything tested here (42.15)
   and the worst bias (−10.96), finishing sixth.
5. **Subsamples mislead.** The 500-series VN1 pilot and the 2,000-series TimesFM-3 run both reported
   orderings that reversed at full population. Full population or nothing.

## Recommendations

- **Accuracy-scored forecasting:** take a high-ranked time-series model and calibrate the point
  forecast by choosing a quantile on a held-out window. Expect the gain to be in bias, not MAE.
- **Inventory or any level-priced decision:** a tabular model with a normalised target (the ratio
  target that transformed VN1's tabular arm), or a tuned moving average. Do not expect a
  scale-invariant model to supply the level.
- **Discipline that paid off here:** pre-register the selection rule, score the held-out window
  once, and use paired fold errors. Twice, an apparent result reversed once that was applied.

## Leakage caveat

The time-series models were released after VN1 (October 2024) and VN2 (2025) published their data,
and their model cards name corpora — *GIFT-Eval Pretrain*, *Chronos Datasets* — without enumerating
series. Training-data contamination therefore cannot be ruled out. Because contamination can only
flatter a model, it does not weaken VN2's loss (a possibly-advantaged model still lost to a moving
average) but it does make VN1's 0.35-point win provisional. See §6 of
[`docs/writeup.md`](docs/writeup.md).

## Still open

- **A second scored window for VN1** — needed to know whether the 0.35-point margin replicates; the
  Phase 2 data for a second window is not public.
- **VN2 has no open foundation-model route.** The ratio target, the last one, was run on 2026-11-22
  and changed nothing (4,580.9 against 4,581.0 raw). A task-fitted tabular model — LightGBM on the
  same feature table, which is what the winning entry used — is the remaining idea, but it is outside
  the foundation-model comparison this repo set out to make.

## Reproduce

See the README's Quickstart. Every number above comes from a script in `scripts/` and a CSV under
`results/` (git-ignored, regenerated on demand).
