# Final write-up

What this project did, what the results say, and what a reader could do next. It is the closing
document; the detailed records are in [`method.md`](method.md), [`vn1-pilot.md`](vn1-pilot.md) and
[`chronos-vs-baseline.md`](chronos-vs-baseline.md).

## 1. The question

Two real forecasting competitions, four pretrained foundation models, and one question:

> When does a foundation model beat a simple moving average — and what decides it?

The models came from two families. **Time-series foundation models** (Chronos-Bolt, Chronos-2,
TimesFM-3) take a sequence and forecast its continuation. **Tabular foundation models** (TabPFN)
take rows of features — lags, rolling means, calendar, hierarchy, price — and predict a value. The
competitions differed in what they paid for:

- **VN1** scored 13-week forecasts of 15,053 series by `MAE% + |Bias%|` — pure accuracy.
- **VN2** scored 8 weeks of inventory orders over 599 series by `0.2 € × held + 1 € × lost` — a
  decision under asymmetric costs.

## 2. Method

The credibility of every number rests on four choices:

1. **The harness is pinned to the organisers.** VN2's official simulator is called, never
   re-implemented; `tests/test_play.py` requires the published 913.8 / 4,334.0 and an identical
   weekly table and order file. VN1's scorer reproduces the organisers' reference scores exactly
   (MA12 0.659707, naive 0.688207).
2. **One policy per task.** In VN2 every arm is wrapped in the same order-up-to rule, so a score
   difference is a forecast difference. In VN1 every arm is scored by the same official formula.
3. **Pre-registered selection.** Anything tuned on tuning anchors (VN2) or Phase 1 (VN1) was written
   into the method record before it was run; the held-out boards were scored once.
4. **Full population.** VN1 has 15,053 series and VN2 599; subsamples were used only as pilots, and
   both times their conclusions reversed at full population.

## 3. What was done, in order

1. **Reproduced the VN2 benchmark** exactly, then built a back-test with seven tuning and five
   evaluation windows, each starting from a shared 52-week warm-up state.
2. **Compared TabPFN with Chronos-Bolt** under the shared policy. TabPFN ran through the Prior Labs
   API because a local fit cost 531.7 s and 4,000 rows exhausted the 3.73 GB MPS budget.
3. **Found no foundation-model winner on VN2**, then moved to VN1, where accuracy is the metric.
4. **Replaced Bolt with stronger models** after checking GIFT-Eval: Chronos-2 and TimesFM-3.
5. **Won VN1** with Chronos-2, cross-learning, and a quantile selected on Phase 1.
6. **Returned to VN2** with the winning model and with every remaining lever: quantiles, scales,
   per-series calibration, both forms of newsvendor, blending, and the TabPFN ratio target.

## 4. Results

### VN1 — won

| configuration | score |
| --- | ---: |
| Bolt-small, median | 51.58 |
| Bolt, scale 1.1 | 51.25 |
| **Chronos-2, cross-learning, q=0.56 (chosen on Phase 1)** | **46.02** |
| 1st place, actual competition | 46.37 |
| 2nd | 46.57 |
| naive forecast | 50.70 |
| 12-week moving average | 80.46 |

Three changes produced it, in this order of importance: **the model** (Bolt is last of eight on
GIFT-Eval; Chronos-2 is third), **cross-learning** (all series in one pass), and **the quantile**
(`MAE% + |Bias%|` makes the median wrong by construction — q=0.56 gave MAE 45.15 and bias −0.86
against the winner's 45.25 and +1.12).

### VN2 — lost to a moving average

| arm | tuning | evaluation | competition weeks |
| --- | ---: | ---: | ---: |
| `seasonal_ma` @3.5 | 4,365.3 | 5,739.8 | **3,915.0** |
| Chronos-2, q=0.6, cov 3.0 | 4,626.2 | 6,013.2 | 4,077.2 |
| Chronos-Bolt, best sweep | 4,858.7 | 6,855.7 | 4,237.2 |
| TabPFN, raw targets | 4,581.0 | 6,302.1 | 4,252.2 |
| TabPFN, ratio target | 4,580.9 | — | — |
| benchmark | 4,806.4 | 6,083.0 | 4,334.0 |

Every other lever tried, with its best tuning score: per-series causal calibration 5,433; newsvendor
over summed demand 4,856; model-free bootstrap newsvendor 4,533; moving-average × Chronos-2 blend
4,332 (75% average — inside the ±670 fold spread).

## 5. Analysis: why the answers differ

**VN1 pays for a central forecast; VN2 pays for a level and an allocation.** That is the whole
difference, and it has a measured cause.

Time-series foundation models are **scale-invariant**: feeding Chronos-2 an input multiplied by
three returns a forecast multiplied by exactly 3.0000. That is a design choice — it lets one model
serve series of any magnitude — but it means the model cannot say *how much* to stock, only the shape
of the next eight weeks. VN2's money is in the level, and the 13-week average already supplies it
well: in the top volume decile, which carries 66.5% of all units, the average forecasts 0.97 of
realised demand while Chronos-2 forecasts 0.80.

The competition-week split makes the cost explicit:

| arm | lost sales | holding | total |
| --- | ---: | ---: | ---: |
| moving average | 1,481 | 2,434 | 3,915 |
| Chronos-2 | 2,491 | 1,586 | 4,077 |

The average's errors land where a held unit costs 0.2 €; the model's land where a lost sale costs
1 €. A tabular model *can* see absolute levels, which is why TabPFN was the last hope — but the
ratio target, its biggest win on VN1, was a no-op here (4,580.9 against 4,581.0), because VN1's
problem was an orders-of-magnitude scale spread that VN2 does not have.

Two further lessons from the process:

- **Benchmarks rank models; they do not predict tasks.** GIFT-Eval correctly recommended leaving
  Bolt for Chronos-2, and also ranked TimesFM-3 first — which had the best MAE of anything tested
  (42.15) and the worst bias (−10.96), finishing sixth on VN1.
- **The metric's second term decides.** Both competitions charge for systematic error at full
  weight. A model can tie the winner on MAE and lose by five points, or beat the benchmark and still
  lose to a moving average.

## 6. Leakage: the models may have been trained on these series

Neither competition's data can be certified absent from pretraining, and this bears directly on the
headline result.

- **What the model cards say.** Chronos-2 (released June 2026) trains on a subset of *Chronos
  Datasets* and a subset of *GIFT-Eval Pretrain*, plus synthetic data. TimesFM-3 trains on
  *GIFT-Eval Pretrain* excluding datasets overlapping its benchmark. Neither card enumerates series,
  so a reader cannot check.
- **The timing allows it.** VN1 published its data in October 2024 and VN2 in 2025; Chronos-2 and
  TimesFM-3 were released after both. A public retail dataset could plausibly be in a large
  pretraining corpus, and no check here rules it out.
- **The bias runs one way.** Contamination can only make a model look better than it is. That
  matters differently for the two results:
  - **VN1's win is the at-risk one.** It rests on a 0.35-point margin over the winning entry, on one
    window, and the scored window (Phase 2) is exactly the kind of public data a later model could
    have seen. Treat it as provisional.
  - **VN2's loss is robust to it.** A model that had seen the data would be advantaged, and the best
    foundation arm still lost to a moving average by 162 €. Leakage cannot rescue a negative result;
    it strengthens it.
- **Bolt-small is not exempt either.** It predates VN2's release but only just VN1's, and its
  pretraining corpus is equally unverifiable.

What a reader could do instead: prefer checkpoints whose pretraining cutoff precedes the data
release, consult GIFT-Eval's "test data leakage" column, evaluate on a window after the model's
release date, or ask the model authors for per-dataset inclusion. None of these is conclusive at
series level, which is why the caveat is stated rather than resolved.

## 7. Threats to validity

- **VN1 is one window.** The margin is 0.35 points, q=0.55 scores 46.98, and Phase 2 was examined
  repeatedly during the search. Treat the win as "would have won that window", not "would win".
- **VN2 fold spread is large** (σ ≈ 600–1,000 €). Differences under ~200 € should be read as ties.
- **API models are pinned by name, not checksum.** TabPFN was pinned to `v3.5_default` through the
  Prior Labs API; the open weights are not byte-identical.
- **VN1 cannot identify censoring.** There is no in-stock flag, so a stock-out is indistinguishable
  from low demand; every VN1 arm faces that equally.
- **Not every model was run.** Toto-2.0, Moirai2 and Sundial were ranked but never tested here, and
  the machine had under 6 GB of free disk, a real constraint on further checkpoints.

## 8. Suggested changes for a reader

- **Separate the core from the competitions.** `src/vn1` and `src/vn2` duplicate loading, forecasting
  and scoring logic. A shared `fm_eval` core with task adapters would make a third competition a
  day's work.
- **Cache model tensors by default.** Quantile tensors were cached late; before that, every rerun
  paid 10–20 minutes of inference. Cache keyed by `(model, revision, history length)`.
- **Pin checkpoints by revision hash**, not name, for anything reported.
- **Add CI** that fetches data and runs the pins on every push; today the harness is enforced only
  when someone runs pytest.
- **Record the decomposition, not just the score.** MAE and bias separately; lost sales and holding
  separately. The scores were ambiguous; the decompositions were not.

## 9. Future tasks, in expected-value order

1. **VN2 with a task-fitted model.** The winning entry used ensembled LightGBM on lags, price and
   hierarchy. The same feature table already exists in `src/vn2/features.py`; adding a LightGBM arm
   to the identical harness is the single highest-value experiment left, and it would answer whether
   the winner's edge was the model class rather than the policy.
2. **A second VN1 window.** One window cannot settle a 0.35-point margin. Any future VN1-style
   competition with a second scored phase would settle it.
3. **Fine-tune Chronos-2 on the competition history.** Zero-shot is a constraint, not a limit; a
   LoRA-style fine-tune on 170 weeks of the 15,053 series could improve both MAE and calibration.
4. **TabPFN with price and hierarchy at full VN1 population.** The 500-series sample suggested it
   could compete; the API quota blocked the full run. The ratio target should be standard here.
5. **Allocation-aware VN2 modelling.** Instead of one policy over independent series, model the
   cross-series structure explicitly (shared seasonality, substitute products) and allocate stock
   accordingly — the one dimension no arm in this project used.
6. **Ensembles chosen on error correlation, not MAE alone.** The two strongest VN1 models are both
   biased in the same direction on the full population (Chronos-2 −2.2%, TimesFM-3 −11%), so a blend
   cannot cancel bias — but measuring their per-series error covariance would show whether their
   accuracy errors offset, which is the part worth ensembling.
7. **A rolling-origin protocol with paired bootstrap errors**, replacing fold standard errors, which
   understate uncertainty because the windows overlap.

## 10. Reproduce

```bash
git clone https://github.com/scotthawes/foundation-models-vn1-vn2.git && cd foundation-models-vn1-vn2
scripts/fetch_data.sh && scripts/fetch_vn1_official.sh
python3 -m pytest -q                    # 33 tests
python3 scripts/reproduce_benchmark.py  # VN2: 913.8 / 4334.0
python3 scripts/competition_run.py      # VN2 arms on the played weeks
python3 scripts/vn1_win.py              # VN1: 46.02%, first on the scored window
```

## Citation

> Vandeput, N. (2025). VN2 Inventory Planning Challenge.
> https://supchains.com/vn2-inventory-competition/

> Vandeput, N. (2024). VN1 Forecasting Competition.
> https://supchains.com/vn1-forecasting-competition/
