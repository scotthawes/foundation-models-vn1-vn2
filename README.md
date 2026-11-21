# VN2 Inventory Planning: a tabular foundation model against a time-series one

**Findings: [`FINDINGS.md`](FINDINGS.md)** — both competitions, the winning VN1 configuration, and why
VN2 went the other way.

The VN2 Inventory Planning Challenge (Vandeput, 2025) hands you 599 Store x Product series and eight
weeks to plan. The score is money: `0.2 EUR x units left on shelf + 1 EUR x lost sales`, over weeks
3-8. The published benchmark costs **4,334.0 EUR**; the winner scored **3,763.2 EUR**.

This repo asks one question: **given the same ordering policy, does a tabular foundation model
(TabPFN) forecast the demand that matters better than a time-series foundation model (Chronos)?**
Ordering, censoring and scoring are shared, so a difference in score is a difference in forecast.

## Status

- [x] The competition's benchmark reproduced exactly: setup **913.8 EUR**, score **4,334.0 EUR**.
- [x] Harness pinned: this repo's loop is checked against the official one, week by week.
- [x] Arms written: benchmark, seasonal moving average, naive mean, seasonal naive, TabPFN (API),
      Chronos (local).
- [x] Back-test across tuning anchors, then evaluation anchors.
- [x] The competition's own eight weeks, played once.

## Result

| where | best arm | score | TabPFN vs Chronos |
| --- | --- | ---: | --- |
| tuning anchors (7) | `seasonal_ma` @3.5 | 4,365.3 | TabPFN better by 278 (t 4.8) |
| evaluation anchors (5) | `seasonal_ma` @3.5 | 5,739.8 | TabPFN better by 554 (t 2.1) |
| the played weeks | `seasonal_ma` @3.5 | 3,915.0 | Chronos better by 15 |

TabPFN beats Chronos on the back-test and loses to it by a hair on the played weeks; neither beats
the competition's own moving average once its coverage is lowered. Fold variance (std 2,000+ EUR)
is larger than every gap being argued about. Full tables, bars and verdicts in
[`docs/method.md`](docs/method.md).

**Direction:** Chronos is the standing foundation-model arm — local, quota-free, and the only one
that beats the organisers' baseline on VN1. TabPFN is retired from the comparison except for one
pre-registered VN2 check with a ratio target, queued for when the API quota resets.

Whether Chronos can beat the moving average depends on the task: **yes on VN1** (0.5502 against
0.6597 once calibrated) and **no on VN2** (6,828 against 5,740 on the evaluation anchors, after
sweeping quantile, scale and coverage). The tests and the mechanism are in
[`docs/chronos-vs-baseline.md`](docs/chronos-vs-baseline.md).

## Quickstart

```bash
scripts/fetch_data.sh                    # competition files into data/raw/ (not redistributed here)
python3 scripts/reproduce_benchmark.py   # must print 913.8 / 4334.0
python3 -m pytest -q                     # harness pins and arm tests
python3 scripts/backtest.py --anchors tuning
```

## The arms

One policy for every arm: order up to `coverage` weeks of forecast demand, net of stock and stock in
transit (`vn2.play.order_up_to`). The competition hard-codes coverage 4; here it is tuned on the
tuning anchors and reported on the evaluation anchors.

| arm | model | runs |
| --- | --- | --- |
| `benchmark` | seasonal moving average, 13 weeks | the competition's own |
| `seasonal_ma`, `naive_mean`, `seasonal_naive` | cheap references | locally, milliseconds |
| `tabpfn_api` | TabPFN-3.5-Plus, tabular, sees lags + calendar + hierarchy | Prior Labs API, ~9 s per round |
| `chronos` | Chronos-Bolt-small, one sequence per series | locally on MPS, ~0.7 s per round |

TabPFN is called rather than run locally because the local fit is 531.7 s per round on this machine
and 4,000 rows exhausts the MPS budget; Chronos is small enough to run here. The reasons, the
censoring rule and the pre-registered bars are in [`docs/method.md`](docs/method.md).

## Keys

One secret, outside the repo:

```bash
# ~/.config/vn2-inventory/keys.env   (chmod 600)
export TABPFN_TOKEN="..."
```

Get one at <https://platform.priorlabs.ai/account/api-keys>. The free tier is 5M tokens/day and a
measured round costs the 10,000-token minimum, so a whole back-test fits comfortably.
`scripts/smoke_arms.py` validates the key and quotes the bill before a run.

## Layout

```
src/vn2/
  data.py        readers; the competition's shapes and the censoring rule
  play.py        the official loop over any window, and the shared order-up-to policy
  backtest.py    folds, warm-up states, tuning vs evaluation anchors
  features.py    the (Store x Product, anchor, horizon) table the tabular arm sees
  forecaster/    one module per arm; the two API arms load lazily
scripts/         fetch, reproduce, smoke, back-test, competition run
tests/           benchmark pins, leakage pins, arm frame tests
```

## Citation

Both competitions ask that their data be cited:

> Vandeput, N. (2025). VN2 Inventory Planning Challenge.
> https://supchains.com/vn2-inventory-competition/

> Vandeput, N. (2024). VN1 Forecasting Competition.
> https://supchains.com/vn1-forecasting-competition/

The competition data is downloaded, not redistributed: `data/` is git-ignored.

## VN1 pilot (this branch)

`vn1-pilot` adds a second, cleaner test of the same question: VN1 scores pure 13-week forecast
error over 15,053 series with a price covariate, so no ordering policy hides the forecast. It
reproduces the organisers' MA12 baseline (0.6597) as a test. On Phase 1 Chronos beats that baseline
(0.5502 vs 0.6597). On **Phase 2, the window that actually ranked**, Chronos-2 at quantile 0.56 —
chosen on Phase 1 — scores **46.02%, ahead of the winning entry's 46.37%** by 0.35 points; the
original Bolt arm scored 51.58. The margin is thin and the window is single; results, the failed
levers and the GIFT-Eval model ranking are in
[`docs/vn1-pilot.md`](docs/vn1-pilot.md); run it with `python3 scripts/vn1_pilot.py --series 500`
or `--all`.
