# AGENTS.md

Orientation for agents working in this repo.

## What this is

Foundation models — tabular (TabPFN) and time-series (Chronos-Bolt, Chronos-2, TimesFM-3) — tested
against simple baselines on two forecasting competitions. Results and reasoning:
[`FINDINGS.md`](FINDINGS.md). Detailed records: [`docs/README.md`](docs/README.md).

## Invariants

1. **The VN2 harness is pinned.** `tests/test_play.py` requires the official benchmark to score
   913.8 setup / 4,334.0, and this repo's loop to produce the same weekly table and orders. A change
   that breaks either means the back-test is measuring something else.
2. **The VN1 scorer is pinned** to the organisers' reference scores (MA12 0.659707, naive 0.688207).
3. **No leakage.** Back-test folds end at week 156; the competition's eight weeks are played once.
   Feature builders must not read past their anchor, and `tests/` pins that.
4. **Selections are pre-registered.** Anything tuned on the tuning anchors or Phase 1 is written into
   `docs/method.md` before it is run; held-out boards are scored once.

## Conventions

- Full population for any claim; subsamples are pilots and have reversed conclusions twice here.
- Paired fold differences with standard errors, not bare means, when comparing arms.
- Numbers in documents are generated from `results/*.csv`, never retyped.
- Check scale invariance before expecting a time-series model to set a level (multiply the input by
  three; if the output triples, it cannot).
- State the leakage caveat with any foundation-model result: the checkpoints postdate the public data
  and their cards do not enumerate series, so contamination cannot be ruled out (see §6 of
  `docs/writeup.md`).

## Environment

- Data: `data/` is git-ignored; `scripts/fetch_data.sh` and `scripts/fetch_vn1_official.sh` rebuild
  it.
- Keys: `~/.config/vn2-inventory/keys.env`, never in the repo. `TABPFN_TOKEN` is only needed for the
  TabPFN arms.
- Tests: `python3 -m pytest -q` (33 tests, ~10 s).
