# W8 E1: matchup feature block (encoder v2), dev: CLOSED (no gain)

## Setup

- Features: `src/pokeai/data/matchup.py`, 66 columns per decision appended after the 48 Metamon numbers.
  - Per own move: type multiplier, STAB, P(immune ability), estimated damage, P(KO), damaging flag.
  - Opponent's revealed moves: damage to our active, plus the opponent's STAB threat.
  - Per bench slot: best damage, P(KO), incoming damage, defensive multiplier, outspeed flag.
  - Speed comparisons.
  - Inputs are public information and the Dex only (leakage audit in `.omp/colab/HANDOFF.md`).
- Arms: KK recipe (EPN 1.01M, `--kakuna-weight 1.0`), 20 min from scratch on `.cache/epn-w7c`, seed 20261019, RTX 3060 alone.
  - **ctl** = `checkpoints/epn/kk100_c20` (W7). A v1 model ignores the extra columns, so its computation is unchanged. 35,739 steps, `val_nll` 0.980.
  - **e1** = same recipe + `--config extras=66` (`checkpoints/epn/w8_e1_20`). 32,705 steps, `val_nll` 0.978.
- Dev benchmark: `configs/w8_e1_dev.yaml`, 500 battles per cell, seed 20261017, 0 errors, 0 illegal actions.
  - Evidence: `reports/w8_e1_dev.json`, `reports/w8_e1_dev_pairs.json`.

## Results

Win rate:

| Arm | vs KK | vs GymLeader |
|---|---:|---:|
| KK | 0.466 | 0.952 |
| ctl | 0.500 | 0.948 |
| e1 | 0.490 | 0.970 |

Paired differences, 95% team-pair cluster bootstrap:

| Comparison | vs KK | vs GymLeader | Pooled |
|---|---|---|---|
| e1 − ctl | −0.010 [−0.064, +0.048] | +0.022 [−0.002, +0.046] | **+0.006 [−0.024, +0.037]** |
| e1 − KK | +0.024 [−0.032, +0.080] | +0.018 [−0.002, +0.040] | +0.021 [−0.008, +0.050] |

Vs Kakuna (500 battles, seed 20261018, paired with ctl's W7 run at 0.396):

- e1: **180/500 = 0.360**, valid, 0 errors, 0 illegal actions (`runs/h2h500_kakuna_w8_e1_l{0,1}`, 2 lanes).
- e1 − ctl vs Kakuna: **−0.036 [−0.088, +0.020]**.
- Pooled over KK, GymLeader and Kakuna: **−0.008 [−0.035, +0.019]** (`reports/w8_e1_dev_pairs_kakuna.json`).

## Verdict

- **No gain; E1 is closed.** The user decided to pause this direction (feature/aux levers on the distilled EPN).
- Hand-built damage/type/speed features do not move a Kakuna-distilled 1M EPN.
  - Most likely reason: the student already gets this matchup knowledge from Kakuna's soft labels plus the species and move tokens.
  - Other possible reasons: the features are approximations (fixed spread, no items for the opponent), or the remaining gap is not about matchup arithmetic.
- **Production stays KK** (`models/pokedecision-v4-kk`, tag `w7-kk`): vs Kakuna 43.8% [39.5, 48.2].
- Plan consequence:
  - E3 (belief-averaged calc) is dropped, since it was conditional on E1 being positive.
  - Input-side levers stay bounded by the teacher, so effort moves to levers that can exceed it: search as teacher, then RL.

## Ops notes

- Kakuna h2h now supports `--lane/--lanes`; `scripts/w8_pairs.py` merges lane dirs with `+`.
- The first attempt ran 4 lanes at ~2.1 GB each.
  - Together with ~20 GB of desktop apps, this exhausted RAM: load reached 363, the user's Edge crashed, and the lanes lost their websockets.
  - Rule: check free RAM of the whole box before running lanes in parallel, and cap each with `MemoryMax` and `MemorySwapMax=0`.
