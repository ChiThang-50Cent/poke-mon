# W7: diagnosing EPN's gap to Metamon and choosing the next lever

Context: EPN (`models/pokedecision-v3-epn`, 1.01M parameters, tag `w7-epn`) beats P by
+19.6 pp on a fresh pool (`w7_epn_final_report.md`). Against Metamon it beats Minikazam (76.2%)
but loses to Kakuna (29.0%). The goal is a model of ≤4.7M parameters that is at least as strong
as Metamon's reference agents.

All results below come from the dev pool (`data/teams_w5/dev`) only.

## 1. Diagnosis

### EPN vs Alakazam, the family EPN imitates

- Run: `reports/w7_epn_vs_alakazam.json`, 500 battles, Alakazam with `--metamon-attention flex`.
- 0 errors, 0 illegal actions.
- **EPN wins 41.2% [37.0, 45.6] (Wilson 95%).**
- Reading: the 1M student is still a little weaker than its 50M teacher family. Even a perfect
  imitation of Alakazam would sit below Kakuna (Metamon GXE 73% vs 76%).
- **Conclusion: the BC target, not only the student's imitation error, limits strength.**

### Training curve, 60 minutes on the same 285k trajectories

`runs/epn_long60.log`, `scripts/train_epn.py`:

| Train minutes | Epochs | val_nll | val top-1 |
|---:|---:|---:|---:|
| 10 | 1.9 | 0.889 | 0.679 |
| 20 | 3.8 | 0.866 | 0.686 |
| 30 | 5.5 | 0.855 | 0.691 |
| 40 | 7.3 | 0.837 | 0.697 |
| 50 | 9.1 | 0.812 | 0.706 |
| 60 | 10.9 | **0.797** | 0.712 |

- Production EPN (20 minutes with warmdown) reached 0.808.
- `val_nll` never turns upward. Tripling compute gains only 0.011, so there is no overfitting and
  the returns are small.

## 2. Lever 1: more data from the same teacher family (negative)

- Source:
  - every remaining Alakazam-family POV in `pac-base` (`scripts/w7_select_pile.py`), one per
    battle and disjoint from the W6-P batches: 652,769 new POVs (`.cache/metamon/w7-pacbase-{c..f}`);
  - encoded to `datasets/w7_{c,d,e,f}`.
- Training cache: `.cache/epn-pile953` (`scripts/w7_build_cache.py`), with 905,082 train
  trajectories and 33.1M decisions. Val is the same fixed sample.
- `pile953_60` (60 minutes, 3.2 epochs) reached `val_nll` **0.785**, against 0.797 for `long60`
  on the old data.

Dev benchmark (`configs/w7_data_dev.yaml`, `reports/w7_data_dev.json`; 500 battles per cell,
paired, 250 team-pair clusters):

| Arm | vs EPN | Δ vs EPN arm [95% CI] | vs GymLeader | Δ [95% CI] |
|---|---:|---|---:|---|
| EPN | 0.482 | — | 0.948 | — |
| long60 | 0.510 | +0.028 [−0.024, +0.078] | 0.956 | +0.008 [−0.018, +0.034] |
| pile953_60 | 0.516 | +0.034 [−0.018, +0.086] | 0.940 | −0.008 [−0.034, +0.018] |

All results are inconclusive. Three times the data and three times the compute buy almost
nothing in play. More data from the same teacher is not the lever, which is consistent with §1:
the teacher is the ceiling.

## 3. Lever 2: soft-label distillation from Kakuna on pile states

- Idea: keep the cheap offline setting (no simulator), but replace the Alakazam-family action
  with **Kakuna's full action distribution** at every pile state. Kakuna is Metamon's strongest
  local agent (142M parameters).
- Labeler: `scripts/w7_kakuna_label.py`.
  - Kakuna runs with flex attention over each full battle, from its own
    `OpponentMoveObservationSpace` sequence of the same replay.
  - Every trajectory is checked for order, step count, data action and legal mask against our
    npz.
  - Throughput is ~8,300 decisions/s on the RTX 3060.
- Labels for `datasets/w7_c` train: 189,881 trajectories, 6.94M decisions, labelled in 9 minutes.
  - 0 NaN rows, 0 misaligned rows.
  - Kakuna's argmax agrees with the data action 66.0% of the time; its mean probability on the
    data action is 0.650.
- Loss (`train_epn.py --kakuna-weight w`): `(1-w)·CE(data action) + w·CE(Kakuna probs)`.
- Information-parity caveat:
  - Kakuna's observation carries opponent-move features that our `DefaultObservationSpace`
    encoder does not have.
  - It also has a 96-step attention context and sees the previous action and reward.
  - The student cannot always reproduce its teacher's reasoning, so soft labels (not argmax) are
    used.

### Results, 20 minutes from scratch on the same w7_c data

- Configs: `configs/w7_kakuna_dev.yaml`.
- Evidence: `reports/w7_kakuna_dev.json`, `reports/w7_kakuna_dev_pairs.json`.
- Design: 500 battles per cell, paired, 250 clusters.
- `val_nll` is measured against Alakazam-family actions, so it rises as the student follows
  Kakuna instead: bc_c 0.816, kk50 0.842, kk100 0.980. It no longer predicts strength here.

Win rate per arm:

| Arm | Kakuna weight | vs EPN | vs GymLeader |
|---|---:|---:|---:|
| EPN | — | 0.466 | 0.944 |
| bc_c | 0 | 0.510 | 0.938 |
| kk50 | 0.5 | 0.578 | **0.974** |
| kk100 | 1.0 | **0.622** | 0.950 |

Paired differences, 95% bootstrap CI:

| Comparison | vs EPN opponent | vs GymLeader | Pooled |
|---|---|---|---|
| kk50 − bc_c (teacher effect) | +0.068 [+0.012, +0.122] | +0.036 [+0.014, +0.060] | **+0.052 [+0.022, +0.082]** |
| kk100 − bc_c (teacher effect) | +0.112 [+0.058, +0.166] | +0.012 [−0.014, +0.040] | **+0.062 [+0.033, +0.092]** |
| kk100 − kk50 | +0.044 [−0.016, +0.104] | −0.024 [−0.044, −0.004] | +0.010 [−0.021, +0.040] |
| kk100 − EPN (vs production) | +0.156 [+0.106, +0.206] | +0.006 [−0.020, +0.032] | **+0.081 [+0.053, +0.109]** |

Kakuna (flex, temperature 1), 500 battles, same schedule as EPN's run
(`reports/w7_kk100_vs_kakuna.json`):

| Player | Wins vs Kakuna | Wilson 95% |
|---|---:|---|
| EPN | 0.290 | [0.252, 0.331] |
| kk100_c20 | **0.396** | [0.354, 0.440] |

Paired kk100 − EPN against Kakuna: **+0.106 [+0.050, +0.160]**.

- **Verdict: positive.** Same data, same compute, same 1.01M architecture: only the label
  source changes, and the teacher effect is +5–6 pp pooled with both CIs above 0.
- kk100 closes a third of the gap to Kakuna. kk50 vs kk100 is inconclusive; kk100 is chosen
  for the next step because the primary opponents (EPN and Kakuna) favour it.
- Next: label the whole 905k-trajectory pile and train kk100 for 60 minutes.

## 4. Scale and capacity probes on Kakuna labels (both negative)

- All 905,082 pile trajectories (33.1M decisions) are now labelled by Kakuna:
  - 0 NaN rows, 0 misaligned rows in every batch;
  - Kakuna's argmax agrees with the data action 65.9–66.1% of the time.
- `.cache/epn-pile953/train_kakuna.npy`.
- Evidence: `reports/w7_kakuna_scale_dev*.json`, `reports/w7_kakuna_wide_dev*.json`. Pooled =
  vs EPN and GymLeader, paired, 95% CI.

| Comparison | What changes | Pooled Δ |
|---|---|---|
| kk100_full60 − kk100_c20 | 3.2 times the data, 3 times the minutes | +0.000 [−0.029, +0.030] |
| kk100_wide60 (3.53M) − kk100_full60 (1.01M) | width only | +0.002 [−0.026, +0.031] |
| kk100_full60 − EPN (replicated, two runs) | | +0.074 / +0.068 |

With Kakuna labels, data, compute and width (up to 3.5M) are not the limit. Likely remaining
limits:

- **Information parity:** our encoder lacks Kakuna's opponent-move features and long attention
  context.
- **Off-policy states:** labels are only on states other agents reached. Kakuna labels on the
  student's own games (DAgger) would fix this.

## 5. Final on a fresh pool: PASS, promoted as `models/pokedecision-v4-kk`

- Candidate: `kk100_full60`. The three Kakuna arms tie on dev, so the choice is the 1.01M model
  trained on the most data.
- Bundle:
  - `configs/w7_kk_final.yaml` (seed 20262511);
  - fresh pool `data/teams_w8/heldout` (`build_teamset.py --w5-fresh-heldout --seed 20262501`,
    excluding every used and spent pool including `teams_w7`; 0 content overlap with old pools,
    checked by sha256);
  - manifest `reports/w7_kk_final_manifest.json`, `verify_bundle` returned no problems.
- Results:
  - `reports/w7_kk_final_benchmark.json`, `reports/w7_kk_final_pooled.json`;
  - 3,000 battles, 0 errors, 0 illegal actions.

| Opponent | EPN | KK | KK − EPN [95% CI] |
|---|---:|---:|---|
| EPN | 0.532 | 0.592 | **+0.060 [+0.006, +0.114]** |
| Grunt | 0.938 | 0.940 | +0.002 [−0.024, +0.030] |
| GymLeader | 0.930 | 0.948 | +0.018 [−0.012, +0.046] |
| pooled | | | **+0.027 [+0.005, +0.049]** |

Gate check:

| Condition | Result |
|---|---|
| Pooled CI > 0 | ✔ |
| vs-EPN CI > 0 | ✔ |
| No opponent CI below 0 | ✔ |
| 0 illegal actions, 0 errors | ✔ |

- **Promoted:** `models/pokedecision-v4-kk` (byte-identical to `checkpoints/epn/kk100_full60`).
  Smoke: 19/20 wins vs GymLeader, 0 errors. Tag `w7-kk`.
- The fresh-pool effect vs EPN (+6.0) is about half the dev effect (+12.4). The dev pool has
  been reused many times; trust the fresh-pool number.
- Grunt and GymLeader are near saturation (~94%), so the pooled number mostly reflects the
  vs-EPN cell.

### Against Metamon (dev pool, not part of the gate)

Production KK vs Kakuna (flex, temperature 1): 500 battles, the same schedule as EPN's run,
0 errors, 0 illegal actions (`reports/w7_v4_vs_kakuna.json`).

| Player | Wins vs Kakuna | Wilson 95% |
|---|---:|---|
| EPN | 0.290 | [0.252, 0.331] |
| kk100_c20 | 0.396 | [0.354, 0.440] |
| **KK (`v4-kk`)** | **0.438** | [0.395, 0.482] |

Paired KK − EPN: **+0.148 [+0.094, +0.202]**.

## 6. What this does and does not show

- KK is pure distillation. Its loss is minimised by copying Kakuna, so its expected ceiling is
  about 50% against Kakuna.
  - It can edge past that slightly, through greedy play on the teacher's distribution
    (Kakuna samples at temperature 1). That effect is not measured.
  - It cannot clearly beat its teacher.
- More data, more minutes and 3.5× width did not help.
  - Unmeasured candidates: information parity (Kakuna's richer observation and context) and
    off-policy states.
  - A cheap discriminating test, not run: label the val split with Kakuna, then compare the KL
    of the 1.01M and 3.53M students to it.
- Exceeding Kakuna needs a signal beyond imitation:
  - RL from game outcomes, starting from KK with a KL anchor to KK, against a mixed opponent pool
    (not always Kakuna);
  - search as the teacher (expert iteration), distilled back into the policy.
