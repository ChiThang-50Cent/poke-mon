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
