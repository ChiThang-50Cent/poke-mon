# W6-P: fine-tune C on Metamon self-play (Alakazam-family POVs)

**Verdict: promoted.** C fine-tuned on Metamon self-play beats C on dev (+10 pp vs every opponent; the
same recipe on W4 human data is flat) and passes the one-shot final gate on a fresh pool (pooled
+7.1 pp [+4.0, +10.1]). New champion: `models/pokedecision-v2-pile` (same 1.30M GRU, same speed).

## Data

- Source: `jakegrigsby/metamon-parsed-pile`, `pac-base/gen9ou.tar.lz4`, 35.8 GB, about 4.5M POVs.
- Filtering happened on Colab; only the selected subset was downloaded (664 MB zip). The Colab notebook
  streamed the archive and kept 100k POVs sampled uniformly (seed 20261019) from the 952,816 POVs whose
  player belongs to the Alakazam family (`alakazam`, `alakazamdefault`, `pac-mm-alakazam`; index win rates
  0.585 / 0.614 / 0.767). Wins and losses are both kept.
- `pac-exploratory` was rejected: it was collected at raised sampling temperature (exploration), so its
  actions are noisier BC targets.
- Format: Metamon parsed-replay `{states, actions}` (UniversalState), read by the unchanged W4 path
  (`ParsedReplayDataset` → `encode_trajectory` → `ObservationEncoder`; `encoder.json` byte-identical to W4).
- `scripts/w6p_build_pile.py` → `datasets/w6p/{train,val}.npz` (+ `outcomes`, `family`), 95/5 split by
  battle-id hash; one POV per battle in this subset.

| Split | Trajectories | Decisions | Valid | Unknown tokens | Win rate |
|---|---:|---:|---:|---:|---:|
| train | 95,039 | 3,453,779 | 3,438,707 | 0 | 0.599 |
| val | 4,961 | 178,322 | 177,579 | 0 | 0.601 |

- Excluded decisions: 0.4% `chosen_not_in_mask` (W4: 0); they carry no loss.
- Training drops trajectories longer than 500 decisions (W4's maximum; the pile has stall battles up to
  ~8k decisions that do not fit a padded batch on 12 GB).

## Training (`scripts/train_w6p_pile.py`, `reports/w6p_pile_train.json`)

Only the training data differs from W6-A `control`: init C, BC + 0.5·value BCE, AdamW lr 1e-4 wd 1e-4,
batch 16 full trajectories, clip 1.0, fixed 3 epochs, last epoch kept. Wall time 4.8 min on the 3060.

| Model | Pile val loss / acc | W4 val-A loss / acc | W4 val-A value BCE |
|---|---|---|---|
| C | 1.408 / 0.477 | 1.482 / 0.433 | 0.664 |
| pile (epoch 3) | 1.190 / 0.560 | 1.611 / 0.424 | 0.630 |

The student moves toward the self-play policy and away from human imitation (expected; human imitation is
not the objective).

## Dev benchmark (`configs/w6p_dev.yaml`, `reports/w6p_dev_benchmark.json`)

Paired on `teams_w5/dev`, 500 battles per cell (250 team-pair clusters, both directions), seed 20261017,
greedy play, same preview model. 4,500 battles in 19 min; every cell valid (0 illegal actions, 0 errors).

| Opponent | C win rate | pile − C [95% CI] | control − C [95% CI] |
|---|---:|---|---|
| C | 0.522 | **+0.102 [+0.050, +0.156]** | −0.002 [−0.054, +0.050] |
| Grunt | 0.646 | **+0.096 [+0.044, +0.148]** | −0.012 [−0.062, +0.040] |
| GymLeader | 0.666 | **+0.110 [+0.060, +0.160]** | +0.004 [−0.048, +0.054] |

- The control arm rules out "more fine-tuning" as the cause: the same recipe on human data does nothing.
- All three CIs exclude 0, including the primary opponent C.

## Final (one-shot, fresh pool)

- Pool `data/teams_w6p/heldout`: `build_teamset.py --w5-fresh-heldout --seed 20262301 --exclude-dirs`
  over `teams/gen9ou`, `teams_w5/{collect,dev,heldout}`, `teams_w5_bs14/heldout`, `teams_w5_bs15/heldout`;
  200 server-validated teams.
- Frozen bundle `reports/w6p_final_manifest.json` (code, both checkpoints, config, team files, Showdown,
  versions; `bs14_manifest.py` now also handles policy-only configs) verified before the claim.
  A first launch was refused before any claim or battle: the systemd unit resolved a different Node
  (v22.22.1) from the frozen one (v22.20.0). Relaunched with the frozen PATH.
- `configs/w6p_final.yaml`, seed 20262311, 500 battles per cell (250 pair clusters), C and pile arms.
  3,000 battles, all cells valid, 0 illegal actions, 0 errors.
- Pooled CI: cluster bootstrap within opponent strata (the BS1.2 `stratified` method), 10,000
  resamples (`reports/w6p_final_pooled.json`).

| Opponent | C win rate | pile win rate | pile − C [95% CI] |
|---|---:|---:|---|
| C | 0.526 | 0.622 | **+0.096 [+0.040, +0.152]** |
| Grunt | 0.620 | 0.678 | **+0.058 [+0.006, +0.110]** |
| GymLeader | 0.658 | 0.716 | **+0.058 [+0.002, +0.112]** |
| pooled | | | **+0.071 [+0.040, +0.101]** |

Gate (`w6_0_distill_feasibility.md` §5): pooled CI > 0 ✔, vs-C CI > 0 ✔, no opponent CI < 0 ✔,
illegal actions 0 ✔, errors 0 ✔ → **PASS**. The checkpoint was copied unchanged (sha256 equal to the
manifest) to `models/pokedecision-v2-pile`, with these results in its `metrics.json`.

## Caveats

- Against the heuristics the gain shrinks from ~+10 pp (dev) to ~+6 pp (fresh), and those lower CI bounds
  are near 0; the vs-C gain holds (+9.6).
- Human-imitation loss rises (1.48 → 1.61): the bot plays less like the W4 humans.
- `pokedecision-v1-value` stays the init/reference for older scripts; they are not repointed.

## Scale: 300k POVs (dev, not promoted)

- Second, disjoint batch: 199,998 Alakazam-family POVs, seed 20261020, no battle shared with the first
  100k (`.cache/metamon/w6-pacbase-b/`, zip sha256 `d60805c3…c24f`, tar sha256 `eb943e5b…e701`), encoded
  to `datasets/w6p_b/` (190,103 train / 9,895 val trajectories, unknown rate 0, 99.6% valid).
- `pile300`: same recipe, init C, train = both batches (285,142 trajectories), 3 epochs, 13.5 min.
  Pile val (first batch) loss/acc 1.127 / 0.584 vs 1.190 / 0.560 for P; W4 val-A loss 1.645 (P 1.611).
- Dev, `configs/w6p_scale_dev.yaml`, reference P, 500 battles per cell, 0 errors / 0 illegal:

| Opponent | P win rate | pile300 − P [95% CI] |
|---|---:|---|
| P | 0.518 | +0.022 [−0.032, +0.076] |
| Grunt | 0.722 | +0.022 [−0.024, +0.070] |
| GymLeader | 0.788 | +0.018 [−0.024, +0.062] |
| pooled | | +0.021 [−0.007, +0.049] |

All inconclusive. Tripling the same-source data gives at most a small gain; P stays champion and
`checkpoints/w6p/pile300` is kept for reference only. More battles would only resolve a ~2 pp effect,
which would not change the next step: the cheap lever on this source looks exhausted, so the next gain
must come from a different signal (stronger teacher such as Kakuna, or on-policy data).
