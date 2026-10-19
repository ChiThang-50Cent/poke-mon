# W6-P: fine-tune C on Metamon self-play (Alakazam-family POVs) — dev result

**Verdict: strongest dev signal of the project so far; not yet promoted.** On the dev pool, C fine-tuned on
Metamon self-play beats C by about +10 pp against every opponent, while the identical recipe on W4 human
data (the W6-A `control` arm) is flat. The one final evaluation on a fresh pool has not been run.

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

## Caveats

- Dev only. Selection on dev is trivial here (one candidate), but the fresh-pool evaluation is still
  required before promotion.
- The pile's teams come from Metamon team sets, not from our pools; the gain could partly be
  team-distribution effects. The fresh pool tests this.
- Human-imitation loss rises (1.48 → 1.61). That is not a gate, but the bot will play less human-like.

## Next

1. One final paired evaluation of `checkpoints/w6p/pile` vs C, Grunt, GymLeader on a fresh pool
   (`build_teamset.py --w5-fresh-heldout --exclude-dirs` over every used and spent pool), with the §5 gate
   from `w6_0_distill_feasibility.md`.
2. If it passes: promote. Scaling the pile (more POVs, other strong families, epochs) is the next lever,
   at roughly 1 GB of download per 100k POVs via the same Colab filter.
