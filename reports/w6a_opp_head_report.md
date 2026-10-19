# W6-A — Opponent-action head: tiers 1–2 (prediction, policy regression)

**Verdict**
- **Tier 1 (prediction): pass.** The joint-trained head predicts the opponent's action better than the count baselines.
- **Tier 2 (policy): no improvement, no detected regression.** All paired CIs contain 0.
- The auxiliary task does not make the fast policy stronger at this power, so the joint checkpoint is **not** a policy candidate. C stays champion.
- **Tier 3 (teacher impact): not run yet.** Proposed design and cost are in §5.

Machine-readable sources: `reports/w6a_opp_head.json` (training, val-B metrics, baselines), `reports/w6a_dev_benchmark.json` (battles), `datasets/w6a/labels.json` (label build and QA).

## 1. Labels (`src/pokeai/data/opp_labels.py`, `scripts/w6a_build_opp_labels.py`)

The label at a learner row is the action the opponent chose at the same simultaneous decision. It is taken from the other POV of the same W4 battle and is only ever a target.

**Pairing**
- POVs are paired by the order of their non-forced states. Forced-switch subturns are side-specific and get no label.
- A battle is rejected unless every pair agrees on both active Pokémon. Rejections: **0**.
- Slots are decoded with Metamon's sorted move/switch order (`consistent_move_order` / `consistent_pokemon_order`). The stored list order is not the slot order: a first probe that used it got the labels wrong.

**Label space (learner view)**
- `move:<name>` or `switch:<team-preview name>`, with in-battle formes mapped to the public preview entry; plus a Tera flag on move labels.
- 948 classes, vocabulary from train only. Val labels not in the vocabulary: 14.
- Support at a row: every move class, plus preview species other than the active one.

**Leakage**
- Inputs are the unchanged W4 arrays, built by the parser from the learner's pre-action state.
- The QA below confirms the label is the *upcoming* action, not one already visible in the input.

| Split | Rows | Labelled | Paired / single-POV trajectories | move label = next state's `opponent_prev_move` | = current state's | switch label = next opponent active |
|---|---:|---:|---|---:|---:|---:|
| train | 2,742,268 | 1,917,710 | 85,148 / 7,577 | 84.9% | 26.5% | 92.2% |
| val | 338,968 | 234,425 | 10,356 / 1,003 | 84.7% | 26.4% | 92.1% |

- Moves are 84.9% visible one state later but only 26.5% in the input state; the 26.5% is repeats.
- The remaining ~15% of move labels are actions that never showed up publicly: the user fainted first, flinched, was fully paralysed, and so on.
- Unrevealed opponent actions (−1) get no label: 210,640 train rows.

**Splits:** W4 val is split by battle hash: half A selects, half B reports (5,419 / 5,940 trajectories). W4 test is spent and was never opened.

## 2. Arms (`scripts/train_w6a_opp.py`, seed 20261019, RTX 3060)

| Arm | What trains | Loss | Budget | GPU time |
|---|---|---|---|---:|
| frozen | opponent head only; C asserted bit-identical | class CE + Tera BCE | early stop on val-A NLL (4 epochs) | 159 s |
| joint | all, from C + the frozen head | BC + 0.5·value BCE + 0.5·(class CE + Tera BCE) | 3 epochs, fixed | 237 s |
| control | all, from C + the frozen head | BC + 0.5·value BCE (no opponent loss) | 3 epochs, fixed | 245 s |

- The control isolates the auxiliary task from the extra BC and value training.
- Without it, the joint value gain (0.669 → 0.618 BCE) would be credited to the opponent head. It comes from training the value head jointly: the control gets 0.625.

## 3. Tier 1 — val B (123,642 labelled rows, same support for every predictor)

| Predictor | Opp NLL | Top-1 | Top-5 | P(switch) Brier | Top-1 ECE | Tera Brier | Policy loss | Policy acc | Value BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C (no head) | – | – | – | – | – | – | 1.4859 | 0.4309 | 0.6686 |
| train frequency | 5.136 | 0.033 | 0.130 | 0.212 | 0.007 | 0.0357 | – | – | – |
| species-conditional frequency | 2.886 | 0.241 | 0.633 | 0.204 | 0.054 | 0.0357 | – | – | – |
| frozen | 3.426 | 0.188 | 0.497 | 0.150 | 0.038 | 0.0347 | 1.4859 | 0.4309 | 0.6686 |
| **joint** | **2.425** | **0.319** | **0.726** | **0.133** | 0.035 | **0.0336** | 1.4794 | 0.4334 | 0.6182 |
| control | 3.529 | 0.179 | 0.475 | 0.152 | 0.051 | 0.0349 | 1.4881 | 0.4333 | 0.6245 |

- **C's frozen features do not encode the opponent's move identity well.**
  - The frozen head loses to a species-conditional count table on NLL (3.43 vs 2.89).
  - It does beat the table on move-vs-switch (Brier 0.150 vs 0.204).
- **The joint head beats every baseline:**
  - −0.46 nats NLL vs the species-conditional table;
  - +7.8 pp top-1;
  - switch Brier 0.133.
- **Offline policy:** joint val-B policy loss 1.4794, vs 1.4859 for C and 1.4881 for the control. The opponent loss acts as a mild regulariser: the control overfits after its first epoch (val-A 1.4728 → 1.4847). This is an offline indicator only; no CI is computed for it.
- **Not measured:** the heuristic-mixture baseline. Grunt/GymLeader need a simulator world (their own POV and team) and cannot be run on replay rows. That comparison belongs to tier 3, on reconstructed worlds.

## 4. Tier 2 — paired dev battles (`configs/w6a_dev.yaml`, `data/teams_w5/dev`)

Setup:
- 500 battles per arm and opponent, both directions, team-pair cluster bootstrap (250 clusters).
- Greedy play, same team preview for every arm.
- Total 4,500 battles in 19 min. `illegal_actions` = 0 and battle errors = 0 in every cell.

| Opponent | C | joint | control | joint − C [95% CI] | control − C [95% CI] |
|---|---:|---:|---:|---|---|
| C | 0.516 | 0.502 | 0.542 | −0.014 [−0.066, +0.036] | +0.026 [−0.026, +0.076] |
| Grunt | 0.648 | 0.644 | 0.662 | −0.004 [−0.054, +0.046] | +0.014 [−0.034, +0.064] |
| GymLeader | 0.698 | 0.666 | 0.696 | −0.032 [−0.084, +0.018] | −0.002 [−0.056, +0.052] |

- Every CI contains 0, so all cells are inconclusive.
- The joint point estimates sit below the control's on all three opponents.
- More battles would not change the decision: promotion needs a CI > 0, and the joint arm's point estimates are ≤ 0. Battle count not scaled.

## 5. Decision and next step

- **Policy:** do not adopt joint training as a policy-improvement method.
  - The opponent task improves prediction a lot but not play.
  - This matches W4.5, where better value metrics also did not mean stronger play.
  - C stays champion; no checkpoint from this run is promoted. Checkpoints live in `checkpoints/w6a/{frozen,joint,control}` (gitignored).
- **Opponent model:** the joint-trained head is the useful artifact.

**Tier 3 (teacher impact), proposed, not run**
- **What changes:** in `cf/recon_search.py`, add an `opp_model` that samples the opponent's *root* action from the head instead of the Grunt/GymLeader mixture. The class is mapped onto legal actions in each sampled world; the heuristic continues afterwards.
- **Inputs:** the head predicts p(a_opp | learner history), so it runs on the learner's own view at the root. It needs its own hidden state, advanced alongside C (one extra forward per decision), because its trunk is the joint model's, not C's. No opponent-POV input is needed.
- **Measurement:** on the existing CF1.4/CF1.5 root set (1,600 roots), compare the mixture teacher with the head-root teacher on:
  - usable-label rate (overrides with z > 1.96);
  - label agreement across seeds (split-half);
  - prediction NLL of the actual opponent action vs the heuristic-mixture distribution (the missing tier-1 baseline);
  - seconds per usable label.
- **Prerequisite (checked read-only, 19/10/2026):** `runs/w5-cf1-4-*/roots.jsonl` does **not** store the learner's history (only `public_log_lines` as a count, the mask, C's probs and C's 384-d root features). The roots are, however, deterministically rebuildable from (seed, root id, group) with `cf1.build_root` (it asserts the replay equals pass 1; `scripts/cf1_4_affected.py` already rebuilds roots this way). So no new data is needed. What *is* needed is code: a lockstep shadow joint model on the learner side of the prefix replay (`cf/rollouts.make_players`), so the head's GRU sees the same observations as C. The class → sampled-world action mapping is also still unbuilt; a predicted move absent from a sampled set must be dropped and renormalised.
- **Cost:** about 2.2 CPU-h per teacher at 5 s/root; no battles.
- **Gate:** at least 1.2× usable labels per CPU-hour at equal or better split-half agreement. Otherwise W6-B keeps the mixture teacher.
- **Risk:** the head learned *human* opponents. Search rollouts face bots and C, so the root prior may be mis-specified for those opponents.

W6-B does not wait for this: the collector fix is in (`dd6a7ae`), and the pilot can start with the current teacher.
