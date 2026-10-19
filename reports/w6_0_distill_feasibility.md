# W6.0 — Search-to-policy distillation: feasibility (no training, no battles)

Inputs only: BS1.2 search-arm logs (`runs/w5-bs1-2-b*/lane*/benchmark.json` → 300 `battles.jsonl`
records, 11,229 decision rows), CF1.3/CF1.4/CF1.5/CF1.5d reports, and the code paths `agents/live_search.py`,
`agents/neural_player.py`, `cf/recon_search.py`, `env/collector.py` and `env/runner.py`. Heldout and
sealed data were not touched.

**Verdict: feasible, but BS1.2 is analysis-only.** Its logs have no observations, so a new search
collection on `teams_w5/collect` is required. One small collector bug must be fixed first (§3). Label
noise and label cost are the main risks (§2, §5).

## 1. What the teacher label is

- `decide()` (`recon_search.py:76`) considers ≤4 candidates: a_C plus up to 3 slots sampled from C's
  probabilities (`choose_candidates`).
- Each candidate is scored by paired rollouts across sampled worlds × seeds:
  - the learner side plays **C greedy**;
  - the opponent side plays a **Grunt/GymLeader mixture** (`opp_model=mixture`).
- Search overrides a_C only when the best paired delta is > 0, `delta/se > SNR_MIN = 1.0`, and there are
  ≥ `MIN_PAIRS = 4` pairs.

So a label is a one-step policy-improvement estimate of Q^C(s,·) **against heuristic opponents**. It is
not an oracle, and it is only defined over C's top candidates.

## 2. BS1.2 label audit (300 search battles, 11,229 rows)

| Category | Rows |
|---|---:|
| override | 1,782 |
| fallback: no positive delta | 5,660 |
| fallback: few pairs (< 4) | 2,246 |
| fallback: SNR ≤ 1 | 1,380 |
| single legal / not represented / error | 161 |
| of which `search_fallback_unsupported` (overlaps few_pairs) | 2,178 |

Confidence of the 1,782 overrides:

| Threshold | Count |
|---|---:|
| z > 1 (the live rule) | 1,782 |
| z > 1.645 | 1,001 |
| z > 1.96 | 713 |
| z > 2.576 | 370 |

- Deltas of the overrides: median 0.20, p10 0.07, p90 0.42.
- n_pairs of the overrides: p10 10, p50 20, p90 51.
- **C confirmed:** in only 350 of the 8,822 decisions with stats were all alternatives worse than a_C at
  z < −1.96.

**Multiple-comparison check.** Each decision tests up to 3 alternatives. If none of them were truly
better than a_C, about 632 decisions would still reach max z > 1.96 by chance (independence
approximation; the comparisons share a_C, so this is rough). The observed count is 713. A per-decision
z above 1.96 therefore does **not** isolate true improvements. This agrees with CF1.3: belief_greedy
labels had split-half Spearman 0.206 and only 15% of dQ CIs excluded 0. Individual labels are weak.
Whatever signal there is lies in the aggregate (BS1.2 +0.267 over about 37 decisions per battle).

Overrides by type (all / z > 1.96):

| Change | All | z > 1.96 |
|---|---:|---:|
| move→move | 758 | 263 |
| move→switch | 392 | 156 |
| switch→switch | 307 | 161 |
| move→tera | 182 | 62 |
| switch→move | 105 | 53 |
| switch→tera | 33 | 15 |
| tera→* | 5 | 3 |

- By opponent (all overrides): C 854, Grunt 484, GymLeader 444.
- Per battle: 5.9 overrides, of which 2.4 have z > 1.96.
- Top unsupported mechanics (no label available): pivot switch request 546, Court Change 434,
  Wish 424+121+111, Future Sight ≈ 380.

**Opponent-model confound.** The rollout opponent is a heuristic mixture.
- Against Grunt/GymLeader the search effectively knows the opponent's policy class (+0.32 / +0.30).
- Against C it does not; there the gain was +0.18, and CF1.3's root-level G4 was inconclusive
  (+0.008 [−0.008, +0.025]).

Distilled labels may therefore teach exploitation of the heuristic opponents. **The primary gate must be
student vs C**, not vs Grunt/GymLeader.

## 3. Can logs become training samples?

| Requirement | BS1.2 logs | New collection (runner + `save_trajectories` + search) |
|---|---|---|
| Player-view observation per step | ✗ (no states, no protocol) | ✓ `TrajectoryCollector.states` (UniversalState) |
| Full trajectory for GRU | ✗ | ✓ native lz4 per POV + `trajectory_meta/` |
| Legal mask | ✗ | ✓ `behavior.mask` |
| C probabilities / a_C | a_C only | ✓ `behavior.probabilities` |
| Teacher target + stats (`candidates`, `stats[delta, se]`, `n_pairs`, `reason`) | ✓ | ✓ in `battles.jsonl.search`; one row per `choose_move`, aligned by index with the collector steps; `turn` is available to assert the join |
| Executed action | ✓ `action` | **✗ bug, see below** |

**Blocker (code, small).** With both `search` and `save_trajectories` enabled:
- `neural_player.py:443` records `self._behavior[tag]` from `select_action`;
- `collector.record` then overwrites the recorded action with `behavior["chosen_index"]`, which is
  **C's argmax**, not the overridden action that was sent (`collector.py:87-88`);
- result: on 16% of searched steps the trajectory action would differ from the action actually played.

This combination has never been run (`collect_w5_selfplay.py` runs without search). Fix before
collecting:
- record the executed action;
- keep C's probabilities and a_C as behavior fields;
- attach the search row (or its index) to the step;
- add a regression test with a stub search that overrides.

Hidden state is unaffected: C's GRU is advanced before the search runs, as in live play.

Conclusion: **BS1.2 cannot produce training samples.** Its role is to size and design the experiment.

## 4. Recipes (both start from C; both keep BC)

Notation:
- L_BC: cross-entropy on the W4 human train split, mixed into every batch;
- KL_C: KL(C ‖ student) on every collected step, with C frozen;
- ℓ: masked logits.

Training runs over full collected trajectories with one uninterrupted hidden state, as in BC.

- **R1 — hard labels, filtered.**
  - CE toward the teacher action at overrides with z ≥ 1.96 (about 2.4 per battle).
  - CE toward a_C at confirmed-C decisions (all alternatives at z ≤ −1.96).
  - Every other step: KL_C only.
  - Loss = L_BC + λ_T·CE_teacher + λ_C·KL_C.
- **R2 — pairwise, all stats.**
  - For every decision with ≥ 4 pairs and every alternative a:
    `w·softplus(−sign(δ)·(ℓ_a − ℓ_{a_C}))` with `w = min(|δ/se|, 3)/3` and `|δ/se| ≥ 1`.
  - This uses about 8.8k decisions per 300 battles instead of about 0.8k, and it also learns from
    "C was right" (5,660 no-positive-delta rows).
  - Loss = L_BC + λ_P·pairwise + λ_C·KL_C.

No soft-target-as-probability recipe: the deltas are win-rate differences under a heuristic opponent,
not a policy distribution.

Fixed hyperparameters, no grid:
- λ_C = 1;
- λ_T = λ_P chosen once so that the teacher term is about 0.2× L_BC at initialisation;
- 3 epochs, AdamW lr 3e-5 (one tenth of BC);
- 3 seeds per recipe.

Every training run is GPU-minutes; the cost is in collection.

## 5. Budget, evaluation, stop rule, gate

**Collection.**
- Throughput measured in BS1.2: 300 search battles in about 6.1 h at CPUQuota 1000% (2 lanes × 4
  workers). That is about 1,840 searched decisions/h, about 290 overrides/h, and about 115 overrides/h
  at z > 1.96.
- Pilot: **12 h on `teams_w5/collect`** (1,001 teams), opponents C / Grunt / GymLeader in equal
  shares, same RC4 search settings.
- Expected yield: about 22k searched decisions, about 3.5k overrides, about 1.4k overrides at z > 1.96.
- Unit `w6-distill-collect`, `Restart=no`.

**Dev evaluation (cheap).**
- Student and C are both fast (0.3 s/battle), so N is not a cost issue.
- Paired benchmark on `teams_w5/dev`: student vs C, both directions, same team pairs, team-pair cluster
  bootstrap.
- Opponents and N:
  - **C (primary)**: 1,000 battles per arm;
  - Grunt and GymLeader (secondary): 500 battles per arm each.
- 2 recipes × 3 seeds. Select one (recipe, seed) by the primary Δ.

**Stop rule (pilot).**
- If neither recipe's seed mean is positive on the primary vs-C Δ, or the selected candidate's CI
  contains 0: **stop**. Do not collect more data, do not scale.
- Then diagnose, all offline:
  - teacher agreement on the held-out collect split;
  - Δ restricted to states with confident labels;
  - KL drift.
- Only if the diagnosis shows the student underfits the labels (agreement far below what the data
  allows) is one more 12 h collection allowed.

**Promotion gate (once).**
- One final paired evaluation of the single selected student vs C on a fresh pool:
  `build_teamset.py --w5-fresh-heldout --exclude-dirs` over every used and spent pool.
- Opponents: C, Grunt, GymLeader.
- **Promote only if** the pooled CI is > 0, the vs-C CI is > 0, no opponent has a CI entirely < 0, and
  `illegal_actions` = 0 and battle errors = 0.
- Report student-fast and student+search separately; the latter is optional and not part of the gate.

## 6. Prior evidence to keep in mind

- CF1.5 and CF1.5d (heads on frozen C features, 1,600 roots, same family of labels) failed because
  the ranking did not beat log π_C (CV-G1: −0.009 [−0.037, +0.017]).
- Distillation differs on three counts, which is why it is not a reopening:
  - it updates the whole policy;
  - it uses on-policy trajectories;
  - it trains on about 10× more labelled decisions per pilot.
- If R1 and R2 both fail on the pilot, the likely cause is label noise. That is consistent with CF1, and
  the next lever is teacher quality (more pairs per decision, or a learned opponent model in place of
  the heuristic mixture), not student capacity.
