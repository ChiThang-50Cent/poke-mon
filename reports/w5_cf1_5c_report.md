# W5.CF1.5c semantic action head (pre-registered) - internal-val result: STOPPED at F-G1/F-G2

No fresh roots, no rollouts, no battles. Heldout/dev untouched; spent CF1.4 val not used for training or selection; production C frozen.

## A. ActionDescriptorEncoder
- `src/pokeai/cf/descriptors.py`: one function (`describe_battle(battle, mask)`) used by the live `NeuralPlayer` (`describe=True` -> `last_descriptor`) and the offline harness (same hook, stored in `ModelDecision.descriptor`, returned by `candidate_actions`). Input: the learner's poke-env/Metamon battle object (incl. its request) + public Dex. Dim 115 (names in `reports/w5_cf1_5c_descriptors.json`); logpi and the slot one-hot are added by the head. Normalisation fixed (log2 effectiveness clipped [-2,2]/2, bp/150, priority/3, fractions).
- Backfill: 1600/1600 train roots (400 pilot + 900 sampled + 300 greedy) by deterministic regeneration; mask/probs/candidates/turn/pair match the record for 1600/1600; NaN/inf: 0.
- World invariance QA: 240/240 bit-identical over 120 roots x 2 sampled worlds (descriptor recomputed after `apply_world`).
- Coverage (legal slots, complete features): move 5076/5076, switch 5144/5185, move+Tera 3525/3525; missing by reason: {'switch:switch_no_damaging_known_move': 41} (switch-in with no known damaging move; sw_off_eff left 0 and counted, not hidden).
- Tests `tests/test_cf_descriptors.py`: (i) bit-identical across sampled worlds; (ii) offline hook == direct live entry point on the same battle object/request; (iii) scrambling the opponent's unrevealed item/ability/tera/stats leaves the descriptor unchanged; plus all-slot-type no-missing check.

## B/C. Internal split and variants
- Internal split by team pair (seed 20261022, stratified source x opponent): train 1279 / val 321 roots; shared team pairs 0; shared individual teams 425 (val uses 434 teams, train 734) - pairs are disjoint but teams are not.
- F2 embeddings = frozen C `tokens` rows for the move / incoming-species token: move candidates 4210 (unknown 0), species 2001 (unknown 0); d_model 96.
- Fixed: MLP hidden 128, pairwise loss, AdamW 1e-3/1e-2, batch 128, 3 seeds; epoch chosen per seed by best internal-val mean per-root Spearman (same rule for all variants and the control).

| variant | best epochs | per-seed mean per-root Spearman | seed-mean Spearman | ens. val Delta tau=0 | tau=0.01 | tau=0.02 | tau=0.05 |
|---|---|---|---|---|---|---|---|
| F0 | [3, 1, 6] | [0.126, 0.138, 0.125] | 0.1298 | +0.0021 [-0.0076,+0.0127] sw 0.31 | +0.0029 [-0.0064,+0.0129] sw 0.28 | +0.0027 [-0.0064,+0.0127] sw 0.27 | -0.0016 [-0.0086,+0.0053] sw 0.15 |
| F1 | [3, 3, 3] | [0.162, 0.151, 0.165] | 0.1592 | +0.0056 [-0.0060,+0.0175] sw 0.35 | +0.0078 [-0.0031,+0.0187] sw 0.33 | +0.0068 [-0.0037,+0.0175] sw 0.32 | +0.0053 [-0.0035,+0.0144] sw 0.26 |
| F2 | [4, 3, 4] | [0.205, 0.177, 0.171] | 0.1843 | +0.0082 [-0.0072,+0.0232] sw 0.55 | +0.0072 [-0.0084,+0.0220] sw 0.54 | +0.0068 [-0.0084,+0.0216] sw 0.53 | +0.0078 [-0.0072,+0.0224] sw 0.48 |
| F1_shuffled | [3, 2, 2] | [0.178, 0.144, 0.174] | 0.1652 | +0.0084 [-0.0035,+0.0208] sw 0.37 | +0.0058 [-0.0056,+0.0181] sw 0.33 | +0.0037 [-0.0068,+0.0152] sw 0.28 | +0.0041 [-0.0025,+0.0111] sw 0.17 |
| logp-only | - | - | 0.1396 | - | - | - | - |

Paired per-root Spearman differences (roots with >=3 candidates, n=271), team-pair cluster bootstrap 95% CI:

- Selected variant: **F2** (higher internal-val Spearman than F1).
- F-G1 vs F0: +0.0546 [-0.0119, +0.1200] (n=271) -> FAIL
- F-G1 vs logpi-only: +0.0447 [-0.0273, +0.1130] (n=271) -> FAIL
- F-G2 vs shuffled-label control (F1 shuffled): +0.0191 [-0.0417, +0.0785] (n=271) -> FAIL
- Extra: F1 vs F0 +0.0295 [-0.0171, +0.0752] (n=271); F1 vs shuffled -0.0060 [-0.0482, +0.0359] (n=271); F2 vs F1 +0.0251 [-0.0319, +0.0806] (n=271).
- **F-G1: FAIL; F-G2: FAIL. Stop rule triggered: no lock-retrain, no fresh roots (F-G3/F-G4 not run).**
- Descriptive only (not a gate): the shuffled control reaches Spearman 0.165, above F1 (0.159) and logpi-only (0.140): per-seed epoch selection on a 321-root val optimises noise (best epochs 1-6); the F0->F1->F2 ordering (0.130 / 0.159 / 0.184) is not distinguishable from the control at this n. All internal-val Deltas have CIs including 0.

## D. Imprison / maybeDisabled
- Root cause: the learner's request carries `maybeDisabled` (opponent Imprison unseen by the learner); the policy picks an imprisoned but unmarked move, Showdown replies `[Unavailable choice] Can't move: <mon>'s <move> is disabled` and re-sends a request marking the move. Same mechanism as hidden trap, but it was counted as a model-illegal action and an unexpected error.
- Fix: `expected_retry_reason` in `pokeai/env/runner.py` (hidden_trap | hidden_disable; hidden_disable requires request `maybeDisabled`, a `/choose move`, the exact error text, and the move not already marked disabled). Used by `NeuralPlayer` (live + offline) and the runner baselines; memory is restored exactly as for the trap retry. The offline `step()` re-labels such errors `protocol-retry:hidden_disable:<side>` (excluded from unexpected errors, tallied in `run_variant['protocol_retries']`). Evidence now also records `maybeDisabled` and `disabled_moves`. Worlds are not filtered.
- Tests: `tests/test_runner.py::test_hidden_disable_retry_is_recognised_only_with_maybe_disabled` (4 cases incl. already-disabled move and switch orders) and `::test_hidden_disable_retry_keeps_protocol_evidence` (error + revised request -> retry with reason hidden_disable, no last_server_error); the hidden-trap evidence test was updated for the extra keys.
- CF1.4 impact: 44 unexpected lines / 20 policy-illegal events in 3 of 2000 roots (3074: 18 illegal, Clefable Calm Mind, learner side; 3087: 1, p2 Walking Wake Draco Meteor; 3240: 1, p2 Enamorus Calm Mind). All three are `val_sampled` roots (spent val) - none in internal train or internal val. Labels unchanged. Re-running those roots' belief_greedy variant with the fix (no determinism repeats): illegal 0, unexpected 0; retries `hidden_disable:p1` x24 (3074), `:p2` x1 (3087), none for 3240 (its single event was in a determinism repeat, not reproduced without repeats).

## Verification run
- `pytest tests/test_cf_descriptors.py tests/test_cf_offline.py tests/test_cf_belief.py tests/test_cf_head.py tests/test_runner.py`: 63 passed; `ruff check` clean on touched files.
- Commands: `scripts/cf1_5c_descriptors.py`, `scripts/cf1_5c_train.py` (GPU, 138 s), wrapper with 4 CPU workers.
