# W5.CF1.5d one-shot 5-fold CV of the semantic head: STOPPED (CV-G1 FAIL)

Offline, 1,600 train roots only; no battles, no fresh roots; heldout/dev/spent val untouched. Exploratory (epoch count chosen after CF1.5c). CIs reflect this development design (overlapping training folds, teams recurring across pairs), not a fresh-set confirmation.

## Preflight (F2 embedding mapping)
461 distinct candidate names (moves + incoming species) -> 461 distinct token ids; 0 collisions, 0 multi-token names, 0 unknown. Mapping left unchanged.

## Design
- Folds: 5, grouped by team pair, stratified by source x opponent, seed 20261030; sizes [320, 321, 319, 319, 321] roots ([319, 321, 319, 319, 321] pairs); shared pairs 0; fraction of each eval fold's teams also in its training folds [0.979, 0.991, 0.975, 0.977, 0.975] (teams recur across pairs).
- F2 primary (locked), baselines F0, F1, logpi_C; matched control F2_shuffled (F2 inputs/recipe, non-greedy labels permuted within root, fresh permutation per seed and fold; tests action-ranking among alternatives, not all outcome signal). MLP 128, pairwise loss, AdamW 1e-3/1e-2, batch 128, exactly 3 epochs, no early stopping, seeds 1-3, all normalisation fitted on training folds. One OOF prediction per root per variant; ensemble = mean of 3 seeds.
- Primary metric: mean per-root Spearman (ensemble dQ vs rollout dQ), roots with >=3 candidates (1540/1600), undefined = 0.

## Results (primary)
| variant | mean per-root Spearman | per-seed means | defined-only mean | ens. Delta_head tau=0 [CI] | switch rate |
|---|---|---|---|---|---|
| F0 | 0.1114 | [0.112, 0.11, 0.113] | 0.1342 | -0.0011 [-0.0042, +0.0020] | 0.165 |
| F1 | 0.1198 | [0.124, 0.128, 0.123] | 0.1444 | -0.0026 [-0.0075, +0.0023] | 0.335 |
| F2 | 0.0985 | [0.104, 0.098, 0.091] | 0.1186 | -0.0029 [-0.0089, +0.0030] | 0.492 |
| F2_shuffled | 0.0452 | [0.028, 0.04, 0.051] | 0.0545 | -0.0139 [-0.0204, -0.0073] | 0.538 |
| logpi_C | 0.1077 | - | - | - | - |

## Gates (paired per-root, team-pair cluster bootstrap, 5,000 resamples)
- CV-G1 F2 - F0: -0.0129 [-0.0399, +0.0131] (n=1540) -> FAIL
- CV-G1 F2 - logpi_C: -0.0093 [-0.0369, +0.0173] (n=1540) -> FAIL
- CV-G2 F2 - F2_shuffled: +0.0533 [+0.0266, +0.0797] (n=1540) -> PASS
- Reliability: OOF complete (5 folds x 3 seeds x 4 variants), outputs finite; descriptor tests pass (63 tests incl. test_cf_descriptors).
- **CV-G1 FAIL (CV-G2 pass) -> stop: no lock, no fresh roots, no further tuning.** F2 is not better than F0 or the logpi ordering (point estimates negative); the only positive contrast is against a control that has destroyed the among-alternative ranking, i.e. the heads carry some ranking signal, but not beyond logpi.

## Diagnostics (no gate)
- F1 - F0: +0.0084 [-0.0138, +0.0301] (n=1540); F2 - F1: -0.0213 [-0.0441, +0.0016] (n=1540).
- 2-candidate roots (29 with non-zero rollout dQ), sign accuracy: F0 0.414, F1 0.517, F2 0.448, F2_shuffled 0.448.
- By group (roots>=3; mean Spearman F0/F1/F2/shuffled/logpi; F2-F0 diff [CI]):
  - source=pilot (n=380): 0.139/0.156/0.098/0.028/0.124; F2-F0 -0.041 [-0.091, +0.010]; F2-logpi -0.026; F2-shuffled +0.070
  - source=sampled (n=867): 0.132/0.121/0.121/0.067/0.132; F2-F0 -0.012 [-0.048, +0.024]; F2-logpi -0.011; F2-shuffled +0.054
  - source=greedy (n=293): 0.014/0.069/0.033/0.003/0.015; F2-F0 +0.019 [-0.044, +0.083]; F2-logpi +0.018; F2-shuffled +0.030
  - opponent=C (n=520): 0.057/0.061/0.069/0.026/0.046; F2-F0 +0.013 [-0.035, +0.060]; F2-logpi +0.023; F2-shuffled +0.043
  - opponent=Grunt (n=505): 0.151/0.157/0.112/0.076/0.156; F2-F0 -0.038 [-0.083, +0.007]; F2-logpi -0.043; F2-shuffled +0.036
  - opponent=GymLeader (n=515): 0.128/0.142/0.114/0.034/0.123; F2-F0 -0.014 [-0.057, +0.029]; F2-logpi -0.009; F2-shuffled +0.080
  - phase=0-5 (n=344): 0.071/0.096/0.070/0.000/0.089; F2-F0 -0.001 [-0.059, +0.054]; F2-logpi -0.020; F2-shuffled +0.070
  - phase=6-15 (n=654): 0.139/0.144/0.116/0.070/0.127; F2-F0 -0.023 [-0.065, +0.020]; F2-logpi -0.012; F2-shuffled +0.046
  - phase=16-30 (n=413): 0.105/0.101/0.107/0.026/0.101; F2-F0 +0.002 [-0.046, +0.051]; F2-logpi +0.006; F2-shuffled +0.081
  - phase=31+ (n=129): 0.102/0.125/0.060/0.104/0.079; F2-F0 -0.042 [-0.125, +0.036]; F2-logpi -0.019; F2-shuffled -0.044
- Per fold (F0/F1/F2/shuffled/logpi):
  - fold0 (n=306): 0.102/0.079/0.094/0.015/0.107
  - fold1 (n=309): 0.161/0.176/0.154/0.078/0.126
  - fold2 (n=309): 0.081/0.108/0.076/0.059/0.107
  - fold3 (n=309): 0.058/0.091/0.076/0.029/0.054
  - fold4 (n=307): 0.155/0.143/0.092/0.046/0.145

## Deviations
- `fit_sem` gained a `perm_seed` argument so the control permutation is fresh per seed and fold (1000*seed+fold+77). Folds use a deterministic greedy stratified pair assignment. Wall time ~145 s on GPU (wrapper, 4 CPU workers).
