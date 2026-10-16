# W5 CF0.4 report: Q-signal validation (pilot, exploratory)

Rollouts: 4960 over 90 roots (CF0.3, seed 20261019).
Labels use the true simulator state (privileged); policy inputs are POV-only. One pilot.

## Q1. Is branching correct?  **YES**
0 unexpected error lines, 0 illegal actions
over every rollout; CF0.1/0.2 gates stand.

## Q2. Is there a Q signal across candidates?  **YES**
- Global within-k permutation test: T_global = 1.0680 (null mean 0.4551), p = 0.0004998
- Roots with p < 0.05: 13/90 = 0.144; binomial p vs 0.05 = 0.0005376
- Mean best-minus-worst gap 0.187 (SE across roots 0.019); upward-biased by selection (see JSON for opponent/phase tables).

## Q3. Is the ranking stable?  **YES**
- Split-half Spearman (87 roots with >=3 candidates; undefined as 0): mean 0.177, 95% bootstrap CI [0.071, 0.288]; defined-only 0.286 (n=54)
- Permutation null mean 0.000, 95% interval [-0.105, 0.103], p = 0.001998
- Top-1 agreement 0.500 vs chance 0.299; 2-candidate sign agreement 1.0

## Gate: **PASS**

## Descriptive (not gated)
- C's argmax is best Q-hat candidate: 0.349
- Mean Spearman(probs, Q-hat): 0.2530199327976955
- Non-argmax beats argmax (paired CI excl. 0): 7 roots of 220 uncorrected comparisons; by phase {'6-15': 3, '0-5': 1, '16-30': 3}, by opponent {'Grunt': 2, 'GymLeader': 2, 'C': 3}
- Probe V vs prob-weighted Q-hat: mean V 0.49588093194696636, mean Q-hat 0.44321489430808314, MAE 0.2762325469392741, corr 0.48969795689719947
- Tera candidate effect (Q-hat tera minus others): {'n': 45, 'mean': 0.018750000000000003, 'se': 0.02299544540249784}
- Tied rollouts: 0
