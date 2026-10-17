# W4.5 final benchmark and ablation

All choices were frozen before evaluation. C (Probe) remains the production candidate; D is an unselected ablation, rejected by the W4.4 gate (ΔL_policy 0.0155 > 0.01). No W4.5 result changes that pick.

## Policy finding

Primary question: does joint BC + 0.5·value training change battle strength despite lower imitation accuracy?

| Comparison | Random | Grunt | GymLeader | Pooled |
|---|---|---|---|---|
| D-C | +0.80 pp [-0.40, +2.00] — inconclusive | -5.00 pp [-10.40, +0.40] — inconclusive | -3.00 pp [-8.20, +2.40] — inconclusive | -2.40 pp [-5.00, +0.13] — inconclusive |
| B-A | +3.40 pp [+1.20, +5.60] — positive | +35.40 pp [+30.00, +40.80] — positive | +34.80 pp [+29.00, +40.40] — positive | +24.53 pp [+21.87, +27.20] — positive |
| C-B | +0.40 pp [-1.20, +2.00] — inconclusive | +5.60 pp [+0.40, +10.60] — positive | +4.20 pp [-1.00, +9.40] — inconclusive | +3.40 pp [+0.87, +5.80] — positive |
| C-A | +3.80 pp [+1.60, +6.20] — positive | +41.00 pp [+35.40, +46.20] — positive | +39.00 pp [+33.20, +44.40] — positive | +27.93 pp [+25.27, +30.53] — positive |

Paired percentile bootstrap: 5,000 seeded resamples clustered by team pair; both directions remain together. Pooled resampling is stratified by opponent. Ties count as non-wins. A CI containing zero is inconclusive, not grounds to pick a higher observed win rate. Multiple comparisons are reported without multiplicity correction; D−C is primary.

| Arm | Opponent | W/L/T | Win rate (Wilson 95% CI) | Errors / illegal / retries |
|---|---|---|---|---|
| A | RandomBaseline | 473/27/0 | 94.6% [92.3, 96.3] | 0 / 0 / 1 |
| B | RandomBaseline | 490/10/0 | 98.0% [96.4, 98.9] | 0 / 0 / 1 |
| C | RandomBaseline | 492/8/0 | 98.4% [96.9, 99.2] | 0 / 0 / 0 |
| D | RandomBaseline | 496/4/0 | 99.2% [98.0, 99.7] | 0 / 0 / 0 |
| A | Grunt | 122/378/0 | 24.4% [20.8, 28.4] | 0 / 0 / 0 |
| B | Grunt | 299/201/0 | 59.8% [55.4, 64.0] | 0 / 0 / 0 |
| C | Grunt | 327/173/0 | 65.4% [61.1, 69.4] | 0 / 0 / 0 |
| D | Grunt | 302/198/0 | 60.4% [56.0, 64.6] | 0 / 0 / 0 |
| A | GymLeader | 134/366/0 | 26.8% [23.1, 30.8] | 0 / 0 / 0 |
| B | GymLeader | 308/192/0 | 61.6% [57.3, 65.8] | 0 / 0 / 0 |
| C | GymLeader | 329/171/0 | 65.8% [61.5, 69.8] | 0 / 0 / 0 |
| D | GymLeader | 314/186/0 | 62.8% [58.5, 66.9] | 0 / 0 / 0 |

A=v0/random preview; B=v1/random preview; C=frozen v1+probe/pretrained argmax preview; D=joint/pretrained argmax preview. Every action uses greedy policy argmax; value never guides decisions. All arms use identical paired schedules, 250 team pairs/opponent played in both directions, alternating challenger side. Showdown and heuristic RNG are not seedable. The validated team pool was never used for training or selection; W4.3 only measured on it.

C and D differ in objective, with the same scratch initialization recipe, seed 20261016 and lr 5e-4. This is only one training seed: causal attribution needs multiple seeds.

## Value finding

Full decision-weighted and battle-equally-weighted metrics, battle-bootstrap CIs, reliability tables, phase (decision indices 0–5, 6–15, 16–30, 31+), first/last decision and opponent splits are in `w4_5_value_calibration.json`. Prediction spread accompanies ECE. Low ECE is not evidence of better probabilities when predictions sit near 0.5. Live opponents are bots, unlike the human training distribution.

| Arm | Weighting | Brier | AUC | ECE | Prediction std | Mean abs(p−0.5) |
|---|---|---|---|---|---|---|
| C | decision_weighted | 0.19328 | 0.73868 | 0.18718 | 0.14728 | 0.13385 |
| C | trajectory_equally_weighted | 0.19054 | 0.75812 | 0.19582 | 0.14722 | 0.13293 |
| D | decision_weighted | 0.14231 | 0.82878 | 0.08493 | 0.23473 | 0.24676 |
| D | trajectory_equally_weighted | 0.14083 | 0.83987 | 0.08588 | 0.23564 | 0.24219 |
- C−D live Brier (decision_weighted): +0.050967, 95% battle-bootstrap CI [+0.042899, +0.059401]. Negative favors C.
- C−D live Brier (trajectory_equally_weighted): +0.049702, 95% battle-bootstrap CI [+0.043778, +0.055965]. Negative favors C.

Both heads rank live outcomes above chance, but Probe is poorly calibrated on these bot battles. Joint has lower live Brier and ECE and stronger discrimination; its better value prediction does not establish stronger action selection. Their live Brier comparison describes each arm's own induced battle distribution, not predictions on identical states. Compare Brier with 0.25 (constant 0.5) and the empirical-base-rate references in the JSON; discrimination does not imply calibration.

## Architecture decision

Keep Probe as production. Joint is not significantly stronger in the pooled primary comparison. Focus W5 on action-value learning or self-play, not promotion of Joint.

## Test-set results

This is the first and only use of both test splits in this final frozen evaluation; neither split was used to select these models. Test outcome labels are separately gitignored; the default train/val-only guard remains intact.

| Split | Policy | Loss | Accuracy | Non-forced accuracy |
|---|---|---|---|---|
| w4_test | v0 | 1.737159 | 31.50% | 30.51% |
| w4_test | v1 | 1.489742 | 42.97% | 42.14% |
| w4_test | joint | 1.502895 | 42.11% | 41.27% |
| w3_test | v0 | 1.695338 | 33.49% | 32.64% |
| w3_test | v1 | 1.528074 | 42.02% | 41.29% |
| w3_test | joint | 1.528438 | 41.20% | 40.45% |

| W4 test value | BCE | Brier | AUC | ECE |
|---|---|---|---|---|
| probe | 0.666305 | 0.237017 | 0.629118 | 0.017981 |
| joint | 0.603521 | 0.209407 | 0.729207 | 0.017691 |

## W4 DoD status

| Item | Status and evidence |
|---|---|
| Reproducibility | Complete: checkpoint/team hashes, commit, schedule/bootstrap seeds in `w4_5_ablation.json` and `w4_5_paired_comparison.json`; frozen model configs. |
| Recent dataset | Complete: `w4_dataset_validation.json`, `datasets/w4/split_ids.json`; temporal split and overlap QA. |
| BC improvement | Met: B−A (data) and C−A (total) in `w4_5_paired_comparison.json`; inconclusive/negative results are not improvements. |
| Team preview | Complete: `w4_team_preview.json`; C−B in `w4_5_paired_comparison.json`. |
| Value head | Evaluation complete, not a guaranteed calibration-gate pass: `w4_value_metrics.json` and `w4_5_value_calibration.json`; outcome never an input. |
| Reliability | Met: `w4_5_ablation.json`; classified hidden-trap retries are reported separately. |
| Final evaluation | Complete: four fixed arms, paired CIs, final checkpoints, both one-shot test splits; all four W4.5 reports. |

## Targets

- C vs Grunt 40%+ stretch target: met (65.4%).
- C vs GymLeader 40%+ stretch target: met (65.8%).
- Random must not degrade: met (significant improvement); +3.80 pp [+1.60, +6.20] — positive.

## Execution evidence

Full benchmark wall time: 797.122 s. Additional command/timing evidence: `w4_5_execution.json`.

Machine-readable sources: `w4_5_ablation.json`, `w4_5_paired_comparison.json`, `w4_5_value_calibration.json`; test intermediate `w4_5_test_results.json`.

The first full run is preserved as `w4_5_ablation_run1.json`: one D-vs-RandomBaseline timeout following a Revival Blessing target rejection; zero model-illegal actions. The fork's Pokemon HP parser retains FNT status after a healthy numeric HP update, so cached fainted flags could select an already revived target. Baseline revival now selects explicit fainted party indices from the authoritative current request and disables native default orders during revival. Exactly one corrected full rerun is reported above; the invalid run is excluded from all estimates.
