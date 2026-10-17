# W5.2 pilot critic report

Targeted scale-up only; V learns but action signal is not established. Prioritize train slots below 500 samples: [9, 10, 11, 12] (tera-moves); do not broadly scale to 10,000 yet.

## Gates

Primary: battle-equal validation Brier differences; negative is better. A pass requires the entire 95% CI below zero. 5,000 seeded bootstrap resamples cluster by team pair.

| Gate | Difference | Estimate | 95% CI | Verdict |
|---|---|---:|---|---|
| G1 | Brier(V) - Brier(train win rate) | -0.024104 | [-0.035099, -0.013052] | pass |
| G2 | Brier(Q chosen) - Brier(V) | 0.004343 | [0.001594, 0.007122] | not established |
| G3 | Brier(Q chosen) - Brier(action-shuffled Q chosen) | 0.000601 | [-0.002159, 0.003403] | not established |

## Validation summary

| Predictor | Battle-equal Brier | BCE | ECE | AUC |
|---|---:|---:|---:|---:|
| V | 0.225614 | 0.642163 | 0.010281 | 0.678029 |
| Q chosen | 0.229957 | 0.651627 | 0.029975 | 0.663470 |
| train_win_rate | 0.249718 | 0.692583 | 0.009041 | 0.500000 |
| constant_0.5 | 0.250000 | 0.693147 | 0.019067 | 0.500000 |
| logged_probe | 0.223481 | 0.636767 | 0.017338 | 0.682013 |

Advantage/residual correlation: 0.05028018900644902; 95% CI [0.021434992292062707, 0.07847796484785305].
- train: 45080 decisions; slots below 500 samples: [9, 10, 11, 12]; mean behavior entropy 1.3264550854943098 nats.
- val: 10737 decisions; slots below 500 samples: [7, 8, 9, 10, 11, 12]; mean behavior entropy 1.3387451158362051 nats.

## Answers to the pilot questions

1. Does V learn outcome signal on unseen battles? Yes under G1.
2. Does Q add action-dependent signal beyond V? Not established: both G2 and G3 must pass.
3. Is coverage and consistency sufficient to scale? Consistency passes, but only targeted scale-up is justified: prioritize sparse tera-move slots 9–12 and reassess Q action signal before broadly collecting 10,000 battles.

This is one training seed on the 1,000-battle pilot. Team reuse can limit generalization despite team-pair-disjoint splits; individual teams may occur in multiple pairs. Team-overlap metadata: 254.

## Evidence and limitations

- Dataset QA: PASS. Train-only decision win-rate reference: 0.510027.
- Feature alignment max difference: 0.0 on 55,817 C decisions (required ≤1e-5). POV/outcome checks pass across all 1,000 two-POV battles, with zero ties. The dataset includes 45,080 train and 10,737 validation decisions; 12,225 off-policy decisions, 10,057 heuristic decisions and 55 model fallbacks are excluded.
- JSON reports include BCE, Brier, AUC, ECE, reliability bins, first/last decisions, phase and opponent breakdowns under both weightings.
- Advantage–outcome-residual correlation has a team-pair clustered CI; it is diagnostic, not proof that changing the policy improves win rate.
- Coverage enumerates every slot × opponent × phase cell, sparse slots (<500), behavior entropy and legal-but-unchosen actions.
- Only chosen actions are supervised and evaluated against an outcome.
- Chosen-action gains and shuffled-control gains do not establish alternative-action ranking.
- Q slot dispersion is descriptive only; slots have state-dependent semantics.
- Collection sampled C at temperature 1.0; this is not the greedy C value function.
- No actor update, new battles, dev-team selection or heldout evaluation is performed. W5.3 must test actual policy improvement in the simulator.
- These gates share the pilot validation set and are not independent replications. A CI containing zero is inconclusive, not evidence of no effect.

## Execution caveat

Training was performed once. The first invocation stopped during provenance loading because heuristic behavior entries are null. The corrected invocation completed CUDA feature extraction and all three trainings, but report generation failed on integer-mask indexing. Evaluation was finalized from the saved trained heads after fixing boolean-mask normalization, without retraining. Feature extraction was repeated for recovery because the first GPU cache was only in memory; the recovered cache is now persisted. Per-epoch histories from the completed training invocation were not recovered; `config.yaml` states this rather than fabricating them.
