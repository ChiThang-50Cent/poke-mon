# W5 A2.3 — GAE advantages for AWBC (offline)

Only the advantage estimator changed vs A2.1. Evaluation advantage fixed to A_eval = Y − V_probe for every model. Validation split reused many times: exploratory.

## GAE validation

- λ=1 vs Monte Carlo max |diff|: real 0.00e+00, shuffled 0.00e+00.
- Value path parity (no-fallback trajectories): 2.77e-04.
- Fallback steps kept in the chain: 49; trajectories: 1400.

## Advantage statistics

| Estimator | std | corr with MC | sign flips vs MC | ESS |
|---|---|---|---|---|
| lambda_1.00 | 0.4755 | 1.0000 | 0.00% | 0.948 |
| lambda_0.95 | 0.2440 | 0.8299 | 10.07% | 0.985 |
| lambda_0.90 | 0.1842 | 0.6718 | 21.82% | 0.991 |
| lambda_0.95_shuffled_s1 | 0.2747 | 0.0718 | 47.68% | 0.981 |

## Matched kl 0.05 (S_MC, team-pair cluster CI)

| Variant | S_MC (seed mean) |
|---|---|
| E1 | 0.00179 |
| MC | 0.00220 |
| MC-shuffled | 0.00182 |
| GAE0.95 | 0.00196 |
| GAE0.95-shuffled | 0.00183 |
| GAE0.90 | 0.00191 |

| Contrast | Estimate [95% CI] |
|---|---|
| GAE0.95_minus_E1 | +0.00017 [-0.00018, +0.00050] (inconclusive) |
| GAE0.95_minus_GAE0.95-shuffled | +0.00013 [-0.00024, +0.00049] (inconclusive) |
| GAE0.90_minus_E1 | +0.00012 [-0.00013, +0.00037] (inconclusive) |
| GAE0.90_minus_GAE0.95-shuffled | +0.00008 [-0.00022, +0.00037] (inconclusive) |
| MC_minus_E1 | +0.00041 [-0.00025, +0.00106] (inconclusive) |
| MC_minus_MC-shuffled | +0.00038 [-0.00041, +0.00116] (inconclusive) |
| GAE0.95_minus_MC | -0.00024 [-0.00065, +0.00018] (inconclusive) |

Per seed (primary contrasts):

- s1: GAE0.95_minus_E1 +0.00017 [-0.00020, +0.00052] (inconclusive); GAE0.95_minus_GAE0.95-shuffled -0.00005 [-0.00052, +0.00039] (inconclusive)
- s2: GAE0.95_minus_E1 +0.00017 [-0.00017, +0.00051] (inconclusive); GAE0.95_minus_GAE0.95-shuffled +0.00020 [-0.00028, +0.00067] (inconclusive)
- s3: GAE0.95_minus_E1 +0.00017 [-0.00018, +0.00050] (inconclusive); GAE0.95_minus_GAE0.95-shuffled +0.00023 [-0.00021, +0.00070] (inconclusive)

## Matched kl 0.10 (S_MC, team-pair cluster CI)

| Variant | S_MC (seed mean) |
|---|---|
| E1 | 0.00248 |
| MC | 0.00279 |
| MC-shuffled | 0.00260 |
| GAE0.95 | 0.00256 |
| GAE0.95-shuffled | 0.00250 |
| GAE0.90 | 0.00254 |

| Contrast | Estimate [95% CI] |
|---|---|
| GAE0.95_minus_E1 | +0.00008 [-0.00043, +0.00058] (inconclusive) |
| GAE0.95_minus_GAE0.95-shuffled | +0.00007 [-0.00050, +0.00060] (inconclusive) |
| GAE0.90_minus_E1 | +0.00006 [-0.00032, +0.00043] (inconclusive) |
| GAE0.90_minus_GAE0.95-shuffled | +0.00005 [-0.00043, +0.00048] (inconclusive) |
| MC_minus_E1 | +0.00031 [-0.00070, +0.00126] (inconclusive) |
| MC_minus_MC-shuffled | +0.00019 [-0.00099, +0.00129] (inconclusive) |
| GAE0.95_minus_MC | -0.00023 [-0.00085, +0.00041] (inconclusive) |

Per seed (primary contrasts):

- s1: GAE0.95_minus_E1 +0.00005 [-0.00049, +0.00057] (inconclusive); GAE0.95_minus_GAE0.95-shuffled -0.00038 [-0.00106, +0.00025] (inconclusive)
- s2: GAE0.95_minus_E1 +0.00020 [-0.00031, +0.00071] (inconclusive); GAE0.95_minus_GAE0.95-shuffled +0.00035 [-0.00035, +0.00104] (inconclusive)
- s3: GAE0.95_minus_E1 -0.00002 [-0.00056, +0.00050] (inconclusive); GAE0.95_minus_GAE0.95-shuffled +0.00022 [-0.00046, +0.00092] (inconclusive)

## Gradient alignment at θ = C (100 battle-level permutation nulls)

| λ | real cos(train, val) | null mean ± std | null p95 | null ≥ real |
|---|---|---|---|---|
| 1.00 | -0.0291 | -0.0049 ± 0.0339 | +0.0582 | 73% |
| 0.95 | +0.0645 | +0.0148 ± 0.0409 | +0.0748 | 9% |
| 0.90 | +0.1259 | +0.0394 ± 0.0388 | +0.0911 | 3% |

## Gate (KL 0.05)

- matched: True
- GAE0.95_gt_E1: False
- GAE0.95_gt_shuffled: False
- gradient_alignment_above_null: False
- reliability: True
- pass: False

Decision: A2.3 is a negative experiment; no battles. Stop tuning lr/batch/λ on this pilot (user rule).

Caveats: shuffled outcomes keep TD residuals, so the control isolates only the terminal label; GAE bias depends on the weak probe critic (val Brier 0.2235); single pilot dataset; validation reused.

Wall time: 678.0 s.
