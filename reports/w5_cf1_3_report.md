# W5.CF1.3 signal report

Roots: 400 (train/val {'train': 320, 'val': 80}).

| variant | G3 roots | split-half Spearman [95% CI] | null p95 | G3 | G4 roots | delta_select [95% CI] | G4 | frac a*!=a_C |
|---|---|---|---|---|---|---|---|---|
| belief_greedy | 380 | 0.210 [0.160, 0.259] | 0.042 | PASS | 400 | +0.0258 [+0.0122, +0.0398] | PASS | 0.41 |
| oracle_greedy | 93 | 0.365 [0.261, 0.468] | 0.091 | PASS | 100 | +0.0838 [+0.0437, +0.1269] | PASS | 0.39 |
| belief_t1 | 93 | 0.143 [0.045, 0.245] | 0.092 | PASS | 100 | +0.0131 [-0.0112, +0.0400] | FAIL | 0.46 |
| belief_greedy (control roots) | 93 | 0.217 [0.112, 0.315] | 0.086 | PASS | 100 | +0.0256 [-0.0025, +0.0569] | FAIL | 0.44 |

C greedy is the best candidate in 0.552 of roots (descriptive).

## G4 by group (belief_greedy)

| group | roots | delta | CI | frac a*!=a_C |
|---|---|---|---|---|
| opponent=C | 134 | +0.0084 | [-0.0084, +0.0252] | 0.47 |
| opponent=Grunt | 133 | +0.0315 | [+0.0080, +0.0564] | 0.39 |
| opponent=GymLeader | 133 | +0.0376 | [+0.0108, +0.0663] | 0.36 |
| phase=0-5 | 82 | +0.0450 | [+0.0122, +0.0808] | 0.54 |
| phase=16-30 | 111 | +0.0084 | [-0.0101, +0.0298] | 0.31 |
| phase=31+ | 36 | +0.0226 | [-0.0139, +0.0694] | 0.29 |
| phase=6-15 | 171 | +0.0285 | [+0.0066, +0.0515] | 0.43 |
| split=train | 320 | +0.0182 | [+0.0047, +0.0328] | 0.39 |
| split=val | 80 | +0.0563 | [+0.0203, +0.0961] | 0.48 |
| forced_switch=False | 330 | +0.0208 | [+0.0070, +0.0343] | 0.39 |
| forced_switch=True | 70 | +0.0491 | [+0.0089, +0.0938] | 0.51 |

## Belief vs oracle dQ (control roots, diagnostic)

`{"pairs": 283, "pearson": 0.5017680595665874, "sign_agreement": 0.6554054054054054, "pairs_both_nonzero": 148}`

## dQ precision (belief_greedy)

`{"pairs": 1141, "mean_se": 0.09526111038751522, "mean_ci_width": 0.37342355271905975, "fraction_ci_excluding_0": 0.14986853637160386}`
