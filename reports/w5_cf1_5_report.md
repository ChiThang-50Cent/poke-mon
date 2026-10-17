# W5.CF1.5 head report (frozen C; offline)

**Val was used for early stopping, (config, tau) selection: every CI below is development evidence only.**

Selected: mlp / pairwise / tau=0.02 (saved head = seed 1); train 1600 roots (400 pilot + 900 sampled + 300 greedy), fresh val 400 roots (300 sampled + 100 greedy).

Gates: H1 FAIL, H2 FAIL, H4 PASS; CF1.6 eligible: False.

## Grid: mean-over-3-seeds val Delta_head by tau

| config | tau=0 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| linear/huber | -0.0097 | -0.0101 | -0.0100 | -0.0084 |
| linear/pairwise | -0.0099 | -0.0099 | -0.0094 | -0.0082 |
| mlp/huber | -0.0099 | -0.0090 | -0.0076 | -0.0020 |
| mlp/pairwise | -0.0016 | -0.0008 | +0.0017 | +0.0002 |

## Fresh val (selected head, saved seed)

| subset | roots | H2 Delta_head | switch rate | Delta given switch | control Delta (3 seeds) |
|---|---|---|---|---|---|
| pooled | 400 | +0.0019 [-0.0042, +0.0081] (n=400) | 0.113 (45) | +0.0167 (n=45) | -0.0036, +0.0006, +0.0005 |
| sampled | 300 | -0.0004 [-0.0077, +0.0073] (n=300) | 0.127 (38) | -0.0033 (n=38) | -0.0012, -0.0002, +0.0019 |
| greedy | 100 | +0.0088 [+0.0000, +0.0198] (n=100) | 0.070 (7) | +0.1250 (n=7) | -0.0106, +0.0031, -0.0037 |

## H1 ranking vs shuffled-label control

| subset | head Spearman | control Spearman | paired diff | beats control |
|---|---|---|---|---|
| pooled | +0.0952 [+0.0441, +0.1496] (n=390) | +0.1013 [+0.0525, +0.1505] (n=390) | -0.0060 [-0.0330, +0.0219] (n=390) | False |
| sampled | +0.1083 [+0.0459, +0.1724] (n=292) | +0.1166 [+0.0598, +0.1763] (n=292) | -0.0083 [-0.0435, +0.0264] (n=292) | False |
| greedy | +0.0562 [-0.0467, +0.1584] (n=98) | +0.0555 [-0.0367, +0.1485] (n=98) | +0.0007 [-0.0421, +0.0449] (n=98) | False |

2-candidate roots with a non-zero rollout dQ: 5 (too few to evaluate sign accuracy).

Seed spread of val Delta_head: [0.0018749999580904841, 0.0009374999790452421, 0.0021875000093132257]; switch rates [0.1125, 0.1275, 0.0425].

Train split (overfit check): Delta_head +0.0043 [+0.0014, +0.0072] (n=1600), switch rate 0.134.

H3 (greedy val roots): Delta +0.0088 [+0.0000, +0.0198] (n=100); warning=False.

H4: {"invalid_actions": 0, "outside_candidate_set": 0, "finite_outputs": true, "hidden_state_input": "none: head inputs are (h, slot one-hot, logp) only"}

## Delta_head by group (pooled val)

| group | roots | Delta | switch |
|---|---|---|---|
| opponent=C | 133 | +0.0014 | 0.083 |
| opponent=Grunt | 133 | +0.0117 | 0.105 |
| opponent=GymLeader | 134 | -0.0075 | 0.149 |
| phase=0-5 | 87 | -0.0072 | 0.207 |
| phase=16-30 | 110 | -0.0023 | 0.082 |
| phase=31+ | 32 | +0.0000 | 0.031 |
| phase=6-15 | 171 | +0.0095 | 0.099 |
| forced_switch=False | 327 | +0.0031 | 0.067 |
| forced_switch=True | 73 | -0.0034 | 0.315 |

## Reading

The control reaches the same per-root Spearman as the head: that ranking is carried by C's own log-prob input, not by the learned state-action link. The head overfits quickly (train loss falls while val Delta_head worsens). CF1.6 is not run because H1 and H2 fail.

Data: 2000 roots from `runs/w5-cf1-4-20261017-104148`.

## Fail-path diagnostics (CF1.5b; offline, same fresh val, no new rollouts)

### 1. Learning curve (nested train subsets, stratified by source, 3 seeds)

| config | roots | val Delta tau=0 | val Delta tau_sel | val Spearman | train Delta tau_sel | train Spearman |
|---|---|---|---|---|---|---|
| mlp/pairwise (tau_sel=0.02) | 400 | -0.0025 | -0.0029 | 0.097 | +0.0177 | 0.236 |
| mlp/pairwise (tau_sel=0.02) | 800 | -0.0021 | +0.0003 | 0.100 | +0.0034 | 0.134 |
| mlp/pairwise (tau_sel=0.02) | 1200 | -0.0010 | +0.0009 | 0.103 | +0.0025 | 0.132 |
| mlp/pairwise (tau_sel=0.02) | 1600 | -0.0016 | +0.0017 | 0.105 | +0.0032 | 0.146 |
| trend mlp/pairwise | | | slope +0.0057/100% (upward=True, monotone=True) | slope +0.010 (upward=True) | | |
| linear/pairwise (tau_sel=0.05) | 400 | -0.0134 | -0.0121 | 0.061 | +0.0615 | 0.577 |
| linear/pairwise (tau_sel=0.05) | 800 | -0.0116 | -0.0111 | 0.072 | +0.0455 | 0.458 |
| linear/pairwise (tau_sel=0.05) | 1200 | -0.0124 | -0.0103 | 0.071 | +0.0405 | 0.436 |
| linear/pairwise (tau_sel=0.05) | 1600 | -0.0099 | -0.0082 | 0.080 | +0.0291 | 0.363 |
| trend linear/pairwise | | | slope +0.0051/100% (upward=True, monotone=True) | slope +0.022 (upward=True) | | |

### 2. Supervised noise

| set | split-half Pearson dQ | SB reliability dQ | dQ signal fraction | centred-Q signal fraction | mean split-half Spearman | SB rank reliability | perfect-predictor Spearman ceiling |
|---|---|---|---|---|---|---|---|
| val | 0.462 | 0.632 | 0.632 | 0.651 | 0.199 | 0.333 | 0.577 |
| train | 0.400 | 0.571 | 0.571 | 0.574 | 0.182 | 0.307 | 0.554 |
| val_sampled | 0.433 | 0.604 | 0.604 | 0.612 | 0.181 | 0.306 | 0.553 |
| val_greedy | 0.532 | 0.694 | 0.694 | 0.727 | 0.255 | 0.406 | 0.638 |

Label-oracle Delta_select on fresh val (CF1.3 cross-fit): pooled +0.0298 [+0.0169, +0.0436] (n=400), sampled +0.0231 [+0.0092, +0.0381] (n=300), greedy +0.0500 [+0.0189, +0.0846] (n=100); a*!=a_C 0.40. In-sample best-minus-a_C (optimistic) 0.0822.

C's log-prob alone: Spearman vs labels +0.1152 [+0.0610, +0.1683] (n=390); captured fraction of the oracle Delta_select: logp-only (always the second-highest candidate) -0.86, saved head +0.06.

### 3. Action coverage (train)

Candidates by type: {'move': 2870, 'switch': 2001, 'move+tera': 1340}; by reason: {'c_greedy': 1600, 'next_highest_prob': 1600, 'uniform_random_other': 1540, 'uniform_random_extra': 562, 'best_tera': 909}; a_C type: {'move': 1161, 'switch': 422, 'move+tera': 17}.

Mean rollout dQ of non-greedy candidates by type:

- move: n=1709, -0.0275 [-0.0370, -0.0179]
- move+tera: n=1323, -0.0200 [-0.0331, -0.0071]
- switch: n=1579, -0.0291 [-0.0406, -0.0174]

By type x phase and type x opponent (full tables incl. 3-way in the JSON):

| cell | n | mean dQ | CI |
|---|---|---|---|
| move|0-5 | 253 | -0.0294 | [-0.0515, -0.0062] |
| move|16-30 | 568 | -0.0438 | [-0.0643, -0.0250] |
| move|31+ | 223 | -0.0006 | [-0.0185, +0.0165] |
| move|6-15 | 665 | -0.0220 | [-0.0365, -0.0081] |
| move+tera|0-5 | 345 | -0.0239 | [-0.0454, -0.0017] |
| move+tera|16-30 | 299 | -0.0071 | [-0.0346, +0.0212] |
| move+tera|31+ | 72 | -0.0243 | [-0.1087, +0.0522] |
| move+tera|6-15 | 607 | -0.0237 | [-0.0432, -0.0031] |
| switch|0-5 | 433 | -0.0140 | [-0.0362, +0.0078] |
| switch|16-30 | 373 | -0.0357 | [-0.0555, -0.0154] |
| switch|31+ | 95 | -0.0401 | [-0.0776, -0.0033] |
| switch|6-15 | 678 | -0.0336 | [-0.0530, -0.0146] |
| move|C | 578 | -0.0148 | [-0.0269, -0.0023] |
| move|Grunt | 579 | -0.0345 | [-0.0548, -0.0144] |
| move|GymLeader | 552 | -0.0335 | [-0.0486, -0.0180] |
| move+tera|C | 411 | -0.0024 | [-0.0192, +0.0143] |
| move+tera|Grunt | 460 | -0.0361 | [-0.0596, -0.0123] |
| move+tera|GymLeader | 452 | -0.0196 | [-0.0444, +0.0056] |
| switch|C | 566 | -0.0136 | [-0.0311, +0.0037] |
| switch|Grunt | 477 | -0.0409 | [-0.0624, -0.0207] |
| switch|GymLeader | 536 | -0.0350 | [-0.0586, -0.0133] |

Head switches on val (tau=0.02): by type {'move': {'switches': 17, 'mean_delta': 0.022058824077248573}, 'switch': {'switches': 28, 'mean_delta': 0.013392857275903225}, 'move+tera': {'switches': 0, 'mean_delta': None}}; by reason {'best_tera': {'switches': 0, 'mean_delta': None}, 'c_greedy': {'switches': 0, 'mean_delta': None}, 'next_highest_prob': {'switches': 32, 'mean_delta': 0.017578125}, 'uniform_random_extra': {'switches': 4, 'mean_delta': 0.046875}, 'uniform_random_other': {'switches': 9, 'mean_delta': 0.0}}.

### 4. Non-learned baselines on val (fitted on train)

| rule | pooled Delta | sampled | greedy | switch rate |
|---|---|---|---|---|
| second_highest_candidate | -0.0258 [-0.0434, -0.0089] (n=400) | -0.0213 [-0.0404, -0.0029] (n=300) | -0.0394 [-0.0788, -0.0024] (n=100) | 1.000 |
| train_mean_type_x_phase | +0.0000 [+0.0000, +0.0000] (n=400) | +0.0000 [+0.0000, +0.0000] (n=300) | +0.0000 [+0.0000, +0.0000] (n=100) | 0.000 |
| tera_when_present | -0.0161 [-0.0314, -0.0012] (n=400) | -0.0181 [-0.0333, -0.0029] (n=300) | -0.0100 [-0.0505, +0.0297] (n=100) | 0.613 |
