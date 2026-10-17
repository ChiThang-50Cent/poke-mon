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
