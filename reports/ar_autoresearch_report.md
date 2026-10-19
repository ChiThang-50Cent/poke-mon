# Why more pile data stalled, and the autoresearch loop that found the fix

**Verdict:** the plateau was caused by the architecture, not by data volume or model size. C's
encoder mean/max-pools the 87 observation tokens, which throws away which move or which bench
Pokémon sits in which action slot.

- **Round 1 (flatten encoder, 2.03M).** Keeping each token's slot drops pile val NLL from 1.18
  to 0.97. On dev it beats the champion P by +12.1 pp pooled.
- **Round 2 (entity-pointer network, 1.01M).** The model encodes each Pokémon and move as an
  entity and scores each action against the entity it refers to. Pile val NLL is 0.81.
  - On dev it beats P by +23.8 pp pooled [+21.3, +26.4].
  - It beats Metamon's Minikazam (4.7M RNN) in 76.2% of 500 battles, where P wins 49.4%.

Neither model is promoted yet: a fresh-pool final is still required.

## 1. Offline diagnosis (`scripts/w6p_diagnose.py`, `reports/w6p_diagnosis.json`)

| Model | Pile train NLL | Pile val NLL | Gap |
|---|---:|---:|---:|
| P (100k) | 1.158 | 1.190 | 0.032 |
| pile300 | 1.108 | 1.127 | 0.019 |

- Train and val loss are nearly equal, so the model is underfitting (bias), not overfitting
  (variance). More data cannot fix bias.
- The errors are concentrated in choosing which slot, not which kind of action. Pile val, P:
  - right action group (move / switch / tera): 0.86;
  - right slot given the right group: 0.65;
  - forced switches (only switches legal): 0.50;
  - voluntary switches the teacher made: 0.26.
- Hypotheses ruled out:
  - **Train/serve parity:** the tera-type bug in PAC data does not affect us, because
    `DefaultObservationSpace` has no tera tokens.
  - **Overfitting:** see the gap column above.

## 2. Literature (full reports in the session: research_il.md, research_pokemon.md)

- **Small GRUs underfit Metamon data.** Metamon's 0.8–3.5M GRUs underfit the replay data, so they
  started their Transformers at 15M ([arXiv 2504.04395](https://arxiv.org/abs/2504.04395),
  App. F.1). Their per-turn encoder attends over the tokens; it does not pool them.
- **The pile's labels are sampled, not argmax.** Actions are drawn from the policy at temperature
  (the launcher example uses 1.0–2.0) by large actor-critic models
  ([metamon_to_amago.py](https://github.com/UT-Austin-RPL/metamon/blob/main/metamon/rl/metamon_to_amago.py),
  [launcher config](https://github.com/UT-Austin-RPL/metamon/blob/main/metamon/rl/evaluate/ladder_self_play/example_config.yaml)).
  So NLL/accuracy has a high floor. Metamon itself never trains plain BC on this data: it filters
  BC with a critic.
- **Loss and return are coupled.** Imitation-learning scaling laws: return is a smooth function of
  loss with a teacher ceiling, and the compute-optimal model size grows with data
  ([Tuyls et al. 2023](https://arxiv.org/abs/2307.09423)).
- **Hard argmax BC is the weakest target.** In chess, BC on hard argmax labels was the weakest
  target ([Ruoss et al. 2024](https://arxiv.org/abs/2402.04494)).

## 3. Autoresearch harness (`autoresearch/`, pattern from [karpathy/autoresearch](https://github.com/karpathy/autoresearch))

- **`prepare.py` (fixed, read-only).**
  - Data: the pile (285k trajectories), memory-mapped, peak RAM 1.8 GB.
  - Metric: `val_nll` on a fixed 2,000-trajectory sample of `w6p_b/val`. It is computed
    stepwise, exactly like live play, so the model cannot see future turns. `w6p/val` stays
    sealed for promotion.
  - Contract checks: ≤ 20M parameters, ≤ 10 ms per decision, stepwise logits equal to
    full-sequence logits, a value head.
  - Export: `model.py` plus a `state_dict`, which `load_pokedecision` → `load_research_model`
    plays live. Covered by a regression test and a 20-battle smoke run with 0 errors.
- **`train.py`:** the only file the agent edits. Fixed budget of 300 s of training.
- **`program.md`:** the agent's rules. They follow the original, plus a bounded experiment budget,
  the project invariants (inputs, sealed data, no battles), and a promotion path that goes
  through paired battles.

### First run (branch `autoresearch/oct19`)

| Commit | val_nll | top-1 | Params | Status | Change |
|---|---:|---:|---:|---|---|
| d86f408 | 1.1809 | 0.561 | 1.30M | keep | baseline (C architecture, from scratch, 300 s) |
| d86f408 | 1.1849 | 0.559 | 1.30M | — | repeat baseline: noise ≈ 0.004 |
| d64bfa5 | **1.0105** | **0.629** | 2.03M | keep | flatten encoder: each token's slot kept (87×24 concat) |
| (reset) | 1.1852 | 0.559 | 2.03M | discard | control: pooled encoder, GRU 512, same 2.03M parameters |

The parameter-matched control equals the baseline. The gain comes from slot binding, not size.

### Does the proxy hold in play?

The flatten model, trained from scratch for 20 minutes on the pile only (`val_nll` 0.971), was
checked on the sealed `w6p/val` (`reports/ar_flatten_diagnosis.json`):

| Pile val, sealed | P | flatten20 |
|---|---:|---:|
| NLL | 1.190 | 0.972 |
| Accuracy | 0.560 | 0.647 |
| Forced-switch accuracy | 0.500 | 0.637 |
| Voluntary-switch accuracy | 0.259 | 0.443 |
| W4 human val-A accuracy (never trained on human data) | 0.424 | 0.445 |

Paired dev benchmark, `configs/ar_flatten_dev.yaml`: 500 battles per cell, 0 errors,
0 illegal actions (`reports/ar_flatten_dev_benchmark.json`, `ar_flatten_dev_pooled.json`).

| Opponent | P | flatten20 | Δ [95% CI] |
|---|---:|---:|---|
| P | 0.512 | 0.658 | **+0.146 [+0.090, +0.200]** |
| Grunt | 0.776 | 0.876 | **+0.100 [+0.054, +0.146]** |
| GymLeader | 0.792 | 0.910 | **+0.118 [+0.078, +0.160]** |
| Pooled | | | **+0.121 [+0.093, +0.149]** |

## 4. Round 2: a small model built around the decision's structure

The parameter cap is now 4.7M, the size of Metamon's smallest agent; at equal `val_nll`, fewer
parameters wins. The branch is `autoresearch/small-oct19`.

### Entity-pointer network (EPN)

- **Entity encoders.** The observation is split by its fixed layout into 12 entities: global
  context, our active Pokémon, the opponent's active Pokémon, 4 moves and 5 bench Pokémon. Each
  entity *type* has one shared MLP encoder.
- **Entity mixing.** One small self-attention layer runs over the 12 entities, with a type
  embedding per kind, so a move's vector can depend on the opponent.
- **Memory.** A GRU with 256 units over turns.
- **Pointer heads.** Each action is scored by a shared additive scorer between the GRU state and
  the entity that action refers to: move i drives actions i and 9+i (with Tera), bench Pokémon j
  drives action 4+j.

Because weights are shared across move slots and bench slots, the model is permutation-
equivariant over them and stays small (1.01M parameters, 78% of C).

| Commit | val_nll | top-1 | Params | Status | Change |
|---|---:|---:|---:|---|---|
| 7a7aa88 | 0.9716 | 0.649 | 0.93M | keep | EPN without entity attention |
| 70cceca | 0.9373 | 0.660 | 1.01M | keep | + 1 entity self-attention layer |
| dc1330d | 0.9455 | 0.657 | 0.67M | discard | narrower (token 32, GRU 192): Pareto point |
| f29642a | 0.9329 | 0.661 | 1.01M | keep | bf16 autocast: +23% steps |
| bdb0586 | 0.9307 | 0.663 | 1.08M | discard | 2 attention layers: within noise |
| 960a340 | **0.8696** | 0.684 | 1.01M | keep | LR linear warmdown to 0 over the last 50% |
| 1bd3edc | 0.9205 | 0.666 | 1.01M | discard | LR 2e-3 |
| 0d1a54d | 0.8701 | 0.685 | 1.01M | discard | warmdown over 100% of the budget: same |
| cb04a7d | 0.8785 | 0.683 | 1.90M | discard | wider (1.9M): fewer steps in 5 min |
| f1304d9 | **0.8621** | 0.687 | 1.01M | keep | batch 32 |

For reference, flatten (2.03M) scored 1.0105 on the same 5-minute metric.

### Check in play

`epn20` is the last kept configuration trained from scratch for 20 minutes on the pile only
(`checkpoints/autoresearch/epn20`, `val_nll` 0.808).

Sealed `w6p/val` (`reports/ar_epn_diagnosis.json`):

| | P | flatten20 | epn20 |
|---|---:|---:|---:|
| NLL | 1.190 | 0.972 | **0.808** |
| Accuracy | 0.560 | 0.647 | **0.706** |
| Forced-switch accuracy | 0.500 | 0.637 | **0.734** |
| Voluntary-switch accuracy | 0.259 | 0.443 | **0.551** |
| Tera accuracy | 0.258 | 0.349 | **0.467** |
| W4 human val-A accuracy (never trained on human data) | 0.424 | 0.445 | **0.471** |

Paired dev benchmark, `configs/ar_epn_dev.yaml`: 500 battles per cell, 0 errors, 0 illegal
actions (`reports/ar_epn_dev_benchmark.json`, `ar_epn_dev_pooled.json`).

| Opponent | P | epn20 | Δ [95% CI] |
|---|---:|---:|---|
| P | 0.490 | 0.770 | **+0.280 [+0.226, +0.332]** |
| Grunt | 0.720 | 0.924 | **+0.204 [+0.162, +0.246]** |
| GymLeader | 0.714 | 0.944 | **+0.230 [+0.192, +0.270]** |
| Pooled | | | **+0.238 [+0.213, +0.264]** |

Head-to-head against the Metamon reference (`scripts/run_metamon_h2h.py`,
`reports/ar_epn_vs_minikazam.json`):

- Opponent: Minikazam, 4.7M, checkpoint 40, sampled at temperature 1.0, `pokeagent` backend.
- 500 battles on the dev pool, seed 20261018, paired team swaps; 0 errors, 0 illegal actions.

| Our model | Params | Win rate vs Minikazam | Wilson 95% |
|---|---:|---:|---|
| P | 1.30M | 0.494 | [0.450, 0.538] |
| **epn20** | **1.01M** | **0.762** | [0.723, 0.797] |
| epn20 − P (paired) | | **+0.268 [+0.214, +0.322]** | |

Alakazam (~50M, the pile's teacher) and Kakuna (~142M) cannot be loaded yet: their AMAGO
configs require flash-attn, which is not installed.

## Next

1. **Stronger references.** Get Alakazam and Kakuna running locally, either by installing
   flash-attn or with a vanilla-attention override whose weights still load. Then play
   head-to-head against Alakazam: it is the pile's own teacher, so it is the ceiling for pure
   imitation.
2. **More loop budget for EPN.** Directions:
   - training length (20 min gave 0.862 → 0.808, so it is still improving);
   - entity features (types/effectiveness interactions);
   - the narrow Pareto variant (0.67M).
3. **Promotion.** Run one fresh-pool final of the chosen EPN against P, then promote it.
4. **Battle track (outside the loop).** Winner-only or critic-filtered BC, DAgger relabelling,
   self-play RL starting from EPN. These are the routes past the teacher.
