# Why more pile data stalled, and the autoresearch loop that found the fix

**Verdict:** the plateau was caused by the architecture, not by data volume or model size. C's
encoder mean/max-pools the 87 observation tokens, which throws away which move or which bench
Pokémon sits in which action slot. Keeping each token's slot (a "flatten" encoder) drops pile
val NLL from 1.18 to 0.97. On dev, a 20-minute from-scratch model beats the champion P by
+12.1 pp pooled [+9.3, +14.9]. It is not promoted yet: a fresh-pool final is still required, and
the model has 2.03M parameters (+56%), against the W6 "no parameter growth" rule.

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

## Next

1. **Decide the parameter budget.** 2.03M breaks the W6 "no parameter growth" rule, but stays
   within the 1–20M charter. A ≤ 1.30M flatten variant (for example `flatten_dim` 12) is one
   loop experiment away.
2. **Run the autoresearch loop with an agreed budget.** Candidate directions: attention over
   tokens, sequence-memory variants, schedule and batching, bf16.
3. **Promotion.** Fold the best encoder into the full recipe (pile, optionally with human
   pre-training), check the sealed val, run the dev benchmark against P, then one fresh-pool
   final.
4. **Battle track (outside the loop).** Winner-only or critic-filtered BC (Metamon's recipe), then
   DAgger-style relabelling.
