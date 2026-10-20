# Repository Guidelines

## Goal

A "Jev-like" decision model that plays **Pokémon Showdown Gen 9 OU Singles** (`gen9ou`). It is a practice bot and research benchmark. **Training, selection and evaluation run on the local server only.** The one exception is `scripts/ladder_bot.py`: by user decision, it plays the production model on the public rated ladder against humans, as a read-out and source of real games for error analysis. Ladder games are never used for selection.

The model is small: a structured-state policy + value network, ≤ 4.7M params (user decision), not an LLM. It is built on Metamon (data, 13-action interface, baselines), poke-env and a pinned local Showdown.

Design, roadmap and gates: `research.md`. The overriding aim is a **strong, usable AI**, not maximal certainty about every change.

## Direction on branch `harness` (user decision, 20/10/2026)

The project pivots from "one small model" to a **ruleset-agnostic, self-improving game harness**. Study and evidence: `reports/harness_research.md`.

- **Ruleset pack per format** (gen9ou, gen8ou, gen4ou…): a Showdown format as the exact oracle, a fast engine build (poke-engine, one build per gen), a per-gen mechanics table, usage-based set priors and a team pool. Everything above the pack is format-agnostic; swapping gens is a config change.
- **Search plays.** DUCT/determinized MCTS over sampled opponent sets runs at play time. Live search is allowed again, which replaces the W6+ "fast policy without search" north star.
- **The small model (≤ 4.7M, Jev-like) works inside search**: it is the prior, the value, opponent/belief heads and a fast fallback. It must become a candidate scorer over an explicit action list with engine-computed features, so it can run under any pack. KK is the first prior.
- **LLMs never choose moves.** They stay off the critical path: loss diagnosis and proposed diffs to evaluators, filters and adapters.
- **Self-improvement loops**: expert iteration (search → distill the scorer) and code evolution under a frozen evaluator. The evaluator stays outside every agent's write scope.
- **Gated steps**:
  1. engine behind a `World` interface, with differential tests vs Showdown;
  2. search-only gen9ou;
  3. same code on gen4ou (portability test);
  4. scorer as prior/value;
  5. self-improvement loops.
- `main` keeps the KK line as it is.

## Current state (W1–W7 done; W8 paused)

- **Production: KK** = `models/pokedecision-v4-kk`, tag `w7-kk` (W7). Same 1.01M EPN architecture, trained 60 min from scratch on Kakuna soft labels (`train_epn.py --kakuna-weight 1.0`) over the whole 905k-trajectory Alakazam-family pile; fresh-pool final vs EPN +6.0 pp [+0.6, +11.4], Grunt +0.2, GymLeader +1.8, pooled +2.7 [+0.5, +4.9] (`reports/w7_next_lever_report.md` §5). Vs Kakuna (dev, 500): 43.8% [39.5, 48.2], paired vs EPN +14.8 [+9.4, +20.2] (`reports/w7_v4_vs_kakuna.json`). Pure distillation: its ceiling is ≈ Kakuna; beating Kakuna needs a non-imitation signal (RL from outcomes, search as teacher).
- **Previous: EPN** = `models/pokedecision-v3-epn`, tag `w7-epn`. Entity-pointer network, 1.01M params, BC on the Metamon pile; fresh-pool final vs P pooled +19.6 [+17.1, +22.1] (`reports/w7_epn_final_report.md`). Vs Metamon: Minikazam 76.2%, Alakazam 41.2%, Kakuna 29.0%.
- **Previous: P** = `models/pokedecision-v2-pile`, tag `w6p-pile` (W6-P). Same 1.30M GRU as C, fine-tuned on Metamon self-play; fresh-pool final vs C +9.6 pp, Grunt +5.8, GymLeader +5.8 (`reports/w6p_pile_report.md`).
- **C** = `models/pokedecision-v1-value`, tag `w5-baseline`: previous champion, still the init/reference for W5–W6 scripts.
  - 1.30M GRU BC policy trained on 2025–2026 human replays, plus a probe value head and the pretrained team preview.
  - Win rate: Random 98.4%, Grunt 65.4%, GymLeader 65.8% (`reports/w4_5_final_report.md`).
- **C+Belief Search** (RC4, tag `w5-bs-rc4`) = live root-level search over sampled opponent hidden sets with paired rollouts, 5 s/decision. It is opt-in via `NeuralPlayer(search=LiveSearch)`, or arm `search:` in `evaluate_w5.py`.
  - **Dev-validated only.** Dev, 600 paired battles: Δ +0.267 [0.210, 0.323]; wins/100 Grunt 64→96, GymLeader 65→95, C 54→72 (`reports/w5_bs1_2_report.md`).
  - Independent heldout confirmation is unavailable: BS1.3 and BS1.4 were aborted by infrastructure bugs (now fixed), and BS1.5 was stopped on a cost decision (`reports/w5_bs1_5_closure.md`).
- **Closed W5 variants** (closed as tried; the algorithm families are not refuted, and can be revisited if the data or the estimator/target changes substantially):
  - outcome-weighted AWBC / GAE (`reports/w5_3_final_report.md`, `reports/w5_a2_3_final_report.md`);
  - a supervised counterfactual action head on frozen C features (`reports/w5_cf1_5d_report.md`).
- **"Jev-like"** = a small, non-generative model that takes a structured observation and emits a legal decision directly, with decision quality improved by outcome-driven learning. Opponent head, Q critic and search distillation are tools, not required heads; keep a head only if it makes the fast policy play better or the teacher cheaper/better.
- **W6+ north star: fast-policy strength improvement per unit of training compute** (paired Δ vs C without search, per GPU-h/CPU-h). Search, heuristics and the simulator are training-time teachers only: optimize teacher quality and cost per useful training label, not search inference latency (track coverage, confidence, label time, usable-label rate). **Model size (user decision):** small, structure-aware architectures, ≤ 4.7M params (below Metamon's smallest agent, Minikazam), at least as strong as the Metamon reference. Architecture changes are allowed; one changed component per experiment. Roadmap, gates and experiment card: `research.md` §8.
  - **W6.0** done: distillation feasibility (`reports/w6_0_distill_feasibility.md`). The collector executed-action bug is fixed (`executed_index` + `search_row` in behavior; C's `chosen_index`/probs kept for replay parity).
  - **W6-A** tiers 1–2 done (`reports/w6a_opp_head_report.md`). Labels: `scripts/w6a_build_opp_labels.py` → `datasets/w6a/` (other POV, non-forced row pairing). Head: `PokeDecisionConfig.opponent_classes`. The joint-trained head beats the count baselines, but joint play is not stronger than C (all paired CIs contain 0), so nothing is promoted. Tier 3 (head as search root opponent model, measured on teacher label yield/cost) is proposed, not run. Inputs must be the player view at the simultaneous choice: nothing from the opponent's action in the turn being predicted.
  - **W6-B** (core, independent of W6-A): selective search distillation with the current teacher once labels are confirmed reliable; a better W6-A teacher only feeds the next round. Primary gate: student vs C without search. Collection: `scripts/w6b_collect.py` + `configs/w6b_collect.yaml` (validates the search-row join via `data/search_capture.py`); preflight PASS. Plan, R0 control and pre-registered selection: `reports/w6_0_distill_feasibility.md` §7. Search-augmented trajectories (`search_augmented: true`) are for supervised distillation only: C's probabilities are not propensities, and `critic.include_decision` refuses them.
  - **W6-P done, promoted**: C fine-tuned on 100k Metamon self-play POVs (pac-base gen9ou, Alakazam family; filtered on Colab, `.cache/metamon/w6-pacbase/`) with the W6-A control recipe. Final on fresh `data/teams_w6p/heldout` (now spent): +9.6 / +5.8 / +5.8 pp vs C/Grunt/GymLeader, pooled +7.1 [+4.0, +10.1]. `scripts/w6p_build_pile.py`, `scripts/train_w6p_pile.py`, `reports/w6p_pile_report.md`. Scaling to 300k POVs (second disjoint batch `.cache/metamon/w6-pacbase-b/`, `datasets/w6p_b/`) is inconclusive on dev vs P (pooled +2.1 [−0.7, +4.9]); not promoted. The W6-B search pilot is paused (cost).
  - **Autoresearch** (`autoresearch/`, karpathy pattern; `reports/ar_autoresearch_report.md`): fixed harness `prepare.py` (memory-mapped pile, stepwise `val_nll`, contract checks, cap 4.7M params, export loadable by `load_pokedecision`), agent-edited `train.py`, rules in `program.md`; `datasets/w6p/val` is sealed for promotion. Root cause of the W6-P plateau: C's mean/max token pooling loses action-slot binding (underfit; train≈val).
    - **EPN** (entity-pointer network, 1.01M; `autoresearch/train.py` on main) encodes entities with shared per-type encoders, runs one entity self-attention layer and a GRU, and scores each action against the entity it refers to.
    - `epn20` (20 min from scratch, pile only): `val_nll` 0.808 vs P 1.19; dev vs P +23.8 pp pooled [+21.3, +26.4]; vs Minikazam 76.2% (P 49.4%); **vs Kakuna (142M) 29.0% [25.2, 33.1]**, so it is not yet at Metamon's best (`reports/ar_epn_vs_kakuna.json`).
    - Promoted after the W7 fresh-pool final (see Production).
  - **W6-C** (design only): format spec + candidate-action interface + compatibility checks. No UU training or VGC until the OU fast policy clearly improves. Gen 9 UU is not in Metamon `SUPPORTED_BATTLE_FORMATS` (own replays and team pool needed).
  - **W7** done: EPN promoted (`reports/w7_epn_final_report.md`), then KK (Kakuna soft-label distillation, `scripts/w7_kakuna_label.py`, `scripts/train_epn.py`; `reports/w7_next_lever_report.md`).
  - **W8** (plan: `research.md` §8), **paused by the user** after E1:
    - **E1 closed, no gain** (`reports/w8_e1_report.md`): a 66-column public matchup block (`src/pokeai/data/matchup.py`, `ObservationEncoder(extras="matchup-v0")`, `scripts/w8_build_extras.py`, train with `--config extras=66`). e1 − ctl pooled over KK/GymLeader/Kakuna −0.8 [−3.5, +1.9]. Input-side levers stay bounded by the Kakuna teacher.
    - E2 (training-only opponent aux heads: `Config.aux`, `train_epn.py --aux-weight`) is coded and tested, not run. E3 dropped. E4 (search as teacher) and E5 (KL-anchored RL) not started.
  - **Public ladder read-out**: `scripts/ladder_bot.py` (account `PokeDecisionKK`, credentials in gitignored `.omp/ps/credentials.env`, unit `w8-ladder`, output `runs/ladder_kk_v4/`). Never kill a running ladder unit mid-battle: the opponent is a human waiting on us.

## Evaluation process (W6+)

The process is deliberately light:
1. unit tests;
2. smoke run;
3. bounded dev benchmark;
4. **one** final evaluation on a fresh pool.

- Scale battle counts only when the answer could change a technical decision.
- Do not build more claim/lock/watchdog infrastructure; what exists is enough.
- Win-rate claims come from paired simulator battles: same teams and both directions, team-pair cluster bootstrap. A CI containing 0 means **inconclusive**.
- Select on dev only.

**Spent data (never use again for selection or evaluation):**
- W3 and W4 test splits;
- heldout pools `data/teams_w5/heldout`, `data/teams_w5_bs14`, `data/teams_w5_bs15`, `data/teams_w6p`, `data/teams_w7`, `data/teams_w8`.

Partial BS1.3/BS1.4/BS1.5 gameplay is **sealed**: never read it or merge it.

## Architecture

```
Showdown request + public protocol → player-view state (metamon backend)
  → ObservationEncoder (87 tokens + 48 numbers [+ optional extras])
  → EPN: per-type entity encoders → entity self-attention → GRU over turns
  → policy (13 logits, each scored against the entity its action refers to) | value P(win)
  → engine legality mask → masked softmax / argmax → action
```

Older models (C, P) use a GRU over pooled tokens (`src/pokeai/models/pokedecision.py`).

- **Actions:** Metamon UniversalAction has 13 slots: 4 moves + 5 switches + 4 move+Tera. `-1` means not representable and always carries a reason. Team preview uses a separate model.
- **Masking:** legality comes from the engine, never learned; illegal logits are `-inf`.
- **Policy ≠ win probability.** Only the value head estimates P(win).

## Code map

| Path | Purpose |
|---|---|
| `src/pokeai/env/` | `teams.py` (`load_teamset()`, the only sanctioned team source), `runner.py` (`RunConfig`, bounded concurrent battles → `battles.jsonl` + `summary.json`, lanes/pair windows), `collector.py` (per-decision capture) |
| `src/pokeai/data/` | `features.py` `ObservationEncoder` (**the only** state→features path, shared by training and live play), `matchup.py` (W8 E1 extras), `search_capture.py` (search-row join), datasets, splits, validation, replay download |
| `src/pokeai/models/pokedecision.py` | `PokeDecision`, `masked_logits`, `load_pokedecision()` |
| `src/pokeai/training/` | BC training, GAE utilities |
| `src/pokeai/agents/` | `neural_player.py` (`NeuralPlayer`), `live_search.py` (`LiveSearch`: hard deadline, worker lifecycle, fail-fast on unknown exceptions) |
| `src/pokeai/cf/` | Offline sim: `sim_driver.js` (Node sidecar: snapshot/restore/PRNG over pinned `dist/sim`), `offline.py` (server-less matches; typed `RolloutStalled`/`RolloutHorizonExceeded`), `belief.py` (opponent set sampler from Smogon usage stats), `reconstruct.py`/`recon_world.py` (public → sim world; must never import `recon_oracle.py`), `recon_search.py` (anytime paired root search, `opp_model`) |
| `scripts/` | Thin CLIs per milestone: `run_battles.py`, `evaluate_w5.py` (paired benchmark, `--lanes`), `run_metamon_h2h.py` + `metamon_h2h_worker.py` (paired h2h vs a Metamon pretrained agent via named challenge, `--lanes`), `w8_pairs.py` (paired CIs from a benchmark report + h2h runs), `train_epn.py` (EPN/KK training), `w7_kakuna_label.py`, `w7_build_cache.py`, `w8_build_extras.py`, `ladder_bot.py` (public ladder), `bs1_2_run.py` (block × lane driver), `build_teamset.py` (`--w5-fresh-heldout --exclude-dirs`), `pool_claim.py` + `bs14_manifest.py` + `bs_watchdog.py` (final-run claim/verify/watchdog). Closed one-off experiment runners were removed; their results stay in `reports/` and git history |
| `autoresearch/` | Autonomous architecture search: `prepare.py` (fixed harness: `uv run python autoresearch/prepare.py` once → `.cache/autoresearch/`), `train.py` (the only file the agent edits), `program.md` (rules + run command). Results in untracked `results.tsv` |
| `configs/` | Training and benchmark configs (`w8_e1_dev.yaml` = latest dev design; `w5_bs1_2_benchmark.yaml` = search benchmark) |
| `data/teams/`, `data/teams_w5*/` | Server-validated pools. `teams_w5/collect` = self-play/rollout data, `teams_w5/dev` = selection; the rest are spent |
| `models/` | v0, v1, `v1-value` (C), `v2-pile` (P), `v3-epn` (EPN), `v4-kk` (KK, production); research exports are `model.py` + `best.pt` (state dict), loaded via `load_research_model`; `v2-candidate` (not production), critic pilot |
| `reports/` | Every milestone's evidence and final report; check here for details instead of this file |
| `third_party/metamon/` | Vendored Metamon `0a00a75` + pinned Showdown (gitignored; **never patch**) |
| `runs/`, `.cache/`, `checkpoints/`, `datasets/**/*.np[yz]` | Gitignored outputs and caches; small dataset sidecars (`report.json`, `encoder.json`, `kakuna_train.json`, `extras.json`) are tracked |

## Commands

```bash
# Setup
git clone --recursive https://github.com/UT-Austin-RPL/metamon.git third_party/metamon
git -C third_party/metamon checkout 0a00a75 && git -C third_party/metamon submodule update --init
uv sync && (cd third_party/metamon/server/pokemon-showdown && npm install)
export METAMON_CACHE_DIR=$PWD/.cache/metamon

# Local server (keep running; :8000, no auth)
(cd third_party/metamon/server/pokemon-showdown && node pokemon-showdown start --no-security)

# Resource wrapper for anything heavy (user rule)
W="systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=12G nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8"

# Battles
$W uv run python scripts/run_battles.py --player neural:models/pokedecision-v1-value --opponent GymLeader --battles 20
# Paired dev benchmark (arms/opponents/N in config; C+search ≈ 110 s/battle, use lanes)
$W uv run python scripts/bs1_2_run.py --config configs/w5_bs1_2_benchmark.yaml

# Tests / lint
$W uv run pytest -m "not server"    # ~480 offline tests, ~2 min
uv run pytest -m server             # needs the local server
uv run ruff check src scripts tests
```

Older pipelines (W2–W5 dataset builds, training, CF/BS experiments) have one script each in `scripts/`. Their exact invocations are in the matching `reports/*` file.

Baselines: `metamon.baselines.get_all_baseline_names()` (`RandomBaseline`, `Grunt`, `GymLeader`, …).

## Invariants

- **Hidden information.**
  - Policy and search inputs come only from the acting player's view, the request and the public Dex. Never read unrevealed opponent sets from the true state or from replay futures.
  - Rollouts on the true sim state are *privileged labels* only.
  - Search never sees the real opponent's identity.
- **Leakage rule.** Allowed: the public protocol up to the decision plus Dex knowledge. That includes a revealed ability or a species' unique ability, `-formechange`, Skill Swap, and a form's forced item/Tera.
  - `[of]` is not always the ability owner (Pickpocket/Magician/Frisk/Trace).
  - Every new protocol form needs a regression test.
- **Train/serve parity.** Live play uses the `metamon` backend and `ObservationEncoder` (`DefaultObservationSpace-v1`, vocab 2,542). Changing the encoder invalidates datasets and checkpoints.
- **BC.** Cross-entropy only on valid steps; full trajectories with one uninterrupted hidden state, as live play does.
- **Splits.** A whole battle goes into one split, by battle id and time; val/test are frozen before training.
- **`battle_result` is never a policy input.** Q(s,a) from logs is observed only for the chosen action.
- **Trajectories** use the Metamon native lz4 format, one per POV. Project metadata lives in the `trajectory_meta/` sidecar, never inside `trajectories/`. Join only via `manifest.json`.
- **Self-play logging.** Store the exact behavior probs, the chosen index, the mask and the `policy_version` plus checkpoint sha256. Offline replay must reproduce the probs. Greedy only for benchmarks.
- **Reliability.** `illegal_actions` stays 0, and any non-null battle `error` is a bug to root-cause. Validation scripts FAIL on any non-zero error count.

## Environment gotchas

- **Team legality:** Metamon sets drift from the pinned banlist, and poke-env hangs on rejection. Use only validated pools, and rerun `build_teamset.py` after bumping Showdown.
- **poke-env fork:** a packed team string in `ConstantTeambuilder` breaks; the runner wraps teams in a minimal `Teambuilder`.
- **Revival Blessing:** offer only fainted targets, otherwise the battle stalls silently.
- **Hidden disable (Imprison) / hidden trap:** the server answers `[Unavailable choice]` with a revised request. Treat it as a protocol retry, not as model-illegal.
- **Switch by request party index**, not species name: stale form names (Cramorant) break the latter.
- **Import order:** `pokeai/__init__.py` imports `metamon.baselines` first, because of the GenData constructor. Always import through `pokeai`.
- **Server URL:** always `ws://localhost:8000/showdown/websocket`; never the public auth endpoint.
- **W4 raw data:** strip the `^\d+-` filename prefix, keep one copy per (battle, POV) and prefer the unprefixed one. Drop battles whose two POVs disagree on the result.
- **Search:** reconstructed worlds that violate engine invariants are rejected (`world_rejected_state`). A stalled or over-horizon rollout makes the whole decision fall back to C. Unknown exceptions abort the run (fail-fast).
- **`pool_lock` creates `final.lock`.** For a read-only check, call `bs14_manifest.verify_bundle` directly.
- **Final runs pin Node:** the release manifest records the shell's nvm Node (v22.20.0); a bare `systemd-run` unit resolves the system Node and the claim is refused. Pass `--setenv=PATH=$PATH`.
- **RAM:** the box already uses ~16 of 30 GB. A pile training job loading both npz batches peaks at ~10 GB; never run two such jobs at once (the kernel OOM killer took one). Autoresearch memory-maps instead (~2 GB).
- **Metamon h2h:** vendored `wrappers.py` references `PokeAgentPlayer` without importing it (the `pokeagent` backend needed by Minikazam); the worker injects it at runtime. Alakazam/Kakuna use AMAGO `FlashAttention` (flash-attn has no wheel for torch 2.14/cu130); run them with `--metamon-attention flex` (AMAGO `SlidingWindowFlexAttention`, same window, strict weight load). `--player baseline:<Name>` pits a heuristic against the Metamon agent. Kakuna ≈ 3.8 s/battle on GPU.

## Tooling

- **Python and Node:** Python 3.11 via `uv` only. Node 22 / npm 11 are for Showdown; npm 11 blocks some install scripts, and the server still works.
- **Hardware:** this box has an RTX 3060 12 GB and 16 threads. The simulator is the bottleneck, not the GPU. Heuristic battles run at ~35–40 games/s; C+search at ~110 s/battle.
- **Resource limits (user rule):** never run heavy jobs unbounded; use the wrapper above. Long battle runs go in a transient `systemd-run --user --unit=…` with `Restart=no`. For the dev benchmark, CPUQuota 1000% / MemoryMax 14G is approved.
- **Per-job CPU/RAM profile (measured W8).** CPU is high only for multi-process jobs; most eval jobs are one process:
  - `evaluate_w5.py` / `run_metamon_h2h.py`: **one Python process ≈ 1 core** (asyncio + batch-1 inference), ~2 GB RSS, plus Showdown `node` ~1 GB. Neural-vs-neural dev benchmark ≈ 70 battles/min. A high CPUQuota does not speed it up.
  - Data builds (`w6p_build_pile.py`, `w8_build_extras.py --workers N`): N cores while encoding, then near 0 while concatenating/saving. `w8_build_extras` on w7_c: 11 min with 4 workers, 6 GB peak.
  - `train_epn.py`: ~1 core + GPU, ~5 GB.
  - High total RAM on this box is mostly desktop apps (browsers, IDE), not these jobs. Colab High-RAM does not help.
  - **To go faster, split battles into lanes**: N processes `evaluate_w5.py --lanes N --lane i` (≈ N cores, 2 GB each), merged afterwards as `bs1_2_run.py`/`bs1_2_report.merge_block` do. Use 4 lanes for 500-battle cells.
  - **RAM is shared with the user's desktop (~20 GB of browsers/IDE).** Before running lanes in parallel, check `free -m` "available" and cap each unit with `-p MemoryMax=… -p MemorySwapMax=0`, so our job is killed instead of the user's apps. 4 Kakuna h2h lanes (~2.1 GB each) once exhausted RAM and crashed the user's browser; 2 capped lanes are safe.
- **No LLM fine-tuning.**
