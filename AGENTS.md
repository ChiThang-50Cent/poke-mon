# Repository Guidelines

## Goal

A "Jev-like" decision model that plays **Pokémon Showdown Gen 9 OU Singles** (`gen9ou`). It is a **practice bot and research benchmark on a local server**. Never connect training or eval to the public ladder.

The model is small: a structured-state policy + value network with 1–20M params, not an LLM. It is built on Metamon (data, 13-action interface, baselines), poke-env and a pinned local Showdown.

Design, roadmap and gates: `research.md`. The overriding aim is a **strong, usable AI**, not maximal certainty about every change.

## Current state (W1–W5 done)

- **Production: P** = `models/pokedecision-v2-pile`, tag `w6p-pile` (W6-P). Same 1.30M GRU as C, fine-tuned on Metamon self-play; fresh-pool final vs C +9.6 pp, Grunt +5.8, GymLeader +5.8 (`reports/w6p_pile_report.md`).
- **C** = `models/pokedecision-v1-value`, tag `w5-baseline`: previous champion, still the init/reference for W5–W6 scripts.
  - 1.30M GRU BC policy trained on 2025–2026 human replays, plus a probe value head and the pretrained team preview.
  - Win rate: Random 98.4%, Grunt 65.4%, GymLeader 65.8% (`reports/w4_5_final_report.md`).
- **C+Belief Search** (RC4, tag `w5-bs-rc4`) = live root-level search over sampled opponent hidden sets with paired rollouts, 5 s/decision. It is opt-in via `NeuralPlayer(search=LiveSearch)`, or arm `search:` in `evaluate_w5.py`.
  - **Dev-validated only.** Dev, 600 paired battles: Δ +0.267 [0.210, 0.323]; wins/100 Grunt 64→96, GymLeader 65→95, C 54→72 (`reports/w5_bs1_2_report.md`).
  - Independent heldout confirmation is unavailable: BS1.3 and BS1.4 were aborted by infrastructure bugs (now fixed), and BS1.5 was stopped on a cost decision (`reports/w5_bs1_5_closure.md`).
- **Closed W5 variants** (closed as tried; the algorithm families are not refuted, and can be revisited if the data or the estimator/target changes substantially):
  - outcome-weighted AWBC / GAE (`w5_3_final_report.md`, `w5_a2_3_final_report.md`);
  - a supervised counterfactual action head on frozen C features (`w5_cf1_5d_report.md`).
- **"Jev-like"** = a small, non-generative model that takes a structured observation and emits a legal decision directly, with decision quality improved by outcome-driven learning. Opponent head, Q critic and search distillation are tools, not required heads; keep a head only if it makes the fast policy play better or the teacher cheaper/better.
- **W6+ north star: fast-policy strength improvement per unit of training compute** (paired Δ vs C without search, per GPU-h/CPU-h). Search, heuristics and the simulator are training-time teachers only: optimize teacher quality and cost per useful training label, not search inference latency (track coverage, confidence, label time, usable-label rate). Keep C as champion: no GRU→Transformer swap, no parameter growth, one changed component per experiment. Roadmap, gates and experiment card: `research.md` §8.
  - **W6.0** done: distillation feasibility (`reports/w6_0_distill_feasibility.md`). The collector executed-action bug is fixed (`executed_index` + `search_row` in behavior; C's `chosen_index`/probs kept for replay parity).
  - **W6-A** tiers 1–2 done (`reports/w6a_opp_head_report.md`). Labels: `scripts/w6a_build_opp_labels.py` → `datasets/w6a/` (other POV, non-forced row pairing). Head: `PokeDecisionConfig.opponent_classes`. The joint-trained head beats the count baselines, but joint play is not stronger than C (all paired CIs contain 0), so nothing is promoted. Tier 3 (head as search root opponent model, measured on teacher label yield/cost) is proposed, not run. Inputs must be the player view at the simultaneous choice: nothing from the opponent's action in the turn being predicted.
  - **W6-B** (core, independent of W6-A): selective search distillation with the current teacher once labels are confirmed reliable; a better W6-A teacher only feeds the next round. Primary gate: student vs C without search. Collection: `scripts/w6b_collect.py` + `configs/w6b_collect.yaml` (validates the search-row join via `data/search_capture.py`); preflight PASS. Plan, R0 control and pre-registered selection: `reports/w6_0_distill_feasibility.md` §7. Search-augmented trajectories (`search_augmented: true`) are for supervised distillation only: C's probabilities are not propensities, and `critic.include_decision` refuses them.
  - **W6-P done, promoted**: C fine-tuned on 100k Metamon self-play POVs (pac-base gen9ou, Alakazam family; filtered on Colab, `.cache/metamon/w6-pacbase/`) with the W6-A control recipe. Final on fresh `data/teams_w6p/heldout` (now spent): +9.6 / +5.8 / +5.8 pp vs C/Grunt/GymLeader, pooled +7.1 [+4.0, +10.1]. `scripts/w6p_build_pile.py`, `scripts/train_w6p_pile.py`, `reports/w6p_pile_report.md`. A second disjoint batch (199,998 POVs) sits in `.cache/metamon/w6-pacbase-b/`. The W6-B search pilot is paused (cost).
  - **W6-C** (design only): format spec + candidate-action interface + compatibility checks. No UU training or VGC until the OU fast policy clearly improves. Gen 9 UU is not in Metamon `SUPPORTED_BATTLE_FORMATS` (own replays and team pool needed).
  - Later: W7 self-play league (diverse opponents and teams), W8 multi-format adaptation; error analysis from real games against the bot throughout.

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
- heldout pools `data/teams_w5/heldout`, `data/teams_w5_bs14`, `data/teams_w5_bs15`, `data/teams_w6p`.

Partial BS1.3/BS1.4/BS1.5 gameplay is **sealed**: never read it or merge it.

## Architecture

```
Showdown request + public protocol → player-view state (metamon backend)
  → ObservationEncoder (87 tokens + 48 numbers) → GRU encoder
  → policy (13 logits) | value P(win)
  → engine legality mask → masked softmax / argmax → action
```

- **Actions:** Metamon UniversalAction has 13 slots: 4 moves + 5 switches + 4 move+Tera. `-1` means not representable and always carries a reason. Team preview uses a separate model.
- **Masking:** legality comes from the engine, never learned; illegal logits are `-inf`.
- **Policy ≠ win probability.** Only the value head estimates P(win).

## Code map

| Path | Purpose |
|---|---|
| `src/pokeai/env/` | `teams.py` (`load_teamset()`, the only sanctioned team source), `runner.py` (`RunConfig`, bounded concurrent battles → `battles.jsonl` + `summary.json`, lanes/pair windows), `collector.py` (per-decision capture) |
| `src/pokeai/data/` | `features.py` `ObservationEncoder` (**the only** state→features path, shared by training and live play), datasets, splits, validation, replay download |
| `src/pokeai/models/pokedecision.py` | `PokeDecision`, `masked_logits`, `load_pokedecision()` |
| `src/pokeai/training/` | BC training, GAE utilities |
| `src/pokeai/agents/` | `neural_player.py` (`NeuralPlayer`), `live_search.py` (`LiveSearch`: hard deadline, worker lifecycle, fail-fast on unknown exceptions) |
| `src/pokeai/cf/` | Offline sim: `sim_driver.js` (Node sidecar: snapshot/restore/PRNG over pinned `dist/sim`), `offline.py` (server-less matches; typed `RolloutStalled`/`RolloutHorizonExceeded`), `belief.py` (opponent set sampler from Smogon usage stats), `reconstruct.py`/`recon_world.py` (public → sim world; must never import `recon_oracle.py`), `recon_search.py` (anytime paired root search, `opp_model`) |
| `scripts/` | Thin CLIs per milestone: `run_battles.py`, `evaluate_w5.py` (paired benchmark), `bs1_2_run.py` (block × lane driver), `build_teamset.py` (`--w5-fresh-heldout --exclude-dirs`), `pool_claim.py` + `bs14_manifest.py` + `bs_watchdog.py` (final-run claim/verify/watchdog), `bs_rehearsal_{prep,check}.py` |
| `configs/` | Training and benchmark configs (`w5_bs1_2_benchmark.yaml` = dev design) |
| `data/teams/`, `data/teams_w5*/` | Server-validated pools. `teams_w5/collect` = self-play/rollout data, `teams_w5/dev` = selection; the rest are spent |
| `models/` | v0, v1, `v1-value` (C), `v2-candidate` (not production), critic pilot |
| `reports/` | Every milestone's evidence and final report; check here for details instead of this file |
| `third_party/metamon/` | Vendored Metamon `0a00a75` + pinned Showdown (gitignored; **never patch**) |
| `runs/`, `.cache/metamon/`, `datasets/*.npz` | Gitignored outputs and caches |

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

## Tooling

- **Python and Node:** Python 3.11 via `uv` only. Node 22 / npm 11 are for Showdown; npm 11 blocks some install scripts, and the server still works.
- **Hardware:** this box has an RTX 3060 12 GB and 16 threads. The simulator is the bottleneck, not the GPU. Heuristic battles run at ~35–40 games/s; C+search at ~110 s/battle.
- **Resource limits (user rule):** never run heavy jobs unbounded; use the wrapper above. Long battle runs go in a transient `systemd-run --user --unit=…` with `Restart=no`. For the dev benchmark, CPUQuota 1000% / MemoryMax 14G is approved.
- **No LLM fine-tuning.**
