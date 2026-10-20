# W8 approach research: how a ≤4.7M model could surpass Kakuna (Oct 2026)

Question: after W7, production KK (1.01M EPN, pure Kakuna distillation) wins 43.8% vs Kakuna.
More data and 3.5× width did not help. Which newer methods, architectures and input features
could take a small model past Kakuna?

Method: three parallel web surveys (Pokémon SOTA; beating a teacher; small architectures and
damage-calc features), primary sources preferred. Evidence tags:
- **[SELF]**: self-reported by the authors;
- **[1-SRC]**: single source;
- **[INF]**: our inference.

## 1. State of the art, Gen 9 OU and neighbours

| System | Params | Training signal | Strength | Notes |
|---|---:|---|---|---|
| Kakuna (Metamon, Dec 2025) [1][2] | 142M | Offline RL (AMAGO), 4M human + ~18M self-play replays; last stage higher-temperature self-play | Local Gen9OU GXE 76%; human ladder ~71% | Still "the best public metamon model"; no newer Gen 9 OU checkpoint found |
| FoulPlay [1][3] | no net | Root-parallel MCTS (DUCT, simultaneous moves) on a Rust engine; hand evaluation; usage-stat set inference with damage/speed/item filters | **Won NeurIPS 2025 PokéAgent Gen 9 OU (final 50–14)**; ~80% GXE, 1879 Elo | Author: set prediction matters as much as search |
| Q [1] [SELF] | 50M AC | Heuristic-bot curriculum, then self-play with a fixed expert "coach" | Gen 9 OU finalist | |
| PA-Agent [1] [SELF] | Metamon backbone | 6 rounds of iterative offline RL on model-vs-model battles, human data 100%→10% | Gen 1 OU champion | |
| Jaxcalibur [4] [SELF][1-SRC] | **8.5M** | Pure self-play PPO, ~100M games, 1 week on 1 H100, JAX engine | Gen 9 **Random Battles** #1: 2557 Elo, 95.3 GXE (FoulPlay ~2341) | Damage-calc matchup tokens, opponent-set and next-action aux heads; search only +100–150 Elo |
| Minikazam [2] | 4.7M RNN | Offline RL | Local GXE 50% | Only public sub-10M Gen 9 OU model |

LLM agents (PokéChamp, Porimon 2026, GRPO on a 1.5B Qwen) trail RL and search; the PokéAgent
paper says RL and search dominated [1][5][6].

- **No public sub-10M Gen 9 OU agent matches Kakuna.**
- **Jaxcalibur is the one small-model proof point**, but it plays Random Battles, its results
  are self-reported, and it was not compared with Kakuna.

## 2. Getting past the teacher

| Approach | Evidence | Cost here | Fit [INF] |
|---|---|---|---|
| Search as teacher (expert iteration): belief-sampled DUCT/Gumbel search at selected states, distilled into the policy [7][8][9] | Strongest in this exact format (FoulPlay) | Thousands of sim calls per searched decision. Mitigation: search only where KK and Kakuna disagree or KK has high entropy, ≤1 s per decision | **High**: Node snapshot/restore sidecar and belief sampler already exist (W5 search: +26.7 pp dev for C) |
| KL-anchored online RL (piKL / AlphaStar-style SL init + PPO with KL to the anchor) against a small league [10][11][12] | Well validated for improving on an imitation init; PPO is competitive in imperfect-information games [12] | Millions of battles; the simulator is the bottleneck (~15–20k battles/h optimistic) | Medium: exploitability and team-overfit risk (VGC-Bench [13]; the GRPO repo overfit its training team [6]) |
| On-policy (DAgger) relabelling: KK's own battles labelled by Kakuna [14] | Classic remedy for covariate shift; no Pokémon evidence | Cheap: offline labeller exists, ~8k decisions/s | Medium-low upside, still teacher-bounded |
| Extrapolated / ensemble distillation (ExOPD 2026; Born-Again) [15][16] | Student beats teacher only outside games | Cheap, offline | Low; a baseline at most |
| Offline RL on the same data | Metamon: RL variants differ little; gains came from new self-play data [2] | — | Low: Kakuna already used this recipe on this data |

## 3. Architecture and input features

- **Damage-calc features: high value, untested in small nets.** A calc needs both sides'
  species, stats (EV/IV/nature), item, ability, boosts, Tera, move and field [17].
  - Hidden for the opponent: EVs/nature, item, ability until revealed, unrevealed moves, Tera.
  - Prior work fills the gap with usage stats and filters:
    - FoulPlay samples sets and filters them by observed damage, move order, weather duration
      and status-move use, then groups damage rolls into KO / no-KO [3];
    - PokéChamp estimates unknown attack/defence from usage stats [5].
  - Effect sizes:
    - a damage-formula one-step lookahead bot beats a max-power bot 44% vs 16% [5];
    - type-advantage annotations raised PokéLLMon from 36% to 55% [18];
    - Jaxcalibur feeds calcs as 24 matchup tokens [4].
  - **No published ablation of calc features inside a small learned policy.**
- **Leakage rules:**
  - features may use only public observations plus fixed priors;
  - never the true opponent set, nor Metamon's reconstructed full opponent team;
  - priors come from usage stats dated before the eval period;
  - one code path for replays and live play.
- **Opponent revealed moves, PP, items as explicit slots.** Metamon added
  `OpponentMoveObservationSpace` (Kakuna uses it). It notes PP is hard to recover from memory and
  caused PP-stall failures [2][19].
- **Aux heads.** Jaxcalibur reports that an opponent next-action head made the bot "substantially
  stronger", with opponent item/ability/move/Tera heads alongside [4] [SELF]. Our replays have
  both POVs, so the labels are free.
- **Memory.** No ≤5M Pokémon evidence.
  - GRU is fine for moderate memory; Mamba-2 or Transformer-XL only for the longest-memory
    tasks [20].
  - Metamon reduced its emphasis on long-term memory for the challenge baselines [2].
  - Jaxcalibur uses only the last 16 events [4].
  - Change memory last.
- **Small versus long training.** Jaxcalibur found smaller nets trained longer beat bigger nets
  [4]. This matches W7: width 3.5M gave no gain.

## 4. Proposed W8 order (one variable per step; dev gate each; fresh-pool final only for a promotion candidate)

1. **E1, feature block v0** (encoder change).
   - Per-action keys:
     - type multiplier, STAB, base power, accuracy, priority (free);
     - modal-set damage % and P(KO) dealt;
     - damage taken from the opponent's best revealed or likely move;
     - P(outspeed).
   - Slots for the opponent's revealed moves, PP and item.
   - Same decision rows as now, so the 33M Kakuna labels are reused.
   - Arm: KK recipe + features vs KK. Gate: dev vs KK, plus vs Kakuna.
2. **E2, aux heads:** opponent next action and opponent set (moves/item/ability/Tera); labels
   from the other POV. Loss-only change.
3. **E3, belief-averaged calc features:** usage-stat set belief with FoulPlay-style filters,
   replacing the modal set. Only if E1 shows calc features matter.
4. **E4, search as teacher:**
   - DUCT or Gumbel root search over sampled worlds at disagreement / high-entropy states, ≤1 s;
   - targets = search policy (+ value), KL-anchored to Kakuna labels;
   - this is the main route past Kakuna.
5. **E5, KL-anchored PPO:** a small league (Kakuna, KK checkpoints, Grunt/GymLeader), a KL term
   to the KK anchor and diverse teams; gate against team overfit.
6. **Memory swap** (minGRU or a short causal window + explicit slots) last.

`ObservationEncoder` changes break the train/serve-parity invariant in AGENTS.md. E1 therefore
ships a new encoder version, re-encodes the pile, and adds regression tests for every new
feature's leakage.

## Sources

1. PokéAgent Challenge paper, arXiv 2603.15563 — https://arxiv.org/html/2603.15563v2
2. Metamon README (model table, observation spaces, post-challenge) — https://github.com/UT-Austin-RPL/metamon ; Metamon paper — https://arxiv.org/abs/2504.04395
3. Foul Play write-up (P. Mariglia) — https://pmariglia.github.io/posts/foul-play/
4. Jaxcalibur [SELF] — https://jaxcalibur.github.io/
5. PokéChamp, arXiv 2503.04094 — https://arxiv.org/abs/2503.04094
6. SFT→GRPO 1.5B Gen 9 OU repo [1-SRC] — https://github.com/alphawasd/pokemon-stuff ; Porimon — https://arxiv.org/abs/2609.32544
7. Gumbel AlphaZero/MuZero (ICLR 2022) — https://www.iclr.cc/virtual/2022/spotlight/6419
8. Re-determinizing IS-MCTS — https://arxiv.org/html/1902.06075v2
9. Student of Games — https://arxiv.org/abs/2112.03178v2 ; ReBeL — https://arxiv.org/abs/2007.13544
10. piKL — https://arxiv.org/pdf/2112.07544 ; DiL-piKL — https://arxiv.org/abs/2210.05492
11. AlphaStar — https://www.nature.com/articles/s41586-019-1724-z ; magnetic mirror descent — https://arxiv.org/html/2206.05825v1
12. Reevaluating policy gradient in imperfect-information games — https://arxiv.org/abs/2502.08938
13. VGC-Bench (AAMAS 2026) — https://arxiv.org/abs/2506.10326
14. DAgger — https://proceedings.mlr.press/v15/ross11a.html
15. G-OPD / ExOPD (2026) — https://arxiv.org/abs/2602.12125
16. Born-Again Networks — https://arxiv.org/abs/1805.04770
17. Smogon damage calc — https://github.com/smogon/damage-calc ; usage stats — https://www.smogon.com/stats/
18. PokéLLMon — https://arxiv.org/html/2402.01118v2
19. Metamon paper (PP-stall, observation) — https://arxiv.org/html/2504.04395
20. RLBenchNet [1-SRC] — https://arxiv.org/html/2505.15040v1 ; minGRU — https://arxiv.org/pdf/2410.01201
