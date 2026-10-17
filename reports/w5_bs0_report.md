# W5.BS0 — Belief-search feasibility (research spike)

**Scope of every number below: simulator value inside the local Showdown simulator, against a
KNOWN opponent policy (the GymLeader / Grunt heuristics). It is NOT Showdown win rate.**
Production C (`models/pokedecision-v1-value`) was frozen and unchanged; nothing was trained;
no live-server battle was played; only the collect pool was used.

Gates: **S1 PASS, S2 PASS, S3 PASS (at 5 s only; 2 s misses by 3 points, 0.5 s fails), S4 PASS at
the pre-declared 2 s budget (Δ = +0.0466, 95 % CI [0.0206, 0.0747]).** Caveat (integrator): the
contract runs S4 only for budgets that pass S3, and the pre-declared 2 s primary did NOT pass S3
(87 % < 90 % of roots with ≥ 4 pairs). The only S3-passing budget, 5 s, gives Δ = +0.0450,
95 % CI [0.0194, 0.0734], which also excludes 0, so S1–S4 all pass at 5 s; the 2 s result is
reported as pre-declared but is not gate-valid on its own.

JSON: `reports/w5_bs0_{1_reconstruction,2_correctness,3_profile,4_benefit}.json`.
Per-root data: `runs/w5-bs0-{1,2,3,4}/` (gitignored). Code: `src/pokeai/cf/{reconstruct,recon_world,
recon_match,recon_search,recon_oracle}.py`, `sim_driver.js` (`reconstruct` handler),
`scripts/bs0_{1_reconstruct,2_correctness,search,report}.py`, `tests/test_bs0_recon.py`.

## What the reconstruction uses (oracle-free)

Inputs: the learner's packed team, its pending request, the public protocol lines it received
(`NeuralPlayer.history`, a new unbounded per-battle record of the p1-channel lines + the frame that
arrives with the request), public Dex/poke-env data, the CF1 belief sampler, and the turn number.
`reconstruct.py` / `recon_world.py` / `recon_match.py` never import `recon_oracle.py` and never
touch `public_log`, `public_view`, `p2_full`, `roster`, `toJSON`, `apply_world` (AST test in
`tests/test_bs0_recon.py`). `recon_oracle.py` (true request/view comparison) is imported only by the
S1 script and the integration test. The true p2 snapshot is never read: p2's python battle is built
from the public history re-expressed for p2 (own side exact HP from the reconstructed sets, opponent
side percentages) plus the reconstructed sim's own p2 request.

Method: a Python tracker folds the public lines into a state (hp%, status, boosts, volatiles,
field, side conditions, PP used, tera, last move, counters, turn clock). A new sim battle is built
from the learner's team (exact, ordered by its request) and the sampled opponent sets, advanced
past team preview, then patched in JS (`reconstruct`): hp/status/boosts/volatiles/PP/items/abilities/
tera/forme/field/side conditions/durations/last move, `refreshRequestFlags` (the DisableMove /
TrapPokemon block of `endTurn`), PRNG, then `makeRequest`. Replaying the public actions on sampled
sets was rejected (it diverges as soon as a sampled field matters).

## Field table (exact / sampled / unsupported)

| field | status |
|---|---|
| learner team, request (hp, status, moves, PP of active, items incl. consumed/swapped, ability, tera type/state, trapped/disabled flags), turn | **exact** |
| learner boosts, volatiles, field, side conditions (as seen in the public log) | exact (tracked from public lines) |
| opponent public hp %, status, boosts, active/fainted, tera used, fainted counts, revealed moves/item/ability | exact (public) |
| opponent unrevealed moves, item, ability, nature, EVs, IVs, tera type | **sampled** (CF1 sampler) |
| opponent exact HP numbers / max HP | sampled (percentage exact; max HP depends on sampled EVs/nature) |
| opponent party order | sampled/arbitrary deterministic: active, revealed in reveal order, then preview order |
| opponent hidden forme in preview (`Zamazenta-*`) before reveal | sampled: base forme assumed |
| PRNG state | sampled (fresh seed per world) |
| weather/terrain/Trick Room/screens/Tailwind remaining turns | exact for the default length; **sampled** when an extending item of an opposing setter is unknown (item sampled) |
| sleep counter | sampled: `startTime` uniform in {max(2,c+1)..4}, c = observed `cant slp` |
| confusion counter | sampled: uniform in [max(2,c+1),5] |
| toxic stage | exact (observed poison ticks in the stint) |
| substitute HP | sampled: floor(maxhp/4) (damage to the sub not tracked) |
| Heal Block/Embargo/Telekinesis/Taunt/Encore/Disable/Yawn/Magnet Rise durations | exact (default lengths via turn clock) |
| partial trapping (Fire Spin, Whirlpool, ...) total duration | sampled: uniform {5,6} (Grip Claw/Binding Band of the opponent not inferred from items) |
| Regenerator switch-out heal of benched opponent mons | sampled (applied only if the sampled ability is Regenerator) |
| Choice lock | inferred from the (sampled) item and the stint's first move; learner's own checked via request |
| protect counter (`stall`) | exact for the last turn only |
| Protosynthesis/Quark Drive boosted stat, Supreme Overlord count, Unburden | exact (from public lines) |
| **unsupported** (root counted unsupported, search falls back to a_C) | mid-turn pivot switch requests (U-turn/Volt Switch/Flip Turn/…; the opponent's pending action is hidden), Future Sight/Doom Desire, Wish pending, Healing Wish/Lunar Dance pending, Outrage/Thrash-style lockedmove, two-turn moves in flight, Burn Up/Double Shock/Roost typechange, Court Change, Illusion (Zoroark), Palafin after a switch-out, transform/mimic/imprison, mega/Z/dynamax |

## S1 — reconstruction (`w5_bs0_1_reconstruction.json`)

100 fresh roots (50 GymLeader / 50 Grunt, C-greedy prefixes, fresh collect pairs not used by any
CF root, seed 20261100), 3 belief worlds per root = 300 worlds. Phases: 0-5: 30, 6-15: 46,
16-30: 19, 31+: 5 roots (22 forced-switch roots).

* supported: **90 / 100 roots (90 %)**, 270 / 300 worlds (gate ≥ 80 %). By phase: 0-5 87 %, 6-15 93 %,
  16-30 95 %, 31+ 60 % (3 of 5).
* unsupported reasons (worlds): pivot_switch_request 12, p2 Future Sight 6, p1 Future Sight 3,
  pending Healing Wish 3, p2 typechange_volatile 3, p2 lockedmove 3.
* (a) learner request equal to the true request on **270 / 270** supported worlds (compared on every
  field; only `rqid` normalised — the harness assigns it);
  (b) 13-slot mask identical on **270 / 270**; (c) public view equal on **270 / 270** (normalisations:
  opponent bench ordered by name, p2 `position`/`shiny` ignored, hidden bookkeeping volatiles
  `choicelock`/`stall` ignored, benched-opponent HP of Regenerator-capable species within 1 point).
* mean reconstruct 3.6 ms, mean world sampling 13 ms.
* A separate 400-root × 2-world run (seed 20269000, not part of the gate) found 4 mismatching roots
  (Embody Aspect re-trigger on Tera, Unburden, partial trapping) which were fixed; the re-run was
  clean on 724 supported worlds / 362 of 400 roots (90.5 % supported).

## S2 — correctness (`w5_bs0_2_correctness.json`)

80 fresh roots × 4 belief worlds × 2 seeds × ≤ 4 candidates (CF1 rule), opponent heuristics with
their python state built from the reconstructed sim from their own point of view, learner = C greedy
with its real python state. 292 / 320 worlds supported (28 unsupported: pivot 12, Future Sight 16);
**2,540 paired rollouts** (mean 0.134 s each).

* determinism: 292 / 292 worlds reproduce the same signature digest, winner and final PRNG;
* public invariance: learner root `tokens_sha`/`numbers_sha`/`hidden_sha`/mask identical across all
  worlds, across candidates and **equal to the true root's** in 292 / 292;
* opponent POV: 0 violations (no leaked unrevealed moves/items, no exact HP of the learner's mons);
* illegal actions 0; unexpected errors 0; protocol retries (`hidden_trap`/`hidden_disable`) **0 observed**
  (that path was not exercised in these 2,540 rollouts).
* Bug found and fixed on the way: the learner probe submitted its choice into the world it probed;
  probes now run on a copy of the world.

## S3 — profile (`w5_bs0_3_profile.json`)

100 fresh roots (seed 20261300), 4 persistent workers (own sim driver each), 8 worlds × 8 seeds
enumerated seed-major, dealt round-robin, all candidates per pair, 4 worker processes (4 CPU cores).
Timings include payload transfer. Decision = highest paired mean Δ vs a_C, override only if
Δ̂ > 0 and Δ̂/SE > 1.0 with ≥ 4 complete pairs, else a_C.

| budget | wall p50 / p95 / max (s) | pairs p50 / p95 | rollouts p50 / mean | roots ≥ 4 pairs | override / fallback | gate |
|---|---|---|---|---|---|---|
| 0.5 s | 0.49 / 0.86 / 1.10 | 0 / 11 | 6.5 / 13 | 21 % | 1 % / 99 % | FAIL (coverage; p95 > 0.55) |
| 2 s | 1.96 / 2.19 / 2.27 | 11 / 62 | 47.5 / 73 | 87 % | 24 % / 76 % | FAIL (87 % < 90 %; p95 ok) |
| 5 s | 4.93 / 5.08 / 5.49 | 32 / 64 | 129 / 149 | **93 %** | 33 % / 67 % | **PASS** |

Critical-path component medians (max over the 4 workers, p50 / p95): payload load 0.05 / 0.06 s,
world sampling 0.02 / 0.03 s, reconstruction 0.007 / 0.009 s, opponent POV build 0.026 / 0.04 s,
rollouts fill the rest of the budget (0.33 s at 0.5 s, 1.79 s at 2 s, 4.76 s at 5 s; mean 0.134 s per
rollout). Reconstruction + sampling + POV (~0.06 s per world) is cheap; rollouts are the cost.
Fallback reasons at 2 s: no positive Δ 50, few pairs 13 (7 unsupported roots + 6 slow), SNR 13.
Unsupported roots: 7 / 100 (pivot 4, lockedmove 2, two-turn 1). Illegal actions 0, unexpected errors 0,
fallback always a legal C action (checked on every decision).
The 1 s tail at 0.5 s (max 1.10 s) is a rollout in flight past the deadline plus the payload load.

## S4 — benefit (`w5_bs0_4_benefit.json`)

200 fresh roots (100 GymLeader / 100 Grunt, seed 20261400, one root per battle, 200 distinct team
pairs; phases 0-5: 33, 6-15: 105, 16-30: 46, 31+: 16). Selection from belief worlds. Evaluation
(PRIMARY, oracle, evaluation-only, independent of selection): true-state rollouts from the real
snapshot, 16 fresh seeds, a_search vs a_C paired (0 when equal), C-greedy continuation, true
heuristic opponent; team-pair cluster bootstrap, 5,000 resamples.

| budget | Δ_search | 95 % CI | override rate | mean Δ / override | positive / negative overrides |
|---|---|---|---|---|---|
| **2 s (primary)** | **+0.0466** | **[0.0206, 0.0747]** | 32 % | +0.146 | 33 / 9 |
| 5 s (descriptive) | +0.0450 | [0.0194, 0.0734] | 30.5 % | +0.148 | 31 / 12 |

At 2 s: by opponent Grunt +0.068 (override 36 %), GymLeader +0.025 (28 %); by phase 0-5 +0.038,
6-15 +0.058, 16-30 +0.039, 31+ +0.012; by candidate count 4 cands (185 roots) +0.049, 3 cands
(6) +0.042, 2 cands (9) 0.0. Fallback rate 68 % (of which unsupported 7 %, few pairs 9.5 %).
Hindsight best-of-candidates (in-sample upper bound) is +0.108. Secondary independent belief-world
rollouts (fresh worlds and seeds, override roots only): mean Δ +0.166 over 64 overrides at 2 s and
+0.179 over 61 at 5 s, consistent with the oracle +0.146/+0.148.
A 5 s budget gives no gain over 2 s here. Statement for the reader: Δ is the mean difference in
simulated win outcome per root versus acting as C greedy; 0.047 means about +4.7 points of
simulated win probability against the known heuristic opponents.

## Deviations and caveats

* Roots are random eligible decisions (`pick_root`), not stratified by phase; 31+ is thin (5 roots
  in S1, 16 in S4). Phase coverage is as reported.
* The learner's own policy output (probs, `a_C`, the candidate list) was taken from the normal play
  path (`candidate_actions` on the root), not from inside the timed search; S2 proves the learner's
  root inputs/mask are byte-identical in every belief world, so this is information-equivalent.
* S3 payload transfer (3.4 MB learner snapshot, 0.05 s) is inside the timed decision. In a live
  setting the workers would hold the battle incrementally.
* Worker GC is disabled during a decision (cyclic-GC pauses of 0.3–1 s on the payload would break the
  budget) and collected between decisions, untimed.
* `NeuralPlayer.history` (unbounded public record) and `Sim.reconstruct/stats` + `sim_driver.js`
  handlers were added; no change to production policy or training code.
* S3 decisions were produced in three invocations from one cached payload set (0.5 s; then 2 s;
  then 5 s) because two runs were killed by the 12 G scope limit (main process had cyclic GC off;
  fixed by collecting between decisions). Each budget's 100 decisions are one uninterrupted pass.
* The S4 secondary evaluation reuses the search workers and only covers override roots.
* Protocol retries (`hidden_trap`/`hidden_disable`): none observed in S2 (2,540 rollouts) or S3/S4.
* Tests: `uv run pytest tests/test_bs0_recon.py` 8 passed; `ruff check` clean on touched files.
