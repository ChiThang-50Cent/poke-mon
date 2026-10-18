# W5.BS1.2b — release-candidate stabilization of C + belief search (no tuning)

Base 14446b9. C, preview model, belief sampler, opponent mixture, candidate rule, anytime decision rule,
D = 5.0 s and M = 0.3 s are unchanged; no `third_party/` patch; heldout never opened (its path appears only
in `configs/w5_bs_rc1.yaml`); no commits or tags. Outputs: `reports/w5_bs1_2b_qa.json`,
`reports/w5_bs_rc1_manifest.json`, `configs/w5_bs_rc1.yaml` (frozen, NOT run), tests, `runs/w5-bs1-2b-*`.

## 1. Fixes (tests: `tests/test_bs1_2b_fixes.py`, `tests/test_bs_rc1_claim.py`, 48 related tests pass)

**(a) Metronome-called move (`CalledForeignConsecutive`).** The parser raises when a move is called by
Metronome/Sleep Talk-like moves, the called move is multi-turn (Fire Spin) and the *parsing side* knows
fewer than 4 moves of the user (`len(pokemon.had_moves) < 4`, `forward.py:499`). The failing line is the
literal sim output; our seat translation and reconstruction do not alter it. It only fails on the harness's
fresh opponent-side python battle (the learner's mon has unrevealed moves) after a rollout RNG outcome that
the live game simply did not draw: the protocol is identical to live, the parser is the limitation. Per the
contract's second branch the rollout failure is handled: `ForwardException` in any candidate drops the
whole (world, seed) pair for EVERY candidate (`dropped_pairs`, reasons logged); any other exception still
propagates. Test: pair dropped for all candidates, only complete pairs emitted, `ValueError` not swallowed.
No pair was dropped in the 240 golden roots, smoke or requal.

**(b) Reconstructed-mask mismatch.** The 7 BS1.2 cases cannot be reproduced offline: no payload or protocol
of those live decisions was kept and the server RNG is unseeded; they are not individually root-caused. The
class was instrumented first: rejected worlds log live and world masks and the root payload is persisted
(world construction is a deterministic function of the payload). The 120-battle run before the fix captured
one case, root-caused exactly: **a Choice Scarf received by Trick mid-stint.** `vol_specs` locked the mon on
the first move of its stint (Roost, 0 PP) instead of the first move after the acquisition (Brave Bird), so
the world's learner had only Struggle while live it had Brave Bird (9 PP) + 2 switches; `|-item|…|[from]
move: Trick` had no handler. Fix in our code: `Tracker._h_item` restarts the lock history when an item is
received (Trick/Switcheroo/Bestow/Thief/Covet/Pickpocket/Magician/Symbiosis). Verified on the captured payloads: before 32/32
world-opp builds rejected (all worlds of both decisions), after 0 rejected. Safety net for every other
cause: before rollouts every world's root mask is computed (`world_root_mask`) and a world where any
live-legal candidate is illegal (or the decision is unrepresented) is rejected for ALL candidates
(`world_rejected_mask`, reason `world_rejected_mask`); no world left ⇒ `search_fallback_unsupported`.
Pairs stay complete.

**(c) Exception monitoring.** `KNOWN_ERRORS` allowlist (`forced_action_outside_mask`,
`parser_forward_exception`; both fixed classes, so any occurrence is reported by category, target 0).
Any other exception ⇒ C's legal action is still returned in time, category `search_error_new_class`,
traceback + root payload written to `fail_dir`, `LiveSearch.abort_reason` set and `_gather_lanes` cancels
the run at once (`RuntimeError: ABORT…`); logged battles stay as evidence. Tests cover classification,
no-abort on a known class, evidence files, and the abort within 2 s.

Also (needed for BS1.3, no algorithm change): `evaluate_w5.claim_pool` accepts the (lane, block) windows of ONE
frozen config for the heldout one-shot claim (identical config required, each window once; tested on tmp pools
only), `scripts/bs1_2_run.py` reads a `run:` config section (BS1.2 behaviour is the default).

## 2. Golden-trace parity (`runs/w5-bs1-2b-golden`, worktree 14446b9 vs working tree)
240 cached roots, 60 each from BS0-3, BS0-4, BS1.0-matrix, BS1.0-deadline (dev/collect; no heldout), fixed
pairs (worlds 0-7 × seed 0 = 8 pairs per candidate), no deadline, identical seeds, both trees run through the
same `_decide_chunk` (old tree via PYTHONPATH). **PASS: 230 roots identical (every pair outcome and the
decision), 10 touched, 0 unexplained, 0 decisions changed (0.0 % ≤ 1 %).** The 10 touched roots contain a
mid-stint item acquisition (choice-lock fix) — roots (src, id): (bs0-3, 19, 32, 38, 42, 57), (bs0-4, 53),
(bs1-0-deadline, 16), (bs1-0-matrix, 4, 35, 47); old and new action are equal in all 10 (5/0/2/6/1, 5, 1, 4/2/5). A first
parity run before the choice-lock fix: 240/240 identical, 0 touched. No root hit a mask reject or a dropped
pair. CPU: old tree 43 min / 11.5 min wall, new tree 23 min / 6.2 min wall (8.3-8.4 GB peak).

## 3-4. Smoke and dev requalification (final code; 2 lanes × 4 workers; `reports/w5_bs1_2b_qa.json`)
Smoke: 12 battles (2/arm/opponent, seed 20261301, lane A Grunt + GymLeader, lane B C-greedy at 20261303), 0 team-pair
overlap with BS1.1/BS1.2. Requal: 120 battles (20/arm/opponent = 10 pairs × 2 directions × 2 arms), seed
20261411 (chosen over 20261401, which overlapped 1 BS1.1/BS1.2 pair; overlap 0), run as the unit `bs12b-requal2`.

| | smoke | requal (120) |
|---|---|---|
| battles valid / timeouts / model-illegal / battle errors | 12 / 0 / 0 / 0 | 120 / 0 / 0 / 0 |
| new-class exceptions | 0 | 0 |
| known-class search errors (forced-mask, parser) | 0 | 0 |
| decisions (search arm) over 5.0 s | 0 of 356 | 0 of 2,330 |
| request→send p50 / p95 / max (s) | 4.541 / 4.705 / 4.707 | 4.523 / 4.702 / 4.707 |
| worlds rejected by mask / pairs dropped (parser) | 0 / 0 | 0 / 0 |
| ≥ 4 complete pairs (searched) | 78.9 % | 84.8 % |
| reconstruction support | 78.9 % | 86.0 % |
| complete pairs p50 / p95 | 13 / 62.3 | 16 / 64 |
| override rate of searched | 10.4 % | 15.8 % |
| categories | override 37, no_signal 244, unsupported 75 | override 362, no_signal 1,614, unsupported 321, single legal 26, not_represented 7 |
| hard deadline hits | 50 | 250 |
| worker restarts | 0 | 0 |
| wall / battle p50 / p95 / max: C | 0.7 / 1.3 / 1.5 s | 0.3 / 1.0 / 1.3 s |
| wall / battle p50 / p95 / max: C+search | 145.7 / 554.0 / 653.6 s | 118.1 / 363.1 / 531.2 s |
| CPU-hours / wall / peak RAM | 1.57 / 15.2 min / 10.9 G | 10.35 / 89.4 min / 10.8 G |
| state integrity (forward counts, hidden digests) | pass | pass |

Gates (each stage): **all PASS** (valid battles, 0 new classes, 0 over 5 s, fixed classes 0).
Win rates are descriptive only (n = 20 per cell): requal wins of 20 — Grunt C 15 / C+search 19, GymLeader 14 / 19,
C-greedy 12 / 15; smoke wins of 2 — C 1/1/1 vs C+search 2/2/2 (Grunt/GymLeader/C).

## Superseded and failed runs (kept as evidence, never merged)
* `runs/w5-bs1-2b-smoke-prefix` (12 battles) and `runs/w5-bs1-2b-requal-prefix*` (120 battles, seed 20261411,
  unit `bs12b-requal`) ran with the new validation + allowlist but BEFORE the choice-lock root-cause fix.
  Smoke-prefix: 0 rejects. Requal-prefix: all gates passed but 2 decisions (battle 73800, turns 54 and 55, learner
  p2) rejected 32 worlds (all of them), the evidence of the Trick/Choice Scarf cause; it also shows the validation
  works: both decisions fell back to C (`search_fallback_unsupported`) without an exception. The fix changed the
  reconstruction, so smoke and requal were re-run on the final code (same seeds) and only those are reported
  above. Pre-fix requal for reference: latency max 4.714 s, ≥ 4 pairs 87.5 %, support 89.3 %, 10.27 CPU-h, 11 G.
* No run aborted; the fail-fast abort never fired (0 new classes).

## 5. Freeze manifest and BS1.3 config (prepared, not tagged, NOT run)
`reports/w5_bs_rc1_manifest.json`: sha256 of the C checkpoint dir (5 files incl. tokenizer.json), preview
checkpoint, tokenizer files (6), belief usage-stat files actually opened (92, recorded by an open() spy), 19
`src/pokeai` modules imported by the live path + 8 scripts, `sim_driver.js`, Showdown repo commit and a hash of
`dist/sim/**`, metamon commit, the frozen config + parsed search parameters, `uv.lock`, Python 3.11.9 / torch
2.14.1+cu130 / numpy 2.4.6 / node v22.20.0. It lists 7 tracked files that are uncommitted (Main commits, then
regenerates or verifies the manifest). `configs/w5_bs_rc1.yaml`: heldout pool, `final: true`, 100 battles /
arm / opponent in 4 pair-blocks 12/13/12/13 × 2 lanes × 4 workers, schedule seed 20261501, D 5.0 / M 0.3 /
mixture, fail-fast with `fail_dir`, 16 h budget; run with `uv run python scripts/bs1_2_run.py --config
configs/w5_bs_rc1.yaml` (first window claims `final.started.json`).

## Deviations
* BS1.2 (b) was not root-caused per BS1.2 occurrence (no data); one captured live case was, plus a generic guard.
* Golden: 8 pairs (8 worlds × 1 seed) per candidate and 240 roots (contract ≥ 200); old tree's `Payload` pickles
  predate `opp_model`, so it is set from the stored opponent name (same for both trees).
* Smoke ran as two lane processes split by opponent (A: Grunt + GymLeader, B: C-greedy), not by pair parity.
* Services: requal ran as systemd user services (CPUQuota 1000 %, MemoryMax 14 G), smoke/golden as scopes (golden 400 %/12 G).
