# W5.BS1.3-P fix: QA (gates R1-R7)

Uncommitted tree on HEAD 003fe66. BS1.3 gameplay stays sealed (the failing payload was used only for correctness reproduction). Data: `reports/w5_bs1_3p_fix_qa.json`, `runs/w5-bs1-3-fix/` (`phase1.md`/`phase1.json`, golden, smoke, scripts).

## Cause
`sim_driver.js` bootstrapped with a real `team 123456` start. In world 3, Landorus's Intimidate triggered the sampled Deoxys's Eject Pack. That left `midTurn=true`, a consumed p2 lead item, and a started-but-open `switch` request that was then overwritten. With `midTurn=true` the first rollout turn had no residual, and a mon KO'd before moving got no forceSwitch, so the harness spun for 600 rounds. Forced-switch roots also had `midTurn=false`, so the residual ran twice. The comparison artefact flagged in the earlier fidelity check was the `fnt` status token (0 real mismatches).

## Changes
* Layer 1 (`sim_driver.js`): leads are built with blank item and ability for `team 123456`, then item, ability and states are restored exactly. The lead's once-per-battle `shieldBoost`/`swordBoost` flags are preserved. Forced roots get `midTurn=true`, `faintedThisTurn` and the correct `faintedLastTurn`. Active Neutralizing Gas is a new Unsupported reason (`neutralizing_gas_active`).
* Layer 2 (`recon_world.py`): `world_invariant_violations` over the new `invariants` driver command raises `WorldRejectedState`, an Unsupported subclass. The world is dropped for every candidate and counted as `world_rejected_state`.
* Layer 3 (`offline.py`, `recon_search.py`, `live_search.py`): `RolloutStalled` fires after K=4 rounds with an unchanged progress fingerprint. `RolloutHorizonExceeded` fires at 600 rounds without a proven stall. Either voids the whole decision (C's action, 0 pairs, `fallback:rollout_stalled` / `fallback:rollout_horizon`, forensics kept). Other exceptions still fail fast.

## Gates
| gate | result |
|---|---|
| R1 | Cause reproduced in the engine (stage trace; old tree: root `midTurn=true`, p2 item lost). Synthetic Eject Pack lead: the root equals the Leftovers control, item kept, first-turn residual runs. Failing payload through the fixed `_decide_chunk`: 256 rollouts, 16 worlds, 0 aborts/rejects/errors (old tree: 8/256 failed). |
| R2 | `world_invariant_violations` covers normal and forced roots (`midTurn`, queue, `requestState`, request keys, switch flags, fainted active, `faintedThisTurn`, `pokemonLeft`, `choiceError`). Tests reject on any violation. Golden: 1920 world builds, 0 state rejects. |
| R3 | 8 payload worlds: 0 mismatches over fainted, active, hp% ±1, status, boosts, side conditions, weather, terrain, turn, `pokemonLeft`. The earlier anomaly was the `fnt` token artefact. |
| R4 | Tests (18 in `tests/test_bs1_3p_fix.py`, all pass): learner KO'd before moving gives forceSwitch; opponent KO'd before moving gives a faint plus residual; invalid-choice loop gives `RolloutStalled`; stall gives `RolloutStalled`; legitimate repeats (K locked) never stall; a genuine long rollout gives `RolloutHorizonExceeded`. |
| R5 | A `WorldRejectedState` world is removed from all candidates (test). Stall or horizon voids the whole decision and falls back to C (test). |
| R6 | See below: PASS, 0 unexplained. |
| R7 | PASS (see below). |

### R6 golden parity (240 cached roots, 1746 worlds with real reconstructions, 8 pairs/candidate, HEAD 003fe66 versus working tree)
* 229 roots are identical in every pair outcome and decision. 11 roots differ.
* 8 differing roots are forced-switch roots (37 forced roots in total; the second-residual fix).
* 2 differing roots are Neutralizing Gas worlds, now Unsupported: bs0-3/41 goes 8 to 0 pairs, matrix/2 goes 8 to 2 pairs. 14 NG worlds in total.
* 1 differing root has a lost sampled lead item (matrix/39, Booster Energy), with the same action.
* Action changed in 1 of 240 roots (0.42%): deadline/21, a forced root (6 to 4; old `fallback:snr`, new `override` with 8 pairs). The compare script marks PASS: 0 unexplained.
* State-diff over all 1920 world builds (old bootstrap versus new): the only differing paths are `lastItem`, `item`/`itemState` (45 worlds in 15 roots), forced-root `midTurn`/`faintedThisTurn` (216 worlds in 27 roots), a stale `speedOrder` (92 worlds in 10 roots, new is correct), and the stale `speed`/`disabled` of a fainted active (forced roots). A first run exposed a regression (lead Dauntless Shield `shieldBoost` flag lost), which is fixed. The golden run was then repeated.
* 0 mask rejects, 0 state rejects, 0 aborts, 0 exceptions on all 240 roots.
* Unsupported (new tree): pivot_switch_request 80, neutralizing_gas_active 14, courtchange 8, plus the existing move_state/twoturnmove/pending reasons. The old tree has the same reasons without neutralizing_gas_active.

### R7 requal (final code, 2 lanes x 4 workers, systemd CPUQuota 1000%, MemoryMax 14G, no heldout)
| | smoke (12, seeds 20261701/703) | requal (120, seed 20261711) |
|---|---|---|
| battles valid / timeouts / model-illegal / battle errors | 12 / 0 / 0 / 0 | 120 / 0 / 0 / 0 |
| new-class exceptions | 0 | 0 |
| decisions over 5.0 s | 0 of 217 | 0 of 2080 |
| request to send p50 / p95 / max (s) | 4.47 / 4.66 / 4.70 | 4.54 / 4.70 / 4.70 |
| world_rejected_state / world_rejected_mask | 0 / 0 | 0 / 0 |
| RolloutStalled / RolloutHorizonExceeded | 0 / 0 | 0 / 0 |
| pairs dropped (parser) | 0 | 0 |
| searched with >=4 pairs (BS1.2b) | 65.1% (78.9%) | 88.0% (84.8%) |
| reconstruction support (BS1.2b) | 65.1% (78.9%) | 88.7% (86.0%) |
| categories | no_signal 109, override 31, unsupported 75, single legal 2 | no_signal 1471, override 319, unsupported 227, single legal 61, not_represented 2 |
| CPU-h / wall / peak RAM | 0.83 / 7.1 min / 10.9 G | 9.34 / 77.8 min / 10.9 G |

Unsupported reasons, requal: pivot_switch_request 89, move_state:wish 51 (p1 19, p2 32), pending wish 20, pending healingwish 8, pending lunardance 11, move_state:futuresight 13, pending futuresight 7, outrage 7, lockedmove 9, twoturnmove 11, throat chop 1, lightscreen duration 1. Neutralizing Gas appeared in 0 requal decisions. Smoke: courtchange 59 (concentrated in a few battles), pivot_switch_request 12, lockedmove 3, twoturnmove 1.
The smoke coverage is lower than BS1.2b's; the sample is 12 battles and 59 of the 75 unsupported decisions are `courtchange`, which this change does not touch. The requal (n=120) is at or above BS1.2b, so there is no coverage drop. Seeds differ, so the comparison is descriptive.

## Tests
83 passed: `tests/test_bs0_recon.py`, `test_bs1_1_live.py`, `test_bs1_2b_fixes.py`, `test_bs1_deadline.py`, `test_cf_offline.py`, `test_bs1_2_lanes.py`, `test_bs_rc1_claim.py`, `test_bs1_3p_fix.py`. The full suite was not run by the subagent.

Integrator addendum: before commit 17e609a the integrator ran the full suite on this code: `pytest -m "not server"` 442 passed, `pytest -m server` 4 passed, `ruff check src scripts tests` clean (after 4 lint-only fixes in `tests/test_bs1_3p_fix.py`). See `reports/w5_bs1_4_preflight.md` § Full test suite.

## Known limits
* Benched mons that already entered earlier in the live battle do not carry once-per-battle ability flags (same as before the change).
* Neutralizing Gas, Trace and Imposter persistent state is not in the spec. Active NG and Imposter (`|-transform|`) are Unsupported.
* K=4 is my definition of a stall.
* `reports/w5_bs_rc1_manifest.json` hashes `sim_driver.js` and the changed modules, so the manifest must be regenerated after the commit.
