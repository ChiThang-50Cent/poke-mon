# W5.BS1.1 — live integration of C + belief search on the local Showdown server (pilot)

Local server :8000 only, dev pool `data/teams_w5/dev`, `--concurrent 1`, 4 search workers, C
(`models/pokedecision-v1-value`) and its preview checkpoint unchanged, no training, heldout
untouched, no commits. **BS1.1a: L1–L4 PASS. BS1.1b ran to the 2 × 10 battles / arm / opponent
(120 battles; projected total wall 2.4 h < 4 h).** The gameplay numbers are exploratory
(n = 20 per arm × opponent) and are NOT a ladder win rate.

JSON: `reports/w5_bs1_1_{integration,runtime,paired_pilot}.json`. Runs: `runs/w5-bs1-1a`,
`runs/w5-bs1-1b-block1`, `runs/w5-bs1-1b-block2` (+ the aborted `runs/w5-bs1-1b-aborted1`).
Code: `src/pokeai/agents/live_search.py`, `neural_player.py` hook, `runner.py` (`RunConfig.search`),
`scripts/evaluate_w5.py` (arm option `search`, `battle_timeout_s`), `configs/w5_bs1_1_pilot.yaml`,
`scripts/bs1_1_report.py`, `tests/test_bs1_1_live.py`. Pre-registered constants unchanged:
D = 5.0 s, M = 0.3 s, skip search below 0.5 s, `opp_model=mixture`, schedule seed 20261101.

## Engineering (requirements 1–7)

1. **Hook**: `NeuralPlayer(search=...)`, class default `None` is a strict no-op: no extra state,
   snapshots, softmax or timestamps (test: no new attributes besides the always-present
   `forward_count`; with a stub search that never overrides, C's order, hidden state and
   forward count equal a search-less twin). `RunConfig.search` is dropped from `asdict` when
   `None` so summaries are unchanged; `evaluate_w5` arm option `search: {...}`.
2. **End-to-end deadline**: the request clock starts in `_handle_battle_message` before the
   vendor parses the request (before C inference). Per decision rows (`record["search"]`)
   carry `t_request, t_choose, t_c_done, t_search_start, t_search_end, t_choice_returned,
   t_send` (after `await send_message`) and the category. Search budget = D − elapsed − M.
   `choose_move` blocks the event loop for the search (one battle at a time).
3. **Worker lifecycle**: `SearchPool.usable()`/`maintain()` are non-blocking; a dead or stuck
   worker is respawned asynchronously, `usable()` is False meanwhile and the decision uses C
   immediately (`search_skipped_workers`). `settle()` is no longer called on the live path.
   Worker restarts counted (`pool.restarts`): **0** in all runs. A worker that is still on
   the previous decision accepts the next one queued (absolute deadlines still hold).
4. **Recurrent state**: C forwards once per represented decision (counter per row); the search
   works on a deep copy of a pre-frame snapshot restored inside worker processes; the live
   hidden state is digest-compared before/after the search. A trap/disable retry restores the
   memory and reuses the same frame snapshot (test with search on).
5. **Fail-safe / categories** (as pre-registered plus `search_skipped_workers`,
   `search_skipped_no_snapshot`, `not_searched_single_legal`, `not_represented`): any exception ⇒
   C's legal action; `protocol_retry:<reason>` stays in the existing retry records.
6. **Isolation**: every pool message carries the decision id (late results dropped); frame
   snapshots/start offsets/request clocks are per battle tag and cleared at battle end.
7. **Inputs**: team, request, battle snapshot, protocol history only; `Payload` has no
   opponent identity. **Seat handling (found during the smoke)**: with `alternate_challenger`
   the learner is seated as p2 in half the battles; the sim always seats the learner as p1, so
   the history/request are translated to the canonical seat (`offline.swap_roles`) and every
   frame delivered to the learner's real python state is translated back (names too), so its
   GRU/battle state stays untouched. Both seats were exercised (969 p1 / 1,070 p2 decisions).

## BS1.1a — smoke (12 battles, seed 20261101; `reports/w5_bs1_1_integration.json`)

* **L1 PASS**: 0 timeouts, 0 agent-illegal actions, 0 unexpected errors (battle or search).
* **L2 PASS**: 278 search-arm decisions, 0 over D; request→send p50/p95/max 4.532 / 4.702 /
  4.705 s. Phases (p50 / max): C inference + state 0.004 / 0.04 s; gap C→search 0 s; search
  4.529 / 4.702 s; search end→send 0.001 / 0.002 s.
* **L3 PASS**: 0 per-decision forward violations (1 forward per represented decision, 0 otherwise),
  0 decisions where the hidden digest changed, 0 cross-battle state (≤ 1 hidden entry, tables empty
  at battle end), cumulative forward counter = sum of logged forwards. Offline replay parity of
  the logged observations was NOT run (the live server RNG is not seedable); covered by the counters,
  digests and tests.
* **L4 PASS**: categories exactly: override 38, fallback_no_signal 180, fallback_unsupported 58,
  not_searched_single_legal 2; 77.2 % of searched decisions had ≥ 4 complete pairs; pairs
  p50 12 / max 64; unsupported reasons: courtchange 40 (one battle), pivot 11, healingwish 4,
  futuresight 3.

## BS1.1b — runtime (`reports/w5_bs1_1_runtime.json`)

Block 1 (seed 20261101) 4,466 s, block 2 (seed 20261111; chosen to avoid reusing block-1 team
pairs) 4,072 s: total 8,538 s = 2.37 h. Budget rule: projected total after block 1 = 2 × 4,466 s =
2.48 h ≤ 4 h ⇒ block 2 ran (decided before any win rate was computed).

* 120 battles (60 per arm), 0 timeouts, 0 battle errors, 0 agent-illegal actions, 0 protocol retries.
* Wall per battle (p50 / p95 / max): C 0.25 / 0.87 / 1.10 s; C+search 102.6 / 380.6 / 654.3 s.
  Decisions per battle (p50 / p95 / max): C 31 / 85 / 172; C+search 26 / 85 / 142.
* Latency over 2,057 search-arm decisions: **0 over D**, request→send p50 / p95 / max 4.526 /
  4.703 / 4.719 s; search phase 4.523 / 4.699 / 4.703 s; search end→send ≤ 0.017 s.
* Search exceptions: **2** (0.10 %), both inside a reconstructed world: "forced action N outside mask"
  (a candidate legal in the live mask is not legal at the world's root). The fail-safe returned C's
  action in time; both decisions are `search_error` rows (vs C, turns 2 and 38).
* Worker restarts 0. Peak memory 5.6 GB (scope peak) for both blocks; CPU 4 h 52 min over 74.6 min
  (block 1) and 4 h 27 min over 68.0 min (block 2) ≈ 3.9 cores.
* Search availability (2,057 decisions): override 335 (16.4 % of searched), fallback_no_signal 1,556,
  fallback_unsupported 148, search_error 2, not_searched_single_legal 16. Searched 2,039;
  **89.7 %** had ≥ 4 complete pairs; reconstruction support (≥ 1 world built) 92.7 %; complete
  pairs p50 20 / p95 64; rollouts p50 85 / p95 256; hard deadline fired on 281 decisions (13.8 %).
  By opponent: Grunt 774 decisions (override 102, ≥ 4 pairs 89.5 %), GymLeader 674 (override 84,
  86.7 %), C 609 (override 149, 93.5 %). Unsupported reasons (decisions): pivot_switch_request 54,
  Wish 18, Future Sight 23, two-turn 21, Healing Wish 9, Court Change 10, Throat Chop 5, Lunar Dance 4,
  lockedmove/outrage 4, screen duration 3.
* State integrity over 60 search battles: 0 violations.

## BS1.1b — exploratory gameplay (`reports/w5_bs1_1_paired_pilot.json`; EXPLORATORY, n = 20)

Wins / battles (ties are non-wins), 2 blocks pooled:

| opponent | C | C+search | paired Δ (C+search − C) | 95 % cluster bootstrap |
|---|---|---|---|---|
| Grunt | 8 / 20 | 20 / 20 | +0.600 | [0.350, 0.800] |
| GymLeader | 14 / 20 | 19 / 20 | +0.250 | [0.050, 0.450] |
| C (greedy) | 9 / 20 | 15 / 20 | +0.300 | [0.000, 0.600] (inconclusive) |
| pooled | | | **+0.383** | **[0.233, 0.533]** (30 pair clusters) |

CIs use the W5 benchmark's team-pair cluster bootstrap (10 clusters per opponent, 5,000 resamples).
Directional only: 20 battles per cell, one pool of 200 dev teams, opponents are the fixed heuristics
and C itself; do not read this as a claim about Showdown ladder strength.

## Failed / aborted runs and deviations

* `runs/w5-bs1-1b-aborted1` (first block-1 attempt): the second arm (C+search vs Grunt) hit the
  900 s battle timeout on a 205-decision battle ⇒ `evaluate_w5` raised "invalid battle". Cause: the
  `battle_timeout_s` I had set (900 s) was too small for 5 s per decision. Kept as evidence, not
  merged into any number; rerun as a new run id with `battle_timeout_s: 3600`. **Disclosure**: while
  diagnosing this I printed that aborted run's per-battle winners (C vs Grunt 5/10, C+search 8/10 before
  the abort) before the final runs; this did not influence the stop rule (time-based) or the reported
  numbers (all taken from the final blocks).
* BS1.1a used `battle_timeout_s: 900`; the pilot config now says 3600 (no BS1.1a battle came close to 900 s). The
  default 120 s battle timeout is kept for the C arm (unchanged behaviour).
* Two earlier debug smokes (not counted) found: the p2-seat translation, username canonicalisation
  (`|player|` names → CFp1/CFp2 for the opponent's fresh battle) and sim→live account names in
  `|win|` for the learner's metamon parser; all fixed before BS1.1a (their runs are under `runs/2026*`
  and `/tmp/t`).
* Offline-replay parity (L3) not run (see above).
* `search_error` ×2 (mask mismatch in a reconstructed world) is a reconstruction-fidelity gap, left as is:
  the contract fail-safe handled it; making the worker skip such a world instead of failing the
  whole decision would be a follow-up.
* The event loop is blocked by up to 4.7 s per decision (single battle only; `--concurrent 1`).
* BS0 numbers are unchanged; offline `OfflineMatch`/`Checkpoint` gained `learner_swap` and
  `learner_names` (default off ⇒ identical behaviour for all earlier stages).
