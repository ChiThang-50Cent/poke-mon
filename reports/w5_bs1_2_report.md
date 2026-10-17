# W5.BS1.2 — locked paired dev benchmark: C vs C+search (2 lanes × 4 workers)

DEV benchmark on the local Showdown server (:8000), dev pool only, heldout untouched, C
(`models/pokedecision-v1-value`) unchanged and still production, no training, no commits. Showdown's
RNG is not seedable: the design is paired by team and side, not by RNG. **No promotion claim.**
JSON: `reports/w5_bs1_2_{calibration,benchmark}.json`. Runs: `runs/w5-bs1-2-calibration`,
`runs/w5-bs1-2-b1..b4` (lane0/lane1 each), `runs/w5-bs1-2-projections.jsonl`,
`runs/w5-bs1-2-resources.txt`, `runs/w5-bs1-2-config.sha256`
(`e0d6af36…f26db`, `configs/w5_bs1_2_benchmark.yaml`). Code: `scripts/bs1_2_run.py` (driver),
`scripts/bs1_2_report.py`, `evaluate_w5.py --lane/--lanes/--pairs`, `runner.lane_indices`.

## Step 0 — calibration (seed 20261199, 4 C+search battles: Grunt ×2 in lane A, GymLeader ×2 in lane B)
p50 complete pairs 20 (pilot 20, ratio 1.00 ≥ 0.8); ≥ 4-pair coverage 91.6 % (≥ 85 %); 146 decisions, 0
over 5.0 s (max 4.704 s); peak scope memory 11 GB (≤ 12 GB); 0 errors. **Rule passed ⇒ 2 lanes × 4 workers.**

## Benchmark setup (as locked)
Arms C and C+search exactly as the pilot (D = 5.0 s, M = 0.3 s, `opp_model=mixture`, preview checkpoint,
argmax, search battle timeout 3600 s). Opponents Grunt, GymLeader, C greedy; schedule seed 20261201
(+ opponent index), 100 battles / arm / opponent = 50 team pairs, each pair in both directions, both arms,
alternating challenger. Blocks by complete pairs 12 / 13 / 12 / 13 (all four ran; each block's two lanes
take alternate pairs). Wrapper: one systemd user service, CPUQuota 1000 %, MemoryMax 14 G, nice 10,
OMP/MKL threads 1, CUBLAS_WORKSPACE_CONFIG=:4096:8.

Time projections (observed throughput, budget 16 h = 57,600 s): after block 1 17,588 s (4.9 h); block 2
20,855 s; block 3 21,506 s; block 4 actual 22,045 s. Block walls 4,221 / 6,206 / 5,487 / 6,131 s.
Status **COMPLETE** (4 / 4 blocks, 600 battles, 300 with search).

## Results (win = player win; no ties occurred)
Wins / 100: Grunt C 64, C+search 96; GymLeader C 65, C+search 95; C-vs-C C 54, C+search 72.

| contrast (C+search − C) | Δ | 95 % cluster bootstrap |
|---|---|---|
| **PRIMARY**, pooled over 3 opponents (150 pair clusters, stratified by opponent, 10,000 resamples) | **+0.267** | **[0.210, 0.323]** |
| Grunt + GymLeader pooled (stratified) | +0.310 | [0.245, 0.375] |
| Grunt | +0.320 | [0.22, 0.41] |
| GymLeader | +0.300 | [0.21, 0.39] |
| C greedy (neural stress test) | +0.180 | [0.07, 0.29] |

By learner seat (descriptive, 150 battles each): p1 C 95 → search 136 (+0.273); p2 C 88 → 127 (+0.260).
Pilot (BS1.1b) pooled was +0.383 on n = 20 per cell; the benchmark is the better estimate.

Gates: **B1 PASS** (0 timeouts, 0 model-illegal, 0 battle errors, 0 protocol retries in the 600 battles);
**B2 PASS** (0 of 11,229 decisions over 5.0 s); **B3 PASS** (pooled CI excludes 0); **B4 PASS** (no CI
below 0; every opponent CI is positive, including C-vs-C); **B5 PASS** (config hash recorded, 12/13/12/13
pairs per opponent complete in both directions and arms, all blocks and errors kept). Interpretation:
positive in all three cells including the neural opponent, so the help claim holds for this dev pool,
the two heuristics and greedy C; not a ladder claim.

## Engineering
* Wall per battle p50 / p95 / max: C 0.294 / 0.904 / 2.91 s; C+search 111.1 / 304.6 / 619.9 s (mean 137.6).
  Decisions per battle (mean / max): C 39.8 / 204; C+search 37.4 / 158.
* Latency (11,229 search-arm decisions): request→send p50 / p95 / max 4.511 / 4.702 / 4.714 s;
  search phase p50 4.510 / max 4.702; C inference + state p50 0.0036 s; search end→send max 0.013 s.
* CPU: 46 h 40 min (46.7 CPU-h) over 6 h 07 min wall; peak scope memory 11.2 G (limit 14 G);
  worker restarts max 0.
* Categories: fallback_no_signal 7,108; fallback_unsupported 2,178; override 1,782; single legal 142;
  not_represented 11; search_error 8. Override rate among searched decisions 16.1 % (searched 11,068).
  Complete pairs p50 17 / p95 64 / mean 22.0; rollouts p50 73 / p95 256; ≥ 4 pairs on 79.7 % of searched
  decisions (pilot 89.7 %, calibration 91.6 %); reconstruction support 80.3 % (pilot 92.7 %); hard deadline
  fired on 1,284 decisions; illegal actions in rollouts 0; protocol retries inside rollouts 27.
  Coverage is lower than the pilot because the benchmark is a different team sample with more Court Change
  (434), Wish (656) and Future Sight (381) decisions. Other unsupported reasons: pivot_switch_request 546,
  lockedmove/twoturn/outrage ~100, Doom Desire 18+, Lunar Dance 21, screen-duration 15, attract 3.
* Override rate by opponent: Grunt 18.6 %, GymLeader 15.8 %, C 24.6 %; by phase: turn < 10 25.7 %,
  10–24 18.8 %, 25+ 15.6 %; opponent × phase (turn<10 / 10–24 / 25+): Grunt 24.1 / 16.1 / 13.4 %, GymLeader
  26.2 / 12.4 / 7.8 %, C 26.9 / 27.0 / 20.8 %.
* Battles with ≥ 1 override: 283 (246 won); none: 17 (17 won). Descriptive, selection-confounded, not causal.
* State integrity: 0 forward-count violations, 0 hidden-digest changes, 0 cross-battle state.

## Search exceptions (8 of 11,229 decisions; all fell back to C's own legal action in time)
* 7 × `RuntimeError: forced action N outside mask` (a candidate legal in the live mask is illegal at some
  reconstructed world's root): N = 3 ×3, 2 ×2, 1, 0; in GymLeader / C battles.
* **1 × new error class** (C vs C battle, turn 2): metamon `CalledForeignConsecutive` while a world's sim
  replayed `Pawmot Fire Spin … [from] move: Metronome` (Metronome calling a multi-turn move). It occurred
  inside a reconstructed world (worker), the live battle was unaffected and completed. The contract says a
  new error class has priority over completing N; the run had already finished when I classified the 8
  errors (errors are visible only in the final assemble), so I report it instead of stopping. Not a gate
  failure (B1 counts battle errors; search exceptions are fail-safe by design).
* Regression test added before running: `tests/test_bs1_1_live.py::
  test_candidate_legal_live_but_outside_reconstructed_mask_falls_back_to_c` (stub pool raising the real
  message ⇒ C's legal action, category `search_error`, `error_cause`).

## Team-pair overlap with BS1.1 pilot runs (schedule seeds recomputed; per opponent Grunt / GymLeader / C)
`w5-bs1-1a` 0/0/0, `w5-bs1-1b-aborted1` 0/0/0, `w5-bs1-1b-block1` 0/0/0, `w5-bs1-1b-block2` 0/0/1:
one team pair of the 150 (C-vs-C cell) coincides with a pilot block-2 pair; the seed was locked first
and was kept. Individual teams are reused heavily (200-team pool).

## Failed / aborted runs and deviations
* No failed or aborted BS1.2 run. (Pilot aborted run `w5-bs1-1b-aborted1` is only referenced for overlap.)
* Calibration battles are not part of the benchmark.
* Resources: a systemd user *service* (not `--scope`) with identical CPU/memory properties, for a
  6-hour unattended job; calibration used `--scope`.
* The calibration peak memory (11 GB) is close to the 12 GB rule; the benchmark peak was 11.2 GB, within 14 G.
* Lane/pair-window plumbing is new code: `RunConfig.lane` and `pair_window` (default None ⇒ identical
  behaviour and config bytes), merged per block by re-indexing and validating with
  `cluster_differences`. Tests: `tests/test_bs1_2_lanes.py`.
* Win rates were first computed after all four blocks were final. During the block-1 test of the
  assembler I ran `final` but did not print its output (file deleted, gates not shown).
* The diagnostic "override vs none" and seat contrasts are descriptive.
