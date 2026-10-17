# W5.BS1.0 — hard deadline + opponent-model robustness (offline, pre-registered)

All numbers are **simulator value with a known real-opponent policy, not Showdown win rate**.
C was frozen; nothing was trained; collect pool only; no live battles; no dev pilot.
Gates: **M1 PASS, M2 PASS (pooled mixture Δ = +0.0348, CI [0.0105, 0.0617]), M3 PASS, M4 PASS.**
JSON: `reports/w5_bs1_0_{deadline,matrix}.json`; per-root data `runs/w5-bs1-0-{deadline,matrix}/`
(`decisions.jsonl`, `payload_meta.json`). Code: `src/pokeai/cf/recon_search.py`,
`scripts/bs1_0.py`, `scripts/bs1_0_report.py`, `tests/test_bs1_deadline.py`.

## Step 1 — hard deadline (M4)

Implementation (`SearchPool.decision`): the clock starts at the request (`t_request`) and covers
pickling, reconstruction, sampling, rollouts and finalisation. Workers stop scheduling at
B − 0.3 s = 4.7 s and stream every complete pair as it finishes; at B = 5.0 s the pool returns
the pairs received so far (else C's action) without waiting for in-flight workers. Every message
carries a decision id; replies of any other decision are dropped. `settle()` (untimed, between
decisions) waits for idle workers and kills/respawns one that does not return. The previous
soft deadline (waiting for all workers, `idle` handshake inside the wall) is removed.

Tests (`tests/test_bs1_deadline.py`, fake workers, no simulator): a worker that takes 2.5 s on
one pair → decision returns ≤ B + 50 ms, the pair is absent; the next decision contains only its
own results (late result ignored); `settle` recycles the straggler and the next decision uses all
4 workers; all workers slow → 0 pairs, legal C action, wall ≤ B; a request clock that started
0.4 s earlier is honoured; mixture draw paired/seeded/fair; `Payload` has no real-opponent field.

Re-measurement, 100 fresh roots (seed 20261700, 50 GymLeader / 50 Grunt real; search model
mixture; B = 5.0 s hard; wall measured outside the pool call):

| metric | value |
|---|---|
| wall p50 / p95 / max | 4.829 / 4.999 / **5.0015 s** |
| decisions > 5.05 s | 0 (hard deadline hit on 5 decisions; 0 workers recycled) |
| complete pairs p50 / p95 | 28.5 / 64 (rollouts p50 119) |
| roots ≥ 4 pairs | 91 % of all roots; **100 % (91/91) of supported roots** |
| unsupported roots | 9 (pivot 3, wish 2, two-turn 2, Future Sight 2) |
| override / fallback | 31 % / 69 % (no positive Δ 44, SNR 16, few pairs 9 = the 9 unsupported) |
| fallback legal C action | always; illegal actions 0; unexpected errors 0 |

**M4 PASS.**

## Step 2 — opponent model as a search parameter

`Payload.opp_model ∈ {grunt, gymleader, mixture}` replaces the old `Payload.opponent` (which was
the real identity — an oracle path in BS0; removed). Mixture (locked): per (world, seed) pair a
fair coin from `Random(f"{seed}:{root}:mix:{world}:{k}")`, the same for every candidate of the
pair; the world's opponent Python state is built for the drawn heuristic. The search never sees
the real opponent; `scripts/bs1_0.py` uses the real identity only to pick the true policy for the
oracle rollouts and to label rows (test: `Payload` has no `opponent` field; the BS0 oracle-import
AST test still passes).

## Step 3 — matrix (240 fresh roots, seed 20261800, 80 Grunt / 80 GymLeader / 80 C real; one
root per battle; B = 5.0 s hard for every decision; team pairs exclude all CF/BS0 roots)

Evaluation: true-snapshot rollouts with the REAL opponent policy, 16 fresh seeds, a_search vs
a_C paired (0 when equal), C-greedy continuation; unsupported roots fall back to C ⇒ Δ = 0 and are
included in the primary; 5,000-resample team-pair cluster bootstrap. **Real-opponent C**: its Python
state in evaluation is its real game state restored from the true snapshot (valid by the CF0.1
restore proof) and it samples its actions; the search never uses C as a model, so no
sampled-world GRU exists. Nothing about opponent C's state was found invalid (no blocker).

| real | search model | role | roots | Δ | 95 % CI | supported-only Δ [CI] | override | Δ / override (+/−) |
|---|---|---|---|---|---|---|---|---|
| Grunt | grunt | control | 80 | +0.0375 | [0.0062, 0.0734] | +0.0400 [0.0058, 0.0775] (75) | 36 % | +0.103 (14/3) |
| Grunt | gymleader | mismatch | 80 | +0.0094 | [−0.0281, 0.0430] | +0.0100 [−0.0300, 0.0458] | 31 % | +0.030 (10/6) |
| GymLeader | gymleader | control | 80 | +0.0570 | [0.0180, 0.1023] | +0.0652 [0.0223, 0.1170] (70) | 29 % | +0.198 (14/2) |
| GymLeader | grunt | mismatch | 80 | +0.0555 | [0.0180, 0.1000] | +0.0634 [0.0214, 0.1143] | 33 % | +0.171 (17/3) |
| Grunt | mixture | — | 80 | +0.0250 | [−0.0063, 0.0609] | +0.0267 [−0.0067, 0.0642] | 39 % | +0.065 (12/7) |
| GymLeader | mixture | — | 80 | +0.0445 | [0.0078, 0.0883] | +0.0509 [0.0089, 0.1018] | 29 % | +0.155 (14/3) |
| **Grunt+GymLeader** | **mixture** | **PRIMARY** | **160** | **+0.0348** | **[0.0105, 0.0617]** | +0.0384 [0.0108, 0.0698] (145) | 33.8 % | +0.103 |
| C | mixture | neural stress | 80 | +0.0281 | [0.0086, 0.0516] | +0.0326 [0.0100, 0.0589] (69) | 31 % | +0.090 (16/3) |

Fallback rate = 1 − override rate. Unsupported roots: pooled primary 15/160 (pivot 8, two-turn 2,
Future Sight 4, lockedmove 1); real C 11/80; matrix wall max 5.0023 s.
Pooled-primary by phase (mean Δ, roots): 0-5 +0.039 (32), 6-15 +0.019 (78), 16-30 +0.061 (46),
31+ 0.000 (4; too thin to read). Per-cell phase and candidate-count tables are in the JSON.

Matched − mismatched (paired per root, cluster CI): Grunt +0.0281 [−0.0039, 0.0664];
GymLeader +0.0016 [−0.0188, 0.0234]. Mixture − matched: Grunt −0.0125 [−0.0258, −0.0008],
GymLeader −0.0125 [−0.0375, 0.0078]. Mixture − mismatched: Grunt +0.0156 [−0.0164, 0.0547],
GymLeader −0.0109 [−0.0242, 0.0000]. So the opponent-model mismatch effect is not detectable at
n = 80 per real opponent (both CIs contain 0), and mixture is roughly equal to the matched control
within noise (marginally lower vs Grunt).

## Gates

* **M1 PASS**: 0 unexpected errors, 0 illegal actions (search-induced or chosen), fallback always a
  legal C action, oracle-free tests pass. One documented hidden-trap protocol retry (root 149, real
  C: "Can't switch: The active Pokémon is trapped", the CF0/CF1 `TRAP_PREFIXES` class) is counted
  separately as a protocol retry, not an unexpected error.
* **M2 PASS**: pooled mixture, 160 roots, unsupported = 0: Δ = +0.0348, CI [0.0105, 0.0617].
* **M3 PASS**: neither main group has mixture Δ < 0 with CI excluding 0 (Grunt +0.025
  inconclusive, GymLeader +0.0445 positive). Real-C cell +0.0281 [0.0086, 0.0516] is positive
  (reported, not gated); there is no clear negative result against a neural opponent.
* **M4 PASS** (Step 1).

## Deviations / caveats

* The matrix roots are random eligible decisions (not phase-stratified): 31+ has 4 pooled roots.
* a_C and the candidate list come from the normal learner pass on the root (not charged to the
  budget, as in BS0; identical in every belief world).
* The matrix ran in two invocations (the first hit the tool's 3,600 s limit after 541 of 560
  decisions; the resumable runner finished the rest from the cached payloads); no decision was
  repeated.
* Decisions finish early when all 64 pairs are done (p50 wall < 5 s); the hard deadline fired on
  5 of 100 latency decisions.
* Worker GC is off during a decision and collected between decisions (untimed `settle`).
* A search-side hidden-trap error is now counted as a protocol retry in `recon_search`; the
  BS0 numbers are unchanged (BS0 reports 0 such events).
* BS0 `bs0_search.py` now passes `opp_model` (= real opponent, lower-cased) because the Payload
  field changed; BS0 results were produced before the change and are not re-run.
* Pair CIs cluster on team pair (160 distinct pairs in the primary); roots with 4 candidates
  dominate (the rule is unchanged from BS0).
