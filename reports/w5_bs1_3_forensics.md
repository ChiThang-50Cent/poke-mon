# W5.BS1.3-P — Forensics of `battle exceeded max_rounds`

Follows `reports/w5_bs1_3_incident.md`. Investigation only: no code change; BS1.3 gameplay stays
sealed (no win/loss read). Artifacts: `runs/w5-bs1-3-p/` (gitignored; inputs verified against the
incident-report sha256).

## Cause: harness stall from a reconstruction defect (class c)

In one sampled world the reconstructed battle is left with Showdown `battle.midTurn = true`.
The first rollout turn then runs without `beforeTurn`/`residual`; when the learner's active mon is
KO'd before it moves, `runAction` returns early, `checkFainted` is skipped, no `switchFlag` is
set and no forced-switch request is sent. The learner receives an `active` request for a fainted
mon, its choices are rejected (`[Invalid choice] Can't move: You sent more choices than unfainted
Pokémon`, then `You need a switch response` / `Sorry, too late to make a different move`), and the
sim waits on p1 (rqid 6) with nothing pending. Rounds 4–599 carry no events and no submissions:
**turn stays 10 after 600 delivery rounds** (normal rollouts: ~1 round per turn).

Trigger in this payload: world 3 sampled **Eject Pack on the opponent's Deoxys**, which is team
slot 0 and therefore the simulated lead (the observed lead was Ting-Lu). Changing that item to
Leftovers in a copy of the spec flips `midTurn` true → false. Which reconstruction step sets
`midTurn` with Eject Pack present is not yet traced. The missing `residual` also means the first
rollout turn of that world silently skips residual effects even when it does not stall.

## Reproduction

| Item | Result |
|---|---|
| Decision replayed | 64 (world, seed) pairs × candidates {6, 0, 4, 9} = 256 rollouts, same derivation as `_decide_chunk` |
| Deterministic | yes |
| Failing rollouts | 8 / 256: all world 3, candidates 0 and 9; (seed, opponent model) = (1, Grunt), (2, GymLeader), (4, GymLeader), (5, Grunt) |
| Failing rollout duration | 0.05–0.07 s (not a long game) |
| Other rollouts | all finish; ≤ 36 rounds, ≤ 36 turns; world 3 with candidates 4/6 completes normally |
| Ruled out | (a) genuinely long game: 10 turns in 600 rounds; (b) non-closing strategies: no repeating choice cycle |

## Root fidelity (payload only, no hidden truth)

8 worlds × 2 opponent models: learner request, 13-slot mask and opponent public active / fainted /
status match the payload in 16 / 16. Anomaly: `midTurn = true` in world 3 (both models), false in
the other 14. Open: the comparison flags one opponent mon in all 16 worlds; not yet investigated
and possibly a real mismatch.

## Frequency on non-heldout roots

Fixed pairs (8 worlds × 1 seed) on 220 / 240 BS0 + BS1.0 golden roots (20 unsupported at
reconstruction): 1,760 worlds, 6,864 rollouts. **0 hit max_rounds, 0 exceptions, 0 / 1,760 worlds
with `midTurn = true`.** Rounds per rollout p50 20 / p95 50 / p99 87 / max 166; turns p50 22 /
p95 63 / p99 111 / max 205 (mean 2.04 turns per round; a round can resolve more than one turn).

## Unknown

1. The reconstruction step that leaves `midTurn = true` with an Eject Pack lead, and other items or
   abilities that can do the same (switch-triggering effects during reconstruction).
2. The symmetric case (opponent KO'd before moving) is expected to stall the opponent bot the same
   way; not tested.
3. The opponent-mon flag in the public fidelity comparison (all 16 worlds).
