"""CF0.2 branching: restore a root decision point, force one learner action, roll out.

A root is a ``TakenCheckpoint`` taken BEFORE the pending request frames are delivered:
both sides hold a pending request and nothing has been chosen. For every branch the
root is restored (sim + per-tag player state), the sim PRNG is set to a branch-shared
seed without `resetRNG`, the opponent decides from ITS OWN channel with an RNG stream
seeded independently of the learner candidate, the learner's candidate replaces its
sampled action (the model still runs forward so recurrent memory advances exactly as in
play), and both sides then follow their policies to the terminal result.

Simultaneous moves: both players decide inside their own frame delivery, before either
order is submitted to the sim, so the learner never sees the opponent's choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pokeai.cf.offline import SIDES, OfflineMatch, json_digest, new_tag
from pokeai.cf.restore_check import TakenCheckpoint, signature
from pokeai.cf.sim import Sim, gen5_seed


def reseed(match: OfflineMatch, side: str, seed: int) -> None:
    match.seed_side(side, seed)
    match._bind()


def candidate_actions(
    sim: Sim, players: dict[str, Any], root: TakenCheckpoint, learner: str = "p1",
    max_candidates: int = 3,
) -> dict[str, Any] | None:
    """Mask, policy probabilities and the candidate slots of the learner at `root`.

    Candidates: the two most probable legal slots plus the most probable legal
    move+Tera slot (9-12) when one exists and is not already in the top two.
    Returns None when the learner has no represented model decision at the root
    (revival target, Struggle, forced pass, wait request) or fewer than 2 legal slots.
    """
    probe = OfflineMatch.restore(sim, players, root.checkpoint, new_tag(), record_probs=True)
    try:
        probe.step()
        model = next(
            (
                d["model"] for d in probe.rounds[0].decisions[learner]
                if d["model"] is not None
            ),
            None,
        )
    finally:
        probe.close()
    if model is None or sum(model["mask"]) < 2:
        return None
    legal = [i for i, ok in enumerate(model["mask"]) if ok]
    ranked = sorted(legal, key=lambda i: -model["probs"][i])
    picks = ranked[: max_candidates - 1]
    tera = [i for i in ranked if i >= 9 and i not in picks]
    if tera and max_candidates >= 3:
        picks.append(tera[0])
    else:
        picks = ranked[:max_candidates]
    return {
        "mask": model["mask"], "probs": model["probs"], "value": model["value"],
        "sampled_action": model["sampled_action"], "candidates": picks,
        "features": model["features"],
    }


@dataclass
class Rollout:
    candidate: int
    seeds: dict[str, int]
    winner: str | None
    rounds: int
    turns: int
    wall_s: float
    errors: list[str]
    illegal_actions: int
    final_prng: str
    sig_digest: str = ""
    signature: list[dict[str, Any]] = field(repr=False, default_factory=list)
    root_decisions: dict[str, Any] = field(default_factory=dict)


def rollout(
    sim: Sim, players: dict[str, Any], root: TakenCheckpoint, learner: str,
    candidate: int, seeds: dict[str, int], max_rounds: int = 600,
    world: dict[str, Any] | None = None, learner_sample: bool | None = None,
    keep_signature: bool = True,
) -> Rollout:
    """One branch: `seeds` = {"sim": .., "opponent": .., "learner": ..}.

    `world` (CF1) transplants a sampled opponent world into the restored root;
    `learner_sample` overrides the learner's sampling mode after the root restore
    (restore rewrites scalar player attributes from the snapshot).
    """
    opponent = "p2" if learner == "p1" else "p1"
    started = time.perf_counter()
    tag = new_tag()
    match = OfflineMatch.restore(sim, players, root.checkpoint, tag)
    try:
        if learner_sample is not None:
            players[learner].sample = learner_sample
        if world is not None:
            match.apply_world(world)
        sim.set_prng(tag, gen5_seed(seeds["sim"]))
        reseed(match, opponent, seeds["opponent"])
        reseed(match, learner, seeds["learner"])
        match.forced[learner].append(candidate)
        illegal_before = {
            s: getattr(players[s], "illegal_actions", 0) for s in SIDES
        }
        match.play(max_rounds=max_rounds)
        final_prng = sim.get_prng(tag)
        errors = [e for r in match.rounds for e in r.errors]
        sig = signature(match.rounds, tag)
        root_round = match.rounds[0]
        result = Rollout(
            candidate=candidate, seeds=seeds, winner=match.winner(),
            rounds=len(match.rounds), turns=match.turn,
            wall_s=time.perf_counter() - started, errors=errors,
            illegal_actions=sum(
                getattr(players[s], "illegal_actions", 0) - illegal_before[s]
                for s in SIDES
            ),
            final_prng=final_prng, sig_digest=json_digest(sig),
            signature=sig if keep_signature else [],
            root_decisions={
                s: [
                    {"message": d["message"], "model": d["model"]}
                    for d in root_round.decisions[s]
                ]
                for s in SIDES
            },
        )
    finally:
        match.close()
    return result


def root_consistency(
    a: list[dict[str, Any]], b: list[dict[str, Any]], learner: str
) -> dict[str, Any]:
    """Compare two branches (same root + seeds, different learner candidate).

    Divergence may only start AFTER the learner commits: the root round's input frames,
    the opponent's root decision and the learner's root observation/hidden must match.
    """
    opponent = "p2" if learner == "p1" else "p1"
    ra, rb = a[0], b[0]
    skip = {"action", "forced"}

    def view(decisions: list[dict[str, Any]]) -> list[Any]:
        return [
            None if d["model"] is None else {
                k: v for k, v in d["model"].items() if k not in skip
            }
            for d in decisions
        ]

    first_output = None
    for index, (xa, xb) in enumerate(zip(a, b, strict=False)):
        if xa["events_out"] != xb["events_out"] or (
            index > 0 and xa["delivered"] != xb["delivered"]
        ):
            first_output = index
            break
    return {
        "root_inputs_identical": ra["delivered"] == rb["delivered"]
        and ra["omni"] == rb["omni"],
        "opponent_root_decision_identical": ra["decisions"][opponent]
        == rb["decisions"][opponent],
        "learner_root_observation_identical": view(ra["decisions"][learner])
        == view(rb["decisions"][learner]),
        "learner_root_choice_differs": ra["submitted"] != rb["submitted"],
        "first_output_divergence_round": first_output,
    }
