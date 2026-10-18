"""BS0: belief worlds from the learner's own information only (oracle-free path).

Inputs are exactly what the learner holds at a decision: its packed team, its pending
request, the public protocol lines it has received (history + the not-yet-processed frame
that arrives with the request), public Dex data, and the CF1 belief sampler.
This module (and `reconstruct.py`) MUST NOT import `recon_oracle` or read the true state
(`Battle.toJSON`, `p2_full`, `public_log`, `public_view`, the true p2 team); a test enforces it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pokeai.cf.belief import BeliefSampler, annotate_categories, parse_knowledge
from pokeai.cf.offline import PLAYER_NAMES, OfflineMatch, new_tag, restore_player
from pokeai.cf.reconstruct import (
    Unsupported,
    build_public,
    build_spec,
    p2_team_sets,
    reveal_order,
    roster_for,
)
from pokeai.cf.sim import Sim, gen5_seed


@dataclass
class LearnerInfo:
    lines: list[str]  # learner's p1-channel protocol lines, in order, incl. pending frame
    request: dict[str, Any]  # learner's pending request (parsed JSON)
    own_team: str  # packed team
    pending_p1_lines: list[str] = field(default_factory=list)
    # True: lines/request above are in the canonical seat (learner = p1) but the learner's real
    # python state sits at p2; deliveries to it are translated back (offline.swap_roles).
    swapped: bool = False
    # sim account name -> the live account name known to the learner's python battle
    names: dict[str, str] | None = None


@dataclass
class Recon:
    tag: str
    spec: dict[str, Any]
    world: dict[str, Any]
    events: list[dict[str, Any]]
    p1_request: dict[str, Any]
    p2_request: dict[str, Any]
    timing: dict[str, float]


def sample_world(
    sim: Sim, sampler: BeliefSampler, info: LearnerInfo, rng: np.random.Generator
) -> tuple[Any, Any, list[Any], dict[str, Any]]:
    public, tracker = build_public(info.lines)
    order2 = reveal_order(public, "p2")
    roster = roster_for(order2)
    know = parse_knowledge(info.lines, roster)
    annotate_categories(know, sim)
    world = sampler.sample_world(roster, know, rng)
    return tracker, order2, know, world


class WorldRejectedState(Unsupported):
    """The reconstructed engine state is not the decision boundary of the live request."""


def world_invariant_violations(
    inv: dict[str, Any], request: dict[str, Any], spec: dict[str, Any]
) -> list[str]:
    """Layer 2: engine invariants of a reconstructed root versus the live request.

    Normal move root (verified against a genuine battle): turn open at its start (`midTurn`
    false), empty action queue, nothing pending, a move request on both sides, live active mons.
    Forced-switch root: the turn's residual already ran (`midTurn` true, empty queue), the
    fainted active of each switch side carries the switch flag and the forceSwitch request.
    """
    forced = "forceSwitch" in request
    switch_sides = set(spec["switch_sides"]) if forced else set()
    bad: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            bad.append(what)

    check(not inv["ended"], "ended")
    check(inv["midTurn"] is forced, f"midTurn={inv['midTurn']} (expected {forced})")
    check(inv["queue"] == 0, f"queue={inv['queue']}")
    check(inv["faintQueue"] == 0, f"faintQueue={inv['faintQueue']}")
    check(inv["requestState"] == ("switch" if forced else "move"),
          f"requestState={inv['requestState']!r}")
    for side in inv["sides"]:
        sid = side["id"]
        flagged = sid in switch_sides
        want = ["forceSwitch"] if flagged else ["wait"] if forced else ["active"]
        check(side["request"] == want, f"{sid} request {side['request']} (expected {want})")
        check(side["switchFlags"] == int(flagged), f"{sid} switchFlags={side['switchFlags']}")
        check(side["activeFainted"] is flagged, f"{sid} activeFainted={side['activeFainted']}")
        check(side["faintedThisTurn"] is flagged, f"{sid} faintedThisTurn")
        check(side["pokemonLeft"] == side["aliveCount"], f"{sid} pokemonLeft/alive")
        check(not flagged or side["pokemonLeft"] >= 1, f"{sid} no replacement")
        check(not side["choiceError"], f"{sid} choiceError")
        for mon in side["active"]:
            check(mon is not None and mon["isActive"] is not flagged, f"{sid} active mon state")
    return bad


def reconstruct_world(
    sim: Sim, sampler: BeliefSampler, info: LearnerInfo, world_seed: int, battle_seed: int,
    tag: str | None = None,
) -> Recon:
    """One belief world: raises `Unsupported` (with a reason) outside the supported subset."""
    rng = np.random.default_rng([world_seed, 11])
    t0 = time.perf_counter()
    tracker, order2, know, world = sample_world(sim, sampler, info, rng)
    t1 = time.perf_counter()
    spec = build_spec(
        tracker, info.request, info.own_team, world, know, np.random.default_rng(
            [world_seed, 13]), gen5_seed(battle_seed), PLAYER_NAMES["p1"],
        PLAYER_NAMES["p2"], p2_sets=p2_team_sets(order2, world),
    )
    tag = tag or new_tag()
    events = sim.reconstruct(tag, spec)
    violations = world_invariant_violations(sim.invariants(tag), info.request, spec)
    if violations:
        sim.destroy(tag)
        raise WorldRejectedState("; ".join(violations))
    t2 = time.perf_counter()
    import json

    reqs: dict[str, Any] = {}
    for ev in events:
        if ev["t"] == "sideupdate" and ev["data"].startswith("|request|"):
            reqs[ev["side"]] = json.loads(ev["data"][len("|request|"):])
    return Recon(
        tag=tag, spec=spec, world=world, events=events, p1_request=reqs["p1"],
        p2_request=reqs["p2"], timing={"sample_s": t1 - t0, "reconstruct_s": t2 - t1},
    )


def learner_probe(
    sim: Sim, players: dict[str, Any], learner_snapshot: dict[str, Any], old_tag: str,
    info: LearnerInfo, recon: Recon,
) -> dict[str, Any] | None:
    """Run the learner's own policy on the reconstructed request; returns its model record.

    Only the learner's restored python state (its real battle + hidden memory) and its
    request enter; nothing from the true simulator.
    """
    # probe on a copy: the probe submits the learner's choice, the world must stay pristine
    tag = new_tag()
    sim.restore(sim.snapshot(recon.tag), tag)
    match = OfflineMatch(sim, players, tag, record_probs=True)
    restore_player(players["p1"], learner_snapshot, old_tag, tag)
    for side in ("p1", "p2"):
        match.seed_side(side, 0)
    match._bind()
    match.pending = [
        {"t": "update", "p1": list(info.pending_p1_lines), "p2": [], "omni": []},
        {"t": "sideupdate", "side": "p1", "data": "|request|" + __import__("json").dumps(
            recon.p1_request, separators=(",", ":"), ensure_ascii=False)},
    ]
    players["p1"].describe = False
    try:
        match.step()
        model = next((d["model"] for d in match.rounds[0].decisions["p1"]
                      if d["model"] is not None), None)
    finally:
        from pokeai.cf.offline import discard_player_tag

        discard_player_tag(players["p1"], tag)
        sim.destroy(tag)
    return model


__all__ = [
    "LearnerInfo", "Recon", "Unsupported", "WorldRejectedState", "learner_probe",
    "reconstruct_world", "world_invariant_violations",
]
