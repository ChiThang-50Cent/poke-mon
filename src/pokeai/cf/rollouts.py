"""CF0.3: one random learner decision per fresh battle, K paired rollouts per candidate."""

from __future__ import annotations

import random
from typing import Any

from pokeai.cf.offline import OfflineMatch, PlayerSpec, make_player, new_tag
from pokeai.cf.restore_check import (
    C_POLICY,
    PREVIEW,
    TakenCheckpoint,
    pending_request,
    upcoming_turn,
)
from pokeai.cf.sim import Sim


class _Stop(Exception):
    pass


def phase_bucket(turn: int) -> str:
    if turn <= 5:
        return "0-5"
    if turn <= 15:
        return "6-15"
    if turn <= 30:
        return "16-30"
    return "31+"


def make_players(opponent: str) -> dict[str, Any]:
    c_spec = PlayerSpec(C_POLICY, sample=True, temperature=1.0, preview=PREVIEW)
    if opponent == "C":
        o_spec = PlayerSpec(C_POLICY, sample=True, temperature=1.0, preview=PREVIEW)
    else:
        o_spec = PlayerSpec(opponent)
    return {"p1": make_player(c_spec, "p1"), "p2": make_player(o_spec, "p2")}


def eligible_rounds(match: OfflineMatch) -> list[tuple[int, list[int]]]:
    """(round index, mask) of learner model decisions with >= 2 legal slots."""
    out = []
    for record in match.rounds:
        for d in record.decisions["p1"]:
            model = d["model"]
            if model is not None and sum(model["mask"]) >= 2:
                out.append((record.index, model["mask"]))
                break
    return out


def pick_root(
    sim: Sim, teams: dict[str, str], opponent: str, seeds: dict[str, int],
    rng: random.Random,
) -> dict[str, Any] | None:
    """Pass 1 plays the battle; pass 2 replays it (identical seeds) to the chosen round.

    Replay determinism is the CF0.1 result; the chosen round's mask is re-checked.
    """

    def play(stop_at: int | None) -> tuple[OfflineMatch, Any, dict[str, Any]]:
        players = make_players(opponent)
        match = OfflineMatch(sim, players, new_tag())
        match.start(seeds["sim"], teams, {"p1": seeds["p1"], "p2": seeds["p2"]})
        captured: dict[str, Any] = {}

        def hook(m: OfflineMatch) -> None:
            if stop_at is not None and m.round_index == stop_at:
                requests = pending_request(m.pending)
                captured["cp"] = m.checkpoint()
                captured["turn"] = upcoming_turn(m)
                captured["forced_switch"] = "forceSwitch" in requests.get("p1", {})
                raise _Stop

        try:
            match.play(hook=hook if stop_at is not None else None)
        except _Stop:
            pass
        return match, players, captured

    first, _, _ = play(None)
    candidates = eligible_rounds(first)
    errors = [e for r in first.rounds for e in r.errors]
    winner = first.winner()
    first.close()
    if not candidates:
        return None
    round_index, mask = candidates[rng.randrange(len(candidates))]
    match, players, captured = play(round_index)
    match.close()
    cp = captured["cp"]
    root = TakenCheckpoint(
        battle=cp.tag, reasons=["random"], checkpoint=cp, state_at={},
        turn=captured["turn"], round_index=round_index,
    )
    return {
        "root": root, "players": players, "mask_pass1": mask, "turn": captured["turn"],
        "forced_switch": captured["forced_switch"], "eligible_decisions": len(candidates),
        "pass1_errors": errors, "pass1_winner": winner,
    }


def choose_candidates(
    info: dict[str, Any], rng: random.Random
) -> list[int]:
    """Top-2 by probability, one uniformly random other legal slot, best legal Tera."""
    mask, probs = info["mask"], info["probs"]
    legal = [i for i, ok in enumerate(mask) if ok]
    ranked = sorted(legal, key=lambda i: -probs[i])
    picks = ranked[:2]
    others = [i for i in legal if i not in picks]
    if others:
        picks.append(rng.choice(others))
    tera = [i for i in ranked if i >= 9 and i not in picks]
    if tera:
        picks.append(tera[0])
    return picks


def outcome(winner: str | None) -> float:
    return 1.0 if winner == "p1" else 0.0 if winner == "p2" else 0.5
