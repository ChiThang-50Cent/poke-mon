"""BS0.2: a rollout-ready match root built from a reconstructed belief world.

Oracle-free (see `recon_world`). p1 = the learner's REAL python state (restored snapshot,
including its recurrent memory). p2 = a FRESH python player whose battle is built from the
opponent's point of view: the public history re-expressed for p2 (own side exact HP from the
reconstructed sets, opponent side percentages) followed by the reconstructed sim's own p2
request. The true p2 snapshot is never read.
"""

from __future__ import annotations

import re
from typing import Any

from pokeai.cf.offline import (
    _HP_FIELD,
    _HP_LINES,
    OfflineMatch,
    hp_for_pct,
    public_pct,
    restore_player,
)
from pokeai.cf.recon_world import LearnerInfo, Recon
from pokeai.cf.sim import Sim

_IDENT = re.compile(r"^(p[12])[a-z]?: (.*)$")


def _max_hp(request: dict[str, Any]) -> dict[str, int]:
    out = {}
    for entry in request["side"]["pokemon"]:
        name = entry["ident"].split(": ", 1)[1]
        out[name] = int(entry["condition"].split(" ")[0].split("/")[1]) if "/" in entry[
            "condition"] else 0
    return out


def pov_p2(lines: list[str], p1_max: dict[str, int], p2_max: dict[str, int]) -> list[str]:
    """The learner-channel (p1) protocol as p2 would have received it.

    HP fields of p1's own Pokemon are exact in the p1 channel and percentages for p2; the
    reverse holds for p2's Pokemon. Everything else is public and identical in both channels.
    """
    out = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in _HP_LINES:
            out.append(line)
            continue
        slots = [(2, 3)] if parts[1] != "-sethp" else [(2, 3), (4, 5)]
        if parts[1] in ("switch", "drag", "replace"):
            slots = [(2, 4)]
        elif parts[1] in ("-damage", "-heal"):
            slots = [(2, 3)]
        for ident_i, hp_i in slots:
            if hp_i >= len(parts):
                continue
            ident = _IDENT.match(parts[ident_i])
            match = _HP_FIELD.match(parts[hp_i])
            if not ident or not match:
                continue
            side, name = ident.groups()
            cur, mx, status = int(match.group(1)), int(match.group(2)), match.group(3)
            if side == "p1":  # exact for the learner -> percentage for p2
                parts[hp_i] = f"{public_pct(cur, mx)}/100{status}"
            else:  # percentage for the learner -> exact for p2
                new_max = p2_max.get(name)
                if new_max and mx == 100:
                    parts[hp_i] = f"{hp_for_pct(cur, new_max)}/{new_max}{status}"
        out.append("|".join(parts))
    return out


def build_world_root(
    sim: Sim, players: dict[str, Any], learner_snapshot: dict[str, Any], old_tag: str,
    info: LearnerInfo, recon: Recon, seeds: tuple[int, int] = (0, 0),
) -> Any:
    """Checkpoint of the reconstructed world just before the root requests are delivered.

    Both players receive the public history; no request is delivered, so nobody acts.
    """
    match = OfflineMatch(sim, players, recon.tag)
    match.learner_swap = info.swapped
    match.learner_names = info.names
    match.seed_side("p1", seeds[0])
    match.seed_side("p2", seeds[1])
    restore_player(players["p1"], learner_snapshot, old_tag, recon.tag)
    match._bind()
    p1_max, p2_max = _max_hp(recon.p1_request), _max_hp(recon.p2_request)
    history = info.lines[: len(info.lines) - len(info.pending_p1_lines)]
    p2_history = pov_p2(history, p1_max, p2_max)
    p2_pending = pov_p2(info.pending_p1_lines, p1_max, p2_max)
    requests = [
        ev for ev in recon.events if ev["t"] == "sideupdate" and ev["data"].startswith("|request|")
    ]
    match.pending = [
        {"t": "init", "p1": [], "p2": ["\n".join(p2_history) + "\n"]},
        {"t": "update", "p1": list(info.pending_p1_lines), "p2": p2_pending, "omni": []},
    ]
    from pokeai.cf.offline import _run

    async def begin() -> None:
        import random

        player = players["p2"]
        random.setstate(match.py_rng["p2"])
        player.randomize()
        match.py_rng["p2"] = random.getstate()
        player.last_server_error = None

    _run(begin())
    match.step()  # deliver history; players only act on requests
    match.pending = [{**ev} for ev in requests]
    match.rounds.clear()
    cp = match.checkpoint()
    match.close()
    return cp


def p2_pov_facts(cp: Any) -> dict[str, Any]:
    """What the opponent's python battle (inside a world checkpoint) knows about p1."""
    battle = cp.players["p2"]["battle"]
    return {
        "turn": battle.turn,
        "opp_active": getattr(battle.opponent_active_pokemon, "species", None),
        "opp": {
            m.species: {
                "moves": sorted(m.moves), "item": m.item,
                "hp_fraction": m.current_hp_fraction, "max_hp": m.max_hp,
                "status": m.status.name if m.status else None,
            }
            for m in battle.opponent_team.values()
        },
        "own_active": getattr(battle.active_pokemon, "species", None),
    }


__all__ = ["build_world_root", "p2_pov_facts", "pov_p2"]
