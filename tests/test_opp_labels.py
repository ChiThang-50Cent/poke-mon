"""W6-A label join: same-turn opponent action as target, never shifted by a turn."""

import pytest

from pokeai.data.opp_labels import (
    AlignmentError,
    decode_action,
    pair_labels,
    preview_name,
)


def move(name):
    return {"name": name, "move_type": "normal", "category": "physical", "base_power": 80,
            "accuracy": 1.0, "priority": 0, "current_pp": 16, "max_pp": 16}


def mon(name, moves=()):
    stats = {f"base_{s}": 80 for s in ("atk", "spa", "def", "spd", "spe", "hp")}
    boosts = {f"{s}_boost": 0 for s in ("atk", "spa", "def", "spd", "spe", "accuracy",
                                         "evasion")}
    return {"name": name, "hp_pct": 1.0, "types": "normal notype", "item": "noitem",
            "ability": "noability", "lvl": 100, "status": "nostatus", "effect": "noeffect",
            "moves": [move(m) for m in moves], **boosts, **stats, "tera_type": "notype",
            "base_species": name}


def state(me, them, *, moves=(), bench=(), forced=False, opp_prev="nomove",
          preview=("rotom", "garchomp", "kingambit")):
    return {"format": "gen9ou", "player_active_pokemon": mon(me, moves),
            "opponent_active_pokemon": mon(them), "available_switches": [mon(b) for b in bench],
            "player_prev_move": move("nomove"), "opponent_prev_move": move(opp_prev),
            "opponents_remaining": 6, "player_conditions": "noconditions",
            "opponent_conditions": "noconditions", "weather": "noweather",
            "battle_field": "nofield", "forced_switch": forced, "battle_won": False,
            "battle_lost": False, "can_tera": True, "opponent_teampreview": list(preview)}


def test_decode_uses_metamon_sorted_slots_not_stored_order():
    s = state("garchomp", "x", moves=("swordsdance", "earthquake", "scaleshot", "fireblast"),
              bench=("rotomwash", "kingambit"))
    assert decode_action(s, 0) == ("move", "earthquake", False)
    assert decode_action(s, 10) == ("move", "fireblast", True)
    assert decode_action(s, 4) == ("switch", "kingambit", False)
    assert decode_action(s, -1) is None
    with pytest.raises(AlignmentError):
        decode_action(s, 8)


def test_preview_maps_formes_to_public_entry():
    assert preview_name("rotomwash", ["rotom", "garchomp"]) == "rotom"
    assert preview_name("garchomp", ["rotom", "garchomp"]) == "garchomp"
    assert preview_name("zapdos", ["rotom"]) is None


def battle():
    """Learner (garchomp side) has one forced-switch subturn that the opponent lacks."""
    learner = {"states": [
        state("garchomp", "rotomwash", moves=("earthquake",)),
        state("garchomp", "rotomwash", moves=("earthquake",), opp_prev="voltswitch"),
        state("kingambit", "rotomwash", forced=True),
        state("kingambit", "rotomwash", moves=("suckerpunch",), opp_prev="hydropump"),
        state("kingambit", "garchomp", moves=("suckerpunch",)),
    ], "actions": [0, 0, 4, 0, -1]}
    opponent = {"states": [
        state("rotomwash", "garchomp", moves=("hydropump", "voltswitch")),
        state("rotomwash", "garchomp", moves=("hydropump", "voltswitch")),
        state("rotomwash", "kingambit", moves=("hydropump", "voltswitch"), bench=("garchomp",)),
        state("garchomp", "kingambit"),
    ], "actions": [1, 0, 4, -1]}
    return learner, opponent


def test_label_is_same_turn_opponent_action_and_forced_rows_unlabelled():
    labels, qa = pair_labels(*battle())
    # row 0 -> opponent turn-1 voltswitch (seen in learner row 1's prev move, not row 0's)
    assert labels == [("move:voltswitch", 0), ("move:hydropump", 0), None,
                      ("switch:garchomp", -1)]
    assert qa["move_eq_next_prev_move"] == 1 and qa["move_eq_current_prev_move"] == 0


def test_misaligned_battles_are_rejected():
    learner, opponent = battle()
    opponent["states"][1]["player_active_pokemon"]["name"] = "zapdos"
    with pytest.raises(AlignmentError, match="active"):
        pair_labels(learner, opponent)
    learner, opponent = battle()
    opponent["states"].insert(0, opponent["states"][0])
    opponent["actions"].insert(0, 0)
    with pytest.raises(AlignmentError, match="non-forced"):
        pair_labels(learner, opponent)
