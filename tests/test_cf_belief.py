"""Pure-Python tests for CF1 (no sim): knowledge parser, HP round trip, candidates, stats."""

from __future__ import annotations

import random

import numpy as np

from pokeai.cf import cf1_metrics as M
from pokeai.cf.belief import parse_knowledge
from pokeai.cf.cf1 import choose_candidates
from pokeai.cf.offline import hp_for_pct, public_pct, rewrite_p2_hp

ROSTER = [
    {"name": "Gholdengo", "legal_abilities": ["Good as Gold"]},
    {"name": "Garchomp", "legal_abilities": ["Rough Skin", "Sand Veil"]},
]


def test_hp_round_trip_same_public_percent() -> None:
    for old_max in (261, 301, 341, 404):
        for hp in range(old_max + 1):
            pct = public_pct(hp, old_max)
            for new_max in (233, 300, 351, 420):
                assert public_pct(hp_for_pct(pct, new_max), new_max) == pct


def test_rewrite_p2_hp_line() -> None:
    by_name = {"Garchomp": {"old_max": 200, "new_max": 300}}
    line = "|-damage|p2a: Garchomp|101/200 par|[from] item: Life Orb"
    want = "|-damage|p2a: Garchomp|153/300 par|[from] item: Life Orb"
    assert rewrite_p2_hp(line, by_name) == want
    other = "|-damage|p1a: Garchomp|101/200"
    assert rewrite_p2_hp(other, by_name) == other


def test_parse_knowledge_rules() -> None:
    lines = [
        "|switch|p2a: Garchomp|Garchomp, M|100/100",
        "|move|p2a: Garchomp|Earthquake|p1a: X",
        "|move|p2a: Garchomp|Swords Dance|p2a: Garchomp",
        "|-damage|p2a: Garchomp|90/100|[from] item: Rocky Helmet|[of] p2a: Garchomp",
        "|move|p2a: Gholdengo|Make It Rain|p1a: X|[from]move: Sleep Talk",
        "|-ability|p2a: Gholdengo|Good as Gold",
        "|-terastallize|p2a: Gholdengo|Steel",
        "|-enditem|p2a: Gholdengo|Leftovers|[from] move: Knock Off|[of] p1a: X",
    ]
    know = parse_knowledge(lines, ROSTER)
    garchomp, gholdengo = know["Garchomp"], know["Gholdengo"]
    assert garchomp.moves == ["Earthquake", "Swords Dance"]
    assert garchomp.choice_excluded and garchomp.item == "Rocky Helmet"
    assert gholdengo.moves == []  # called move is not an own move
    assert gholdengo.ability == "Good as Gold" and gholdengo.tera == "Steel"
    assert gholdengo.item_state == "gone"


def test_candidate_rule() -> None:
    mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    probs = [0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0, 0, 0.25, 0.1, 0, 0]
    out = choose_candidates({"mask": mask, "probs": probs}, random.Random(0))
    slots = [c["slot"] for c in out]
    assert slots[0] == 0 and slots[1] == 9 and len(slots) == len(set(slots)) <= 4
    assert all(mask[s] for s in slots)
    assert out[0]["reason"] == "c_greedy"


def test_select_delta_prefers_better_on_other_half() -> None:
    q = np.zeros((2, 16))
    q[1] = 1.0  # candidate 1 strictly better in both halves
    delta, moved = M.select_delta(q)
    assert delta == 1.0 and moved == 1.0
    q[1] = 0.0  # tie -> a_C
    assert M.select_delta(q) == (0.0, 0.0)


def test_cluster_bootstrap_and_split() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0])
    lo, hi = M.cluster_bootstrap(values, ["a", "a", "b", "b"], 200, 0)
    assert lo == hi == 1.0
    split = M.assign_split([f"p{i}" for i in range(10)] * 2, 1)
    assert sorted(split.values()).count("val") == 2
    assert M.assign_split([f"p{i}" for i in range(10)], 1) == split


def test_consumed_item_not_resurrected_by_later_item_effect() -> None:
    """Sitrus heal / Weakness Policy boost are logged AFTER the `-enditem` line."""
    lines = [
        "|switch|p2a: Garchomp|Garchomp, M|100/100",
        "|-enditem|p2a: Garchomp|Sitrus Berry|[eat]",
        "|-heal|p2a: Garchomp|100/100|[from] item: Sitrus Berry",
        "|-enditem|p2a: Gholdengo|Custap Berry|[eat]",
        "|-activate|p2a: Gholdengo|item: Custap Berry|consumed",
    ]
    know = parse_knowledge(lines, ROSTER)
    assert know["Garchomp"].item_state == "gone"
    assert know["Gholdengo"].item_state == "gone"
    # a genuinely re-acquired item (Harvest / Trick) still counts as known
    again = parse_knowledge(
        [*lines[:3], "|-item|p2a: Garchomp|Sitrus Berry|[from] ability: Harvest"], ROSTER
    )
    assert again["Garchomp"].item_state == "known"


def test_of_naming_attacker_is_not_the_ability_owner() -> None:
    """Water Absorb heal on a p1 mon logs `[of]` = the p2 attacker (not the owner)."""
    roster = [
        {"name": "Suicune", "legal_abilities": ["Pressure", "Inner Focus"]},
        {"name": "Garchomp", "legal_abilities": ["Rough Skin", "Sand Veil"]},
    ]
    absorb = "|-heal|p1a: Ogerpon|211/301|[from] ability: Water Absorb|[of] p2a: Suicune"
    assert parse_knowledge([absorb], roster)["Suicune"].ability is None
    # contact effect: `[of]` really is the owner and the dex allows it
    rough = "|-damage|p1a: X|90/100|[from] ability: Rough Skin|[of] p2a: Garchomp"
    assert parse_knowledge([rough], roster)["Garchomp"].ability == "Rough Skin"
    # same-side / no `[of]`: the subject owns the ability
    own = "|-heal|p2a: Suicune|100/100|[from] ability: Water Absorb"
    assert parse_knowledge([own], roster)["Suicune"].ability == "Water Absorb"
