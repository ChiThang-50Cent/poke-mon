"""Adversarial public-view checks: future knowledge may never become input."""

import copy
import json
from pathlib import Path

import lz4.frame
import pytest

from pokeai.data.validate import leakage_errors, read_native, validate_run


def trajectory(run: Path):
    entry = json.loads((run / "manifest.json").read_text())["trajectory_files"][0]
    return (
        entry,
        read_native(Path(entry["file"])),
        json.loads(Path(entry["meta"]).read_text()),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "future_move",
        "future_prev_move",
        "final_result",
        "reserve",
        "ability",
        "item",
        "tera",
        "terminal_flag",
    ],
)
def test_hidden_information_mutations_fail(synthetic_run, mutation):
    run, human = synthetic_run
    entry, data, meta = trajectory(run)
    state = data["states"][0]
    opponent = state["opponent_active_pokemon"]
    if mutation == "future_move":
        opponent["moves"] = copy.deepcopy(
            data["states"][1]["opponent_active_pokemon"]["moves"]
        )
    elif mutation == "future_prev_move":
        state["opponent_prev_move"] = copy.deepcopy(
            data["states"][1]["opponent_active_pokemon"]["moves"][0]
        )
    elif mutation == "final_result":
        state["final_result"] = "win"
    elif mutation == "reserve":
        state["opponent_reserves"] = ["mewtwo"]
    elif mutation == "ability":
        opponent["ability"] = "adaptability"  # Eevee has multiple abilities.
    elif mutation == "item":
        opponent["item"] = "leftovers"
    elif mutation == "tera":
        opponent["tera_type"] = "water"
    elif mutation == "terminal_flag":
        state["battle_won"] = True
    assert leakage_errors(data, meta)
    Path(entry["file"]).write_bytes(lz4.frame.compress(json.dumps(data).encode()))
    report = validate_run(run, human)
    assert report["overall_status"] == "FAIL"
    assert report["hidden_information_leakage_failures"] > 0


def test_unique_dex_ability_is_public(synthetic_run):
    run, _ = synthetic_run
    _, data, meta = trajectory(run)
    for state in data["states"]:
        state["opponent_active_pokemon"]["name"] = "greattusk"
        state["opponent_active_pokemon"]["ability"] = "protosynthesis"
    meta["public_events"][1][3] = "Great Tusk, L100"
    assert leakage_errors(data, meta) == []
    data["states"][0]["opponent_active_pokemon"]["ability"] = "levitate"
    assert leakage_errors(data, meta)


def test_reveal_after_decision_is_future(synthetic_run):
    run, _ = synthetic_run
    _, data, meta = trajectory(run)
    meta["decision_event_counts"][1] = 2
    assert any(
        "unrevealed opponent move" in error for error in leakage_errors(data, meta)
    )


@pytest.mark.parametrize(
    ("protocol", "expected", "forbidden"),
    [
        (
            "|-enditem|p2a: B|Choice Scarf|[silent]|[from] ability: Pickpocket|[of] p2a: B",
            [("zarude", "item", "choicescarf")],
            [("zarude", "ability", "pickpocket")],
        ),
        (
            "|-item|p1a: A|Choice Scarf|[from] ability: Pickpocket|[of] p2a: B",
            [
                ("tinkaton", "item", "choicescarf"),
                ("tinkaton", "ability", "pickpocket"),
            ],
            [("zarude", "ability", "pickpocket")],
        ),
        (
            "|-item|p1a: A|Leftovers|[from] ability: Magician|[of] p2a: B",
            [("tinkaton", "item", "leftovers"), ("tinkaton", "ability", "magician")],
            [("zarude", "ability", "magician")],
        ),
        (
            "|-item|p1a: A|Leftovers|[from] ability: Frisk|[of] p2a: B",
            [("tinkaton", "item", "leftovers"), ("zarude", "ability", "frisk")],
            [("tinkaton", "ability", "frisk"), ("zarude", "item", "leftovers")],
        ),
        (
            "|-ability|p1a: A|Intimidate|boost",
            [("tinkaton", "ability", "intimidate")],
            [("zarude", "ability", "intimidate")],
        ),
        (
            "|-unboost|p2a: B|atk|1|[from] ability: Intimidate|[of] p1a: A",
            [("tinkaton", "ability", "intimidate")],
            [("zarude", "ability", "intimidate")],
        ),
        (
            "|-ability|p1a: A|Water Absorb|Trace|[from] ability: Trace|[of] p2a: B",
            [
                ("tinkaton", "ability", "waterabsorb"),
                ("tinkaton", "ability", "trace"),
                ("zarude", "ability", "waterabsorb"),
            ],
            [("zarude", "ability", "trace")],
        ),
        (
            "|-activate|p1a: A|ability: Mummy|p2a: B|[ability] Iron Fist",
            [("tinkaton", "ability", "mummy"), ("zarude", "ability", "mummy")],
            [],
        ),
        (
            "|-ability|p2a: B|Mummy|Iron Fist|[from] ability: Mummy|[of] p1a: A",
            [("tinkaton", "ability", "mummy"), ("zarude", "ability", "mummy")],
            [],
        ),
        (
            "|-activate|p1a: A|ability: Lingering Aroma|p2a: B|[ability] Iron Fist",
            [
                ("tinkaton", "ability", "lingeringaroma"),
                ("zarude", "ability", "lingeringaroma"),
            ],
            [],
        ),
        (
            "|-item|p1a: A|Air Balloon",
            [("tinkaton", "item", "airballoon")],
            [("zarude", "item", "airballoon")],
        ),
        (
            "|-enditem|p1a: A|Air Balloon",
            [("tinkaton", "item", "airballoon")],
            [("zarude", "item", "airballoon")],
        ),
        (
            "|-heal|p1a: A|90/100|[from] item: Leftovers",
            [("tinkaton", "item", "leftovers")],
            [("zarude", "item", "leftovers")],
        ),
        (
            "|-damage|p2a: B|90/100|[from] item: Rocky Helmet|[of] p1a: A",
            [("tinkaton", "item", "rockyhelmet")],
            [("zarude", "item", "rockyhelmet")],
        ),
        (
            "|-damage|p1a: A|90/100|[from] item: Life Orb",
            [("tinkaton", "item", "lifeorb")],
            [("zarude", "item", "lifeorb")],
        ),
        (
            "|-damage|p2a: B|90/100|[from] ability: Rough Skin|[of] p1a: A",
            [("tinkaton", "ability", "roughskin")],
            [("zarude", "ability", "roughskin")],
        ),
        (
            "|-heal|p1a: A|100/100|[from] ability: Water Absorb|[of] p2a: B",
            [("tinkaton", "ability", "waterabsorb")],
            [("zarude", "ability", "waterabsorb")],
        ),
        (
            "|-block|p2a: B|ability: Aroma Veil|[of] p1a: A",
            [("tinkaton", "ability", "aromaveil")],
            [("zarude", "ability", "aromaveil")],
        ),
        (
            "|-block|p1a: A|item: Ability Shield",
            [("tinkaton", "item", "abilityshield")],
            [("zarude", "item", "abilityshield")],
        ),
        (
            "|-activate|p1a: A|ability: Forewarn|Thunderbolt|[of] p2a: B",
            [("tinkaton", "ability", "forewarn")],
            [("zarude", "ability", "forewarn")],
        ),
    ],
)
def test_effect_reveal_ownership(protocol, expected, forbidden):
    from pokeai.data.validate import _public_knowledge

    events = [
        ["", "switch", "p1a: A", "Tinkaton, F", "100/100"],
        ["", "switch", "p2a: B", "Zarude", "100/100"],
    ]
    before = {**_public_knowledge(events, "p1"), **_public_knowledge(events, "p2")}
    events.append(protocol.split("|"))
    after = {**_public_knowledge(events, "p1"), **_public_knowledge(events, "p2")}
    for species, field, value in expected:
        assert value not in before[species][field]
        assert value in after[species][field]
    for species, field, value in forbidden:
        assert value not in after[species][field]


def test_pickpocket_reveal_is_not_hidden_information(synthetic_run):
    run, _ = synthetic_run
    _, data, meta = trajectory(run)
    # The actual server protocol identifies the item donor in [of]. The
    # ability belongs to the recipient, not that donor.
    meta["public_events"][2:2] = [
        [
            "",
            "-enditem",
            "p1a: Pikachu",
            "Choice Scarf",
            "[silent]",
            "[from] ability: Pickpocket",
            "[of] p1a: Pikachu",
        ],
        [
            "",
            "-item",
            "p2a: Eevee",
            "Choice Scarf",
            "[from] ability: Pickpocket",
            "[of] p1a: Pikachu",
        ],
    ]
    meta["decision_event_counts"] = [4, 6]
    data["states"][0]["opponent_active_pokemon"]["ability"] = "pickpocket"
    data["states"][0]["opponent_active_pokemon"]["item"] = "choicescarf"
    assert leakage_errors(data, meta) == []
    meta["decision_event_counts"][0] = 2
    assert any("pickpocket" in error for error in leakage_errors(data, meta))
