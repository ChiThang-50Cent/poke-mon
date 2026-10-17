"""Synthetic preview adapters: prediction inputs, random behavior, and accounting."""

import random
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pokeai.agents.neural_player import NeuralPlayer

SPECIES = ["pikachu", "charizard", "blastoise", "venusaur", "gengar", "dragonite"]


def preview_battle(tag="preview", our_size=6, opponent_size=6):
    def team(size):
        return {
            f"slot-{i}": SimpleNamespace(
                species=species,
                moves={"thunderbolt": SimpleNamespace(id="thunderbolt")},
                ability="static", item="leftovers",
            )
            for i, species in enumerate(SPECIES[:size])
        }

    return SimpleNamespace(
        battle_tag=tag, team=team(our_size), opponent_team=team(opponent_size)
    )


def preview_player(model=None):
    player = NeuralPlayer.__new__(NeuralPlayer)
    player.team_preview_model = model
    player._format = "gen9ou"
    player.hidden = {"preview": object(), "other": object()}
    player._model_choice = {"preview": True, "other": True}
    player.fallbacks = Counter()
    player.lead_sources = Counter()
    player.team_preview_records = {}
    player.protocol_retries = {}
    player.history = {}
    player.latencies_ms = []
    player.illegal_actions = 0
    player.unknown_tokens = 0
    player.total_tokens = 0
    return player


def preview_model(predicted="blastoise", formats=("gen9ou",)):
    return SimpleNamespace(
        trained_formats=formats,
        predict_lead=Mock(return_value=(
            predicted, torch.tensor([0.1, 0.125, 0.5, 0.125, 0.1, 0.05]),
            sorted(SPECIES),
        )),
    )


def test_default_random_preview_preserves_shuffle_and_counts(monkeypatch):
    player = preview_player()
    calls = []

    def shuffle(members):
        calls.append(members.copy())
        members[:] = [4, 2, 6, 1, 3, 5]

    monkeypatch.setattr(random, "shuffle", shuffle)
    assert player.teampreview(preview_battle()) == "/team 426135"
    assert calls == [[1, 2, 3, 4, 5, 6]]
    assert player.fallbacks == {"team_preview": 1}
    assert player.neural_stats()["lead_sources"] == {"random_default": 1}
    assert player.team_preview_records["preview"] == {
        "source": "random_default", "chosen_lead": "venusaur",
        "top_probability": None, "order": "/team 426135",
        "probabilities": None, "chosen_index": -1, "fallback_reason": "team_preview",
    }
    assert set(player.hidden) == {"other"}
    assert player._model_choice == {"preview": False, "other": True}


def test_model_preview_matches_vendor_inputs_and_stable_remaining_order(caplog):
    model = preview_model()
    player = preview_player(model)
    assert player.teampreview(preview_battle()) == "/team 312456"
    model.predict_lead.assert_called_once_with(
        our_team=SPECIES,
        our_team_moves=[["thunderbolt"] for _ in SPECIES],
        our_team_abilities=["static"] * 6,
        our_team_items=["leftovers"] * 6,
        opponent_team=SPECIES,
        battle_format="gen9ou",
    )
    assert player.team_preview_records["preview"] == {
        "source": "model", "chosen_lead": "blastoise",
        "top_probability": 0.5, "order": "/team 312456",
        "probabilities": model.predict_lead.return_value[1].tolist(),
        "chosen_index": -1, "fallback_reason": "team_preview",
    }
    assert player.fallbacks == {}
    assert player.neural_stats()["lead_sources"] == {"model": 1}
    assert not caplog.records


@pytest.mark.parametrize(
    "reason,formats,our_size,opponent_size,predicted,probability",
    [
        ("format_not_trained", ("gen8ou",), 6, 6, "blastoise", None),
        ("team_size_not_six", ("gen9ou",), 5, 6, "blastoise", None),
        ("team_size_not_six", ("gen9ou",), 6, 5, "blastoise", None),
        ("predicted_lead_not_found", ("gen9ou",), 6, 6, "mewtwo", 0.5),
    ],
)
def test_model_fallback_reason_is_retained_without_vendor_warning(
    reason, formats, our_size, opponent_size, predicted, probability, monkeypatch,
    caplog,
):
    model = preview_model(predicted, formats)
    player = preview_player(model)
    monkeypatch.setattr(random, "shuffle", lambda members: members.reverse())
    battle = preview_battle(our_size=our_size, opponent_size=opponent_size)
    order = player.teampreview(battle)
    source = f"model_fallback:{reason}"
    assert order == "/team " + "".join(str(i) for i in range(our_size, 0, -1))
    assert player.team_preview_records["preview"] == {
        "source": source, "chosen_lead": SPECIES[our_size - 1],
        "top_probability": probability, "order": order,
        "probabilities": (
            model.predict_lead.return_value[1].tolist()
            if probability is not None else None
        ),
        "chosen_index": -1, "fallback_reason": "team_preview",
    }
    assert player.fallbacks == {"team_preview": 1}
    assert player.neural_stats()["lead_sources"] == {source: 1}
    assert model.predict_lead.call_count == (reason == "predicted_lead_not_found")
    assert not caplog.records


def test_preview_records_and_source_counts_are_isolated_across_battles(monkeypatch):
    player = preview_player(preview_model())
    player.teampreview(preview_battle("model-battle"))
    player.team_preview_model = None
    monkeypatch.setattr(random, "shuffle", lambda members: None)
    player.teampreview(preview_battle("random-battle"))
    assert player.team_preview_records["model-battle"]["chosen_lead"] == "blastoise"
    assert player.team_preview_records["random-battle"]["chosen_lead"] == "pikachu"
    assert player.neural_stats()["lead_sources"] == {"model": 1, "random_default": 1}
    assert player.fallbacks == {"team_preview": 1}
