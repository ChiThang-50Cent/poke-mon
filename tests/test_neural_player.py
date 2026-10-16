"""Policy legality and recurrent-memory ownership, including local full games."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pokeai.agents.neural_player import NeuralPlayer
from pokeai.models.pokedecision import PokeDecision, PokeDecisionConfig


def policy() -> NeuralPlayer:
    player = NeuralPlayer.__new__(NeuralPlayer)
    player.model = PokeDecision(
        PokeDecisionConfig(vocab_size=20, token_len=3), np.zeros(48), np.ones(48)
    ).eval()
    player.encoder = SimpleNamespace(unknown_id=0)
    player.device = torch.device("cpu")
    player.sample = False
    player.temperature = 1.0
    player.hidden = {}
    player._model_choice = {}
    player.latencies_ms = []
    player.unknown_tokens = 0
    player.total_tokens = 0
    player.fallbacks = Counter()
    return player


def encoded() -> dict:
    return {"tokens": np.array([1, 2, 3]), "numbers": np.zeros(48, np.float32)}


@pytest.mark.parametrize("sample", [False, True])
def test_action_is_always_masked_and_memory_isolated(sample: bool) -> None:
    player = policy()
    player.sample = sample
    for index in range(13):
        mask = np.arange(13) == index
        assert player.select_action("a", encoded(), mask) == index
    state_a = player.hidden["a"].clone()
    player.select_action("b", encoded(), np.ones(13, bool))
    assert torch.equal(player.hidden["a"], state_a)
    assert not torch.equal(player.hidden["a"], player.hidden["b"])


def test_all_masked_uses_fallback_without_advancing_memory() -> None:
    player = policy()
    assert player.select_action("a", encoded(), np.zeros(13, bool)) is None
    assert "a" not in player.hidden
    battle = SimpleNamespace(battle_tag="a", reviving=False)
    player.choose_random_move = lambda _: "fallback-order"
    assert player._fallback(battle, "all_masked") == "fallback-order"
    assert player.fallbacks == {"all_masked": 1}
    assert player._model_choice["a"] is False


def test_revival_fallback_targets_fainted_party_slot() -> None:
    player = policy()
    battle = SimpleNamespace(
        battle_tag="a", reviving=True,
        _last_request={"side": {"pokemon": [
            {"condition": "100/100"}, {"condition": "0 fnt"},
            {"condition": "100/100"},
        ]}},
    )
    order = player._fallback(battle, "revival_target")
    assert order.message == "/choose switch 2"
    assert not player._model_choice["a"]


def test_preview_resets_only_its_own_hidden_state(monkeypatch) -> None:
    player = policy()
    player.hidden = {"a": torch.ones(1), "b": torch.ones(1)}
    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer.teampreview", lambda *_: "/team 123456"
    )
    assert player.teampreview(SimpleNamespace(battle_tag="a")) == "/team 123456"
    assert "a" not in player.hidden
    assert "b" in player.hidden


def test_finished_battle_releases_memory(monkeypatch) -> None:
    player = policy()
    player.hidden = {"a": torch.ones(1), "b": torch.ones(1)}
    player._model_choice = {"a": True, "b": True}
    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer._battle_finished_callback",
        lambda *_: None,
    )
    player._battle_finished_callback(SimpleNamespace(battle_tag="a"))
    assert set(player.hidden) == {"b"}
    assert player._model_choice == {"b": True}


def test_server_errors_are_attributed_to_model_not_fallback(monkeypatch) -> None:
    import asyncio

    player = policy()
    player.illegal_actions = 0
    player._model_choice = {"a": True, "b": False}

    async def accept_messages(*_) -> None:
        return None

    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer._handle_battle_message",
        accept_messages,
    )
    for tag in ("a", "b"):
        asyncio.run(player._handle_battle_message([
            [tag], ["", "error", "[Invalid choice] rejected"],
        ]))
    assert player.illegal_actions == 1
    assert player.last_server_error == "|error|[Invalid choice] rejected"


@pytest.mark.server
def test_neural_completes_two_battles(tmp_path) -> None:
    import asyncio
    from dataclasses import asdict

    import yaml

    from pokeai.data.features import ObservationEncoder
    from pokeai.env.runner import RunConfig, run_battles
    from pokeai.models.pokedecision import save_checkpoint

    encoder = ObservationEncoder()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metadata_path = model_dir / "tokenizer.json"
    encoder.save(metadata_path)
    import json

    metadata = json.loads(metadata_path.read_text())
    model = PokeDecision(
        PokeDecisionConfig(
            vocab_size=metadata["vocab_size"], token_len=metadata["token_len"]
        ), np.zeros(48), np.ones(48),
    )
    save_checkpoint(model, model_dir / "best.pt")
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": asdict(model.config)})
    )
    summary = asyncio.run(run_battles(RunConfig(
        player=f"neural:{model_dir}", opponent="RandomBaseline", battles=2,
        concurrent=2, out=str(tmp_path / "runs"),
    )))
    assert summary["completed"] == 2
    assert summary["error_count"] == 0
    assert summary["neural"]["player"]["illegal_actions"] == 0
