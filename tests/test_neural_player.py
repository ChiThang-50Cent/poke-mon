"""Policy legality and recurrent-memory ownership, including local full games."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pokeai.agents.neural_player import NeuralPlayer
from pokeai.models.pokedecision import PokeDecision, PokeDecisionConfig


@pytest.mark.parametrize("request_species", ["Cramorant-Gorging", "Cramorant"])
def test_switch_uses_request_slot_not_form_name(request_species) -> None:
    from pokeai.agents.neural_player import switch_by_party_slot

    battle = SimpleNamespace(_last_request={"side": {"pokemon": [
        {"ident": "p1: Rillaboom", "details": "Rillaboom, L100"},
        {"ident": "p1: Cramorant", "details": f"{request_species}, L100"},
    ]}})
    order = SimpleNamespace(
        message="/choose switch cramorantgorging",
        order=SimpleNamespace(species="cramorantgorging"),
    )
    assert switch_by_party_slot(battle, order).message == "/choose switch 2"

def policy(value_head: bool = False) -> NeuralPlayer:
    player = NeuralPlayer.__new__(NeuralPlayer)
    player.model = PokeDecision(
        PokeDecisionConfig(
            vocab_size=20, token_len=3, d_model=8, hidden_size=16,
            numeric_size=8, value_head=value_head,
        ), np.zeros(48), np.ones(48)
    ).eval()
    player.encoder = SimpleNamespace(unknown_id=0)
    player.device = torch.device("cpu")
    player.sample = False
    player.temperature = 1.0
    player.hidden = {}
    player.decision_values = {}
    player._model_choice = {}
    player.latencies_ms = []
    player.unknown_tokens = 0
    player.total_tokens = 0
    player.fallbacks = Counter()
    player.team_preview_model = None
    player.team_preview_records = {}
    player.lead_sources = Counter()
    player._hidden_before = {}
    player._pending_choice = {}
    player._public_log = {}
    player.history = {}
    player.protocol_retries = {}
    player.illegal_actions = 0
    player.last_server_error = None
    player.decisions = 0
    player.collector = None
    player._behavior = {}
    player.policy_version = "test-policy"
    player.checkpoint_sha256 = "test-checkpoint"
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
    assert player._behavior["a"]["probabilities"] is None
    assert player._behavior["a"]["chosen_index"] == -1
    assert player._behavior["a"]["fallback_reason"] == "all_masked"
    assert player._model_choice["a"] is False


@pytest.mark.parametrize("sample", [False, True])
def test_values_are_per_battle_and_do_not_change_actions(sample) -> None:
    torch.manual_seed(13)
    legacy = policy()
    torch.manual_seed(13)
    valued = policy(value_head=True)
    legacy.sample = valued.sample = sample
    with torch.no_grad():
        valued.model.value_head.weight.zero_()
        valued.model.value_head.bias.fill_(torch.log(torch.tensor(3.0)))
    for tag in ("a", "b", "a"):
        torch.manual_seed(29)
        expected = legacy.select_action(tag, encoded(), np.ones(13, bool))
        torch.manual_seed(29)
        actual = valued.select_action(tag, encoded(), np.ones(13, bool))
        assert actual == expected
        assert torch.equal(valued.hidden[tag], legacy.hidden[tag])
    assert valued.decision_values["a"] == pytest.approx([0.75, 0.75])
    assert valued.decision_values["b"] == pytest.approx([0.75])
    assert legacy.decision_values == {}
    assert valued.select_action("a", encoded(), np.zeros(13, bool)) is None
    assert len(valued.decision_values["a"]) == 2


@pytest.mark.parametrize("sample", [False, True])
def test_behavior_records_exact_sampling_distribution(sample, monkeypatch) -> None:
    player = policy(value_head=True)
    player.sample = sample
    player.collector = SimpleNamespace()
    player.temperature = 0.75
    mask = np.array([True, False, True] + [False] * 10)
    captured = []
    original = torch.multinomial

    def capture(probabilities, count):
        captured.append(probabilities.clone())
        return original(probabilities, count)

    monkeypatch.setattr(torch, "multinomial", capture)
    action = player.select_action("a", encoded(), mask)
    behavior = player._behavior["a"]
    probabilities = np.asarray(behavior["probabilities"])
    assert probabilities.sum() == pytest.approx(1, abs=1e-6)
    assert not probabilities[~mask].any()
    assert behavior["chosen_index"] == action
    assert behavior["log_probability"] == pytest.approx(np.log(probabilities[action]))
    assert behavior["value"] is not None
    assert behavior["mode"] == ("sample" if sample else "argmax")
    if sample:
        np.testing.assert_array_equal(probabilities, captured[0].numpy())


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
    battle = SimpleNamespace(
        battle_tag="a",
        team={str(i): SimpleNamespace(species=f"species{i}") for i in range(6)},
    )
    assert player.teampreview(battle) == "/team 123456"
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


@pytest.mark.parametrize("value_head", [False, True])
@pytest.mark.parametrize("existing_hidden", [False, True])
def test_expected_trapped_retry_restores_memory_and_preserves_evidence(
    monkeypatch, existing_hidden, value_head
) -> None:
    import asyncio

    player = policy(value_head=value_head)
    if existing_hidden:
        player.select_action("a", encoded(), np.arange(13) == 0)
    before = player.hidden.get("a")
    player.select_action("a", encoded(), np.arange(13) == 4)
    player._model_choice["a"] = True
    player._pending_choice["a"] = {
        "order": "/choose switch 2", "trapped": False, "maybeTrapped": True,
        "active_species": "corviknight", "active_ability": "pressure",
        "opponent_species": "magnezone", "opponent_ability": None,
    }
    vendor_calls = []

    async def accept_messages(_self, messages):
        vendor_calls.append(messages)

    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer._handle_battle_message",
        accept_messages,
    )
    error = "[Unavailable choice] Can't switch: The active Pokémon is trapped"
    asyncio.run(player._handle_battle_message([
        [">a"], ["", "-ability", "p2a: Magnezone", "Magnet Pull"],
        ["", "error", error],
    ]))
    assert len(vendor_calls) == 1  # No second request or order sent by our adapter.
    assert player.illegal_actions == 0
    assert player.last_server_error is None
    assert player.neural_stats()["protocol_retries"] == 1
    evidence = player.protocol_retries["a"][0]
    assert evidence["order"] == "/choose switch 2"
    assert evidence["maybeTrapped"] and not evidence["trapped"]
    assert evidence["opponent_species"] == "magnezone"
    assert evidence["error"] == "|error|" + error
    assert evidence["public_log"][0] == "|-ability|p2a: Magnezone|Magnet Pull"
    if before is None:
        assert "a" not in player.hidden
    else:
        assert torch.equal(player.hidden["a"], before)
    if value_head:
        assert len(player.decision_values["a"]) == int(existing_hidden)
    with torch.inference_mode():
        _, expected = player.model(
            torch.as_tensor(encoded()["tokens"])[None, None],
            torch.as_tensor(encoded()["numbers"])[None, None], before,
        )
    player.select_action("a", encoded(), np.arange(13) == 0)
    assert torch.equal(player.hidden["a"], expected)
    if value_head:
        assert len(player.decision_values["a"]) == int(existing_hidden) + 1


@pytest.mark.parametrize(
    "maybe_trapped,trapped,order",
    [(False, False, "/choose switch 2"), (True, True, "/choose switch 2"),
     (True, False, "/choose move 1")],
)
def test_unexpected_trapped_rejection_remains_illegal(
    monkeypatch, maybe_trapped, trapped, order
) -> None:
    import asyncio

    player = policy()
    player._model_choice["a"] = True
    player._pending_choice["a"] = {
        "order": order, "maybeTrapped": maybe_trapped, "trapped": trapped,
    }

    async def accept_messages(*_):
        return None

    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer._handle_battle_message",
        accept_messages,
    )
    error = "[Invalid choice] Can't switch: The active Pokémon is trapped"
    asyncio.run(player._handle_battle_message([["a"], ["", "error", error]]))
    assert player.illegal_actions == 1
    assert player.last_server_error == "|error|" + error
    assert player.protocol_retries == {}


def test_known_trapped_request_explicitly_masks_switches(monkeypatch) -> None:
    from metamon.interface import UniversalAction, UniversalState

    player = policy()
    player.encoder = SimpleNamespace(encode=lambda _: encoded(), unknown_id=0)
    battle = SimpleNamespace(
        battle_tag="a", reviving=False, trapped=True, force_switch=False,
        available_moves=[SimpleNamespace(id="tackle")],
        _last_request={"active": [{"trapped": True}], "rqid": 1},
        active_pokemon=SimpleNamespace(species="corviknight", ability="pressure"),
        opponent_active_pokemon=SimpleNamespace(species="magnezone", ability=None),
        turn=1,
    )
    monkeypatch.setattr(UniversalState, "from_Battle", lambda _: object())
    # Even if vendor legality accidentally offers a switch, explicit trapped wins.
    monkeypatch.setattr(
        UniversalAction, "definitely_valid_actions",
        lambda *_: {UniversalAction(0), UniversalAction(4)},
    )

    def convert(action, _):
        assert action.action_idx == 0
        return SimpleNamespace(message="/choose move 1")

    monkeypatch.setattr(UniversalAction, "to_BattleOrder", convert)
    assert player.choose_move(battle).message == "/choose move 1"
    assert player._pending_choice["a"]["trapped"]


def test_default_request_clears_previous_model_order_evidence(monkeypatch) -> None:
    import asyncio

    player = policy()
    player._model_choice["a"] = True
    player._pending_choice["a"] = {
        "order": "/choose switch 2", "trapped": False, "maybeTrapped": True,
    }
    calls = []

    async def native_default(_, battle, maybe_default_order=False):
        assert "a" not in player._pending_choice
        assert player._model_choice["a"] is False
        calls.append((battle.battle_tag, maybe_default_order))

    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer._handle_battle_request",
        native_default,
    )
    asyncio.run(player._handle_battle_request(
        SimpleNamespace(battle_tag="a"), maybe_default_order=True
    ))
    assert calls == [("a", True)]
