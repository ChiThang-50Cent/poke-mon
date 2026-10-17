"""Canonical online order mapping must not guess labels for special requests."""

from types import SimpleNamespace

import pytest
from metamon.interface import UniversalAction
from poke_env.environment import Move, Pokemon
from poke_env.player import Player

from pokeai.env.collector import order_to_action


def battle_fixture() -> SimpleNamespace:
    active = Pokemon(gen=9, species="pikachu")
    active._active = True
    active._moves = {
        name: Move(name, gen=9)
        for name in ("thunderbolt", "tackle", "agility", "protect")
    }
    reserves = [Pokemon(gen=9, species=name) for name in ("zapdos", "blissey")]
    return SimpleNamespace(
        active_pokemon=active,
        available_moves=list(active.moves.values()),
        available_switches=reserves,
        team={str(i): p for i, p in enumerate([active, *reserves])},
        reviving=False,
        force_switch=False,
        can_tera=True,
    )


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 9, 10, 11, 12])
def test_order_roundtrip(index: int) -> None:
    battle = battle_fixture()
    order = UniversalAction.action_idx_to_BattleOrder(battle, index)
    assert order is not None
    assert order_to_action(battle, order) == (index, None)


def test_mapping_uses_alphabetical_move_and_switch_order() -> None:
    battle = battle_fixture()
    assert order_to_action(battle, Player.create_order(battle.available_moves[2])) == (
        0, None
    )
    assert order_to_action(battle, Player.create_order(battle.available_switches[1])) == (
        4, None
    )


def test_forced_switch_and_unavailable_tera() -> None:
    battle = battle_fixture()
    battle.force_switch = True
    order = UniversalAction.action_idx_to_BattleOrder(battle, 4)
    assert order_to_action(battle, order) == (4, None)
    battle.force_switch = False
    battle.can_tera = None
    order = UniversalAction.action_idx_to_BattleOrder(battle, 0)
    assert order_to_action(battle, order) == (0, None)


@pytest.mark.parametrize("move", ["struggle", "recharge"])
def test_special_moves_are_missing_not_coerced(move: str) -> None:
    battle = battle_fixture()
    special = Move(move, gen=9)
    battle.available_moves = [special]
    assert order_to_action(battle, Player.create_order(special)) == (-1, move)


def test_revival_is_missing_not_ordinary_switch() -> None:
    battle = battle_fixture()
    battle.reviving = True
    order = Player.create_order(battle.available_switches[0])
    assert order_to_action(battle, order) == (-1, "revival_target")


def test_collector_records_chosen_in_decision_mask(monkeypatch) -> None:
    from pokeai.env.collector import TrajectoryCollector, UniversalState

    battle = battle_fixture()
    battle.turn = 7
    state = SimpleNamespace(to_dict=lambda: {"snapshot": "decision-time"})
    monkeypatch.setattr(UniversalState, "from_Battle", lambda battle: state)
    monkeypatch.setattr(
        UniversalAction, "definitely_valid_actions",
        lambda state, battle: {UniversalAction(0), UniversalAction(9)},
    )
    collector = TrajectoryCollector()
    order = UniversalAction.action_idx_to_BattleOrder(battle, 9)
    collector.record(battle, order)
    assert collector.actions == [9]
    assert collector.masks[0][9] == 1
    assert collector.reasons == [None]
    assert collector.turns == [7]


def test_collector_rejects_chosen_outside_mask(monkeypatch) -> None:
    from pokeai.env.collector import TrajectoryCollector, UniversalState

    battle = battle_fixture()
    monkeypatch.setattr(UniversalState, "from_Battle", lambda battle: object())
    monkeypatch.setattr(
        UniversalAction, "definitely_valid_actions", lambda state, battle: set()
    )
    collector = TrajectoryCollector()
    order = UniversalAction.action_idx_to_BattleOrder(battle, 0)
    with pytest.raises(ValueError, match="outside mask"):
        collector.record(battle, order)
    assert collector.states == collector.actions == []
