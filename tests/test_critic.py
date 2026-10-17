"""Fast synthetic W5.2 critic loss, provenance, split, and control tests."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pokeai.training.critic import (
    chosen_action_loss,
    include_decision,
    shuffled_actions,
    split_pairs,
    validate_outcome,
)

C_HASH = "c" * 64
SEED = 20261016


def test_q_loss_backpropagates_only_through_observed_action():
    logits = torch.linspace(-1, 1, 13 * 13).reshape(13, 13).requires_grad_()
    actions = torch.arange(13)
    targets = (actions % 2).float()

    loss = chosen_action_loss(logits, actions, targets)
    expected = F.binary_cross_entropy_with_logits(
        logits[torch.arange(13), actions], targets
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()

    observed = torch.eye(13, dtype=torch.bool)
    assert torch.count_nonzero(logits.grad[~observed]) == 0
    assert torch.all(logits.grad[observed] != 0)
    assert torch.all(logits.grad.diagonal()[targets == 0] > 0)
    assert torch.all(logits.grad.diagonal()[targets == 1] < 0)

    # Unobserved counterfactual values must not become implicit MC labels.
    changed = logits.detach().clone()
    changed[~observed] += 100
    torch.testing.assert_close(
        chosen_action_loss(changed, actions, targets), loss.detach()
    )


@pytest.mark.parametrize("probability_key", ["behavior_probs", "probabilities"])
def test_include_only_behavior_logged_c_checkpoint(probability_key):
    record = {
        "checkpoint_sha256": C_HASH,
        probability_key: [1.0] + [0.0] * 12,
    }
    assert include_decision(record, C_HASH)


@pytest.mark.parametrize(
    "record",
    [
        {"checkpoint_sha256": "0" * 64, "behavior_probs": [1 / 13] * 13},
        {"checkpoint_sha256": "1" * 64, "behavior_probs": [1 / 13] * 13},
        {"checkpoint_sha256": None, "behavior_probs": None},
        {"checkpoint_sha256": C_HASH, "behavior_probs": None},
        {"checkpoint_sha256": C_HASH},
        {"behavior_probs": [1 / 13] * 13},
    ],
    ids=["v0", "joint", "heuristic", "fallback", "missing-probs", "missing-hash"],
)
def test_exclude_offpolicy_and_unlogged_decisions(record):
    assert not include_decision(record, C_HASH)


@pytest.mark.parametrize("uppercase", [False, True])
@pytest.mark.parametrize("learner", [False, True])
@pytest.mark.parametrize("winner", ["player", "opponent"])
def test_outcome_is_from_trajectory_pov(winner, learner, uppercase):
    target = float((winner == "player") == learner)
    result = "win" if target else "loss"
    meta = {
        "learner": learner,
        "player_side": "p1" if learner else "p2",
        "final_result": result.upper() if uppercase else result,
    }
    assert validate_outcome(meta, {"winner": winner}) == target


@pytest.mark.parametrize("player_side", ["p1", "p2"])
def test_outcome_can_map_explicit_learner_side(player_side):
    meta = {
        "player_side": player_side,
        "final_result": "WIN" if player_side == "p2" else "LOSS",
    }
    battle = {"winner": "player", "learner_pov": "p2"}
    assert validate_outcome(meta, battle) == float(player_side == "p2")


@pytest.mark.parametrize("mutation", ["side", "team"])
def test_learner_mapping_matches_battle_challenger_and_team(mutation):
    meta = {
        "learner": True, "player_side": "p2",
        "final_result": "win", "team_file": "learner-team",
    }
    battle = {
        "winner": "player", "player_challenges": False,
        "player_team": "learner-team", "opponent_team": "opponent-team",
    }
    assert validate_outcome(meta, battle) == 1
    meta["player_side" if mutation == "side" else "team_file"] = "wrong"
    with pytest.raises(ValueError):
        validate_outcome(meta, battle)


@pytest.mark.parametrize("learner", [False, True])
def test_reversed_pov_label_raises(learner):
    meta = {
        "learner": learner,
        "player_side": "p1" if learner else "p2",
        "final_result": "loss" if learner else "win",
    }
    with pytest.raises(ValueError):
        validate_outcome(meta, {"winner": "player"})


@pytest.mark.parametrize("result", ["tie", "TIE"])
def test_ties_are_excluded_from_binary_outcome_targets(result):
    meta = {"learner": True, "player_side": "p1", "final_result": result}
    assert validate_outcome(meta, {"winner": "tie"}) is None


@pytest.mark.parametrize(
    "meta,battle",
    [
        ({"learner": True, "final_result": "unknown"}, {"winner": "player"}),
        ({"learner": True}, {"winner": "player"}),
        ({"learner": True, "final_result": "win"}, {"winner": None}),
        ({"learner": True, "final_result": "win"}, {}),
    ],
    ids=["unknown-result", "missing-result", "incomplete-battle", "missing-winner"],
)
def test_unknown_or_incomplete_outcomes_fail_closed(meta, battle):
    with pytest.raises(ValueError):
        validate_outcome(meta, battle)


def test_split_keeps_battles_and_unordered_team_pairs_together():
    battles = []
    pair_ids = []
    for pair in range(50):
        ids = []
        for reverse in (False, True):
            for opponent in ("C", "GymLeader"):
                battle_id = f"battle-{pair}-{reverse}-{opponent}"
                teams = (f"team-{pair}-a", f"team-{pair}-b")
                player_team, opponent_team = teams[::-1] if reverse else teams
                battles.append(
                    {
                        "battle_id": battle_id,
                        "player_team": player_team,
                        "opponent_team": opponent_team,
                        "opponent_type": opponent,
                    }
                )
                ids.append(battle_id)
        pair_ids.append(ids)

    split = split_pairs(battles, seed=SEED)
    assert split == split_pairs(battles, seed=SEED)
    assert set(split) == {battle["battle_id"] for battle in battles}
    assert set(split.values()) == {"train", "val"}
    for ids in pair_ids:
        assert len({split[battle_id] for battle_id in ids}) == 1

    train_pairs = {
        tuple(sorted((b["player_team"], b["opponent_team"])))
        for b in battles
        if split[b["battle_id"]] == "train"
    }
    val_pairs = {
        tuple(sorted((b["player_team"], b["opponent_team"])))
        for b in battles
        if split[b["battle_id"]] == "val"
    }
    assert train_pairs.isdisjoint(val_pairs)


def test_shuffled_action_control_is_legal_and_seeded():
    # Different legal sets make an unconstrained global permutation observable.
    masks = np.zeros((104, 13), dtype=bool)
    actions = np.arange(104, dtype=np.int64) % 13
    for row, action in enumerate(actions):
        masks[row, action] = True
        if row % 4:
            masks[row, (action + 1) % 13] = True
            masks[row, (action + 5) % 13] = True
    probs = masks.astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    original_actions = actions.copy()
    original_masks = masks.copy()
    original_probs = probs.copy()

    shuffled = shuffled_actions(actions, masks, probs, seed=SEED)
    assert isinstance(shuffled, np.ndarray)
    assert shuffled.shape == actions.shape
    assert np.issubdtype(shuffled.dtype, np.integer)
    assert np.all((0 <= shuffled) & (shuffled < 13))
    assert masks[np.arange(len(actions)), shuffled].all()
    np.testing.assert_array_equal(shuffled[::4], actions[::4])
    np.testing.assert_array_equal(
        shuffled, shuffled_actions(actions, masks, probs, seed=SEED)
    )
    np.testing.assert_array_equal(actions, original_actions)
    np.testing.assert_array_equal(masks, original_masks)
    np.testing.assert_array_equal(probs, original_probs)
