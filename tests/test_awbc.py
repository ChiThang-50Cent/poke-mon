"""Synthetic W5.3 formula, provenance, frozen-label and selection contracts."""
import numpy as np
import pytest
import torch

from pokeai.training.awbc import (
    AdvantageSequences,
    awbc_loss,
    awbc_weights,
    forward_trajectory,
    frozen_advantages,
    make_arrays,
    select_median_seed,
    shuffle_advantages,
    weight_stats,
)
from pokeai.training.critic import include_decision
from pokeai.training.train_bc import pad_sequences


def example():
    logits = torch.zeros(3, 13, requires_grad=True)
    masks = torch.zeros(3, 13, dtype=torch.bool)
    masks[:, :2] = True
    actions = torch.tensor([0, 1, -1])
    valid = torch.tensor([True, True, False])
    reference = masks.float() / 2
    return logits, actions, masks, valid, reference


def test_clipped_exponential_weights_and_normalized_ess():
    advantage = torch.tensor([-1., 0., 1.], requires_grad=True)
    weights = awbc_weights(advantage, .5)
    torch.testing.assert_close(weights, torch.tensor([np.exp(-2), 1., 5.]).float())
    assert not weights.requires_grad
    stats = weight_stats(np.zeros(4), 1.)
    assert stats["ESS"] == 1.
    assert stats["ESS_absolute"] == 4.
    assert stats["clipped_fraction"] == 0.


@pytest.mark.parametrize("beta", [0., -1.])
def test_nonpositive_beta_rejected(beta):
    with pytest.raises(ValueError):
        awbc_weights(torch.zeros(1), beta)


def test_batch_weight_normalization_and_legal_forward_kl():
    logits, actions, masks, valid, reference = example()
    with torch.no_grad():
        logits[0, 0] = 1.
        logits[1, 0] = -1.
    reference[0, :2] = torch.tensor([.8, .2])
    weights = torch.tensor([1., 3., 1000.], requires_grad=True)
    loss, bc, kl = awbc_loss(logits, actions, masks, valid, weights, reference)
    logp = logits[:2, :2].log_softmax(-1)
    expected_bc = -(torch.tensor([.5, 1.5]) * logp[torch.arange(2), actions[:2]]).mean()
    expected_kl = (reference[:2, :2] * (reference[:2, :2].log() - logp)).sum(-1).mean()
    torch.testing.assert_close(bc, expected_bc)
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(loss, expected_bc + .1 * expected_kl)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0
    assert torch.count_nonzero(logits.grad[2]) == 0
    assert weights.grad is None
    scaled = awbc_loss(logits, actions, masks, valid, weights * 123, reference)[0]
    torch.testing.assert_close(scaled, loss)


def test_e1_unit_weights_is_masked_bc_plus_same_kl():
    logits, actions, masks, valid, reference = example()
    loss, bc, kl = awbc_loss(logits, actions, masks, valid, torch.ones(3), reference)
    torch.testing.assert_close(bc, torch.tensor(np.log(2)).float())
    torch.testing.assert_close(kl, torch.tensor(0.))
    torch.testing.assert_close(loss, bc)


def test_only_eligible_c_decisions_get_loss():
    c_hash = "c" * 64
    assert include_decision({"checkpoint_sha256": c_hash, "probabilities": [1.]}, c_hash)
    assert not include_decision({"checkpoint_sha256": "other", "probabilities": [1.]}, c_hash)
    assert not include_decision({"checkpoint_sha256": c_hash, "probabilities": None}, c_hash)
    logits, actions, masks, valid, reference = example()
    actions[:] = -1
    assert awbc_loss(logits, actions, masks, valid, torch.ones(3), reference) is None
    actions[:] = 12  # Legal-action check is independent of provenance/valid flag.
    assert awbc_loss(logits, actions, masks, valid, torch.ones(3), reference) is None


def test_advantage_is_detached_snapshot_before_actor_updates():
    outcomes = torch.tensor([1., 0.])
    values = torch.tensor([.3, .7], requires_grad=True)
    advantage = frozen_advantages(outcomes, values)
    weights = awbc_weights(advantage, 1.)
    with torch.no_grad():
        values.add_(.2)
        outcomes.zero_()
    torch.testing.assert_close(advantage, torch.tensor([.7, -.7]))
    torch.testing.assert_close(awbc_weights(advantage, 1.), weights)
    assert not advantage.requires_grad


def test_shuffle_preserves_opponent_phase_and_split_multisets():
    n = 96
    advantage = np.arange(n, dtype=float)
    opponent = np.array(["neural", "heuristic"] * (n // 2))
    phase = np.array(["early"] * 48 + ["late"] * 48)
    split = np.array(["train"] * 24 + ["val"] * 24 + ["train"] * 24 + ["val"] * 24)
    shuffled = shuffle_advantages(advantage, opponent, phase, split, 1)
    np.testing.assert_array_equal(shuffled, shuffle_advantages(advantage, opponent, phase, split, 1))
    assert not np.array_equal(shuffled, advantage)
    assert not np.array_equal(shuffled, shuffle_advantages(advantage, opponent, phase, split, 2))
    for group in set(zip(opponent, phase, split)):
        selected = (opponent == group[0]) & (phase == group[1]) & (split == group[2])
        np.testing.assert_array_equal(np.sort(shuffled[selected]), advantage[selected])
    np.testing.assert_array_equal(advantage, np.arange(n))


def test_median_seed_not_best_seed_and_ties_are_deterministic():
    assert select_median_seed({1: .1, 2: -.2, 3: .3}) == 1
    assert select_median_seed({3: .1, 1: .1, 2: .1}) == 2
    with pytest.raises(ValueError):
        select_median_seed({1: .1})


def test_full_trajectory_keeps_nonmodel_states_without_loss():
    seq = {"tokens": np.array([[1], [2], [3]]),
           "numbers": np.zeros((3, 48), dtype=np.float32),
           "masks": np.ones((3, 13), dtype=bool),
           "actions": np.array([0, -1, 1]),
           "valid": np.array([True, False, True]),
           "reference": np.ones((3, 13), dtype=np.float32) / 13,
           "row": np.array([0, -1, 1]), "split": "train"}
    arrays = make_arrays({"sequences": [seq]}, "train", np.array([2., 3.]))
    loader_batch = pad_sequences([AdvantageSequences(arrays)[0]])
    assert loader_batch["tokens"].shape == (1, 3, 1)
    assert loader_batch["weights"].tolist() == [[2., 0., 3.]]
    assert loader_batch["valid"].tolist() == [[True, False, True]]

    class AccumulatingModel:
        def __call__(self, tokens, numbers):
            return tokens.float().cumsum(1), None

    logits = forward_trajectory(AccumulatingModel(), loader_batch)
    assert logits.flatten().tolist() == [1., 3., 6.]
