"""CF1.5 head: inputs, decision rule, labels, a tiny fit on synthetic data."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from pokeai.cf.head import Arrays, LinearHead, MLPHead, choose, delta_scores
from pokeai.cf.head_train import fit, permute_labels, root_gains


def _arrays(n: int = 400, seed: int = 0) -> Arrays:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, 8)).astype(np.float32)
    slots = np.tile(np.array([0, 1, 9, 2]), (n, 1))
    valid = np.ones((n, 4), bool)
    q = np.zeros((n, 4), np.float32) + 0.5
    q[:, 1] += 0.3 * np.sign(h[:, 0])  # candidate 1 is better iff h0 > 0
    meta = {"pair": np.array([f"p{i % 50}" for i in range(n)])}
    return Arrays(h, np.zeros((n, 4), np.float32), slots, valid, q, meta)


def test_head_inputs_are_only_features_slots_logp() -> None:
    for cls in (LinearHead, MLPHead):
        assert list(inspect.signature(cls.forward).parameters) == ["self", "h", "slots", "logp"]


def test_a_c_is_zero_and_choice_is_valid() -> None:
    for head in (LinearHead(8), MLPHead(8)):
        h = torch.randn(5, 8)
        slots = torch.tensor([[0, 1, 9, 2]] * 5)
        delta = delta_scores(head, h, slots, torch.zeros(5, 4))
        assert torch.all(delta[:, 0] == 0) and torch.isfinite(delta).all()
    delta = np.array([[0.0, 0.03, -1.0, 0.0], [0.0, -0.1, -0.2, -0.3]])
    valid = np.array([[1, 1, 1, 0], [1, 1, 1, 1]], bool)
    assert choose(delta, valid, 0.02).tolist() == [1, 0]
    assert choose(delta, valid, 0.05).tolist() == [0, 0]


def test_permute_keeps_label_multiset_and_a_c() -> None:
    d = np.array([[0.0, 0.2, -0.1, 0.4]])
    out = permute_labels(d, np.ones((1, 4), bool), 3)
    assert out[0, 0] == 0.0 and sorted(out[0]) == sorted(d[0])


def test_fit_learns_synthetic_signal() -> None:
    tr, va = _arrays(600, 1), _arrays(300, 2)
    mean, std = tr.h.mean(0), tr.h.std(0)
    for loss in ("huber", "pairwise"):
        result = fit("linear", loss, 1, tr, va, mean, std, "cpu", max_epochs=40)
        assert result["best_val_delta_tau0"] > 0.05
        out = fit("linear", loss, 1, tr, va, mean, std, "cpu", shuffle_labels=True,
                  max_epochs=40)
        assert out["best_val_delta_tau0"] < result["best_val_delta_tau0"]
        gain, _ = root_gains(np.zeros((300, 4)), va, 0.0)
        assert gain.sum() == 0
