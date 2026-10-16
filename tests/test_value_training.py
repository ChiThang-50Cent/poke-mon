"""Padding and frozen-policy regression coverage for W4.4."""

from dataclasses import asdict

import numpy as np
import torch

from pokeai.models.pokedecision import PokeDecision, PokeDecisionConfig
from pokeai.training.train_bc import initialize_model, pad_sequences, value_loss


def test_value_loss_masks_padding_but_not_bc_exclusion():
    logits = torch.tensor([[0.0, 2.0, 100.0]], requires_grad=True)
    real = torch.tensor([[True, True, False]])
    loss = value_loss(logits, torch.ones_like(logits), real)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0, :2], torch.ones(2)
    )
    assert torch.equal(loss, expected)
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert logits.grad[0, 1] != 0


def test_pad_real_includes_invalid_observations():
    def item(length):
        return {'actions': torch.zeros(length, dtype=torch.long),
                'valid': torch.zeros(length, dtype=torch.bool),
                'real': torch.ones(length, dtype=torch.bool),
                'outcomes': torch.ones(length)}
    padded = pad_sequences([item(2), item(1)])
    assert padded['real'].tolist() == [[True, True], [True, False]]
    assert not padded['valid'].any()


def test_probe_initialization_preserves_all_policy_state(tmp_path):
    cfg = PokeDecisionConfig(vocab_size=50, token_len=2)
    original = PokeDecision(cfg, np.zeros(48), np.ones(48))
    legacy_config = asdict(cfg)
    legacy_config.pop('value_head')
    checkpoint = tmp_path / 'best.pt'
    torch.save({'config': legacy_config, 'state_dict': original.state_dict()}, checkpoint)
    config = {'model': {'value_head': True}, 'init_checkpoint': str(checkpoint),
              'value_mode': 'probe'}
    model = initialize_model(config, 'gru', {'mean': [0.] * 48, 'std': [1.] * 48},
                             {'vocab_size': 50, 'token_len': 2})
    for name, tensor in original.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor)
    assert all(p.requires_grad == name.startswith('value_head.')
               for name, p in model.named_parameters())
