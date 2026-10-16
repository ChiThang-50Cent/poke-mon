"""Behavioral regression tests for the W3 masked policy."""

import torch

from pokeai.models.pokedecision import (
    PokeDecision,
    PokeDecisionConfig,
    load_pokedecision,
    masked_logits,
    save_checkpoint,
)
from pokeai.training.train_bc import masked_bc_loss


def test_loss_excludes_missing_invalid_and_unrepresentable_steps():
    torch.manual_seed(7)
    logits = torch.randn(1, 6, 13)
    masks = torch.ones_like(logits, dtype=torch.bool)
    masks[:, 4] = False
    masks[:, 5, 3] = False
    actions = torch.tensor([[2, -1, 4, 99, 0, 3]])
    valid = torch.tensor([[True, False, False, True, True, True]])
    original = masked_bc_loss(logits, actions, masks, valid)
    changed = actions.clone()
    changed[:, 1:3] = torch.tensor([9, -1])
    assert torch.equal(original, masked_bc_loss(logits, changed, masks, valid))
    assert torch.equal(original, torch.nn.functional.cross_entropy(logits[:, 0],
                                                                  actions[:, 0]))
    assert masked_bc_loss(logits, actions, masks, torch.zeros_like(valid)) is None


def test_masked_policy_never_selects_illegal_actions():
    torch.manual_seed(9)
    logits = torch.randn(20, 32, 13)
    masks = torch.rand(20, 32, 13) > 0.5
    masks[..., 0] = True
    selected = masked_logits(logits, masks).argmax(-1)
    assert masks.gather(-1, selected.unsqueeze(-1)).all()
    assert torch.isneginf(masked_logits(logits, masks)[~masks]).all()


def test_checkpoint_load_is_bit_identical_on_cpu(tmp_path):
    from dataclasses import asdict

    import yaml

    from pokeai.data.features import ObservationEncoder

    torch.manual_seed(42)
    encoder = ObservationEncoder()
    encoder.save(tmp_path / "tokenizer.json")
    import json
    metadata = json.loads((tmp_path / "tokenizer.json").read_text())
    config = PokeDecisionConfig(
        vocab_size=metadata["vocab_size"], token_len=metadata["token_len"],
        d_model=8, hidden_size=16, numeric_size=8,
    )
    for arch in ("gru", "mlp"):
        config.arch = arch
        model = PokeDecision(config, torch.zeros(48), torch.ones(48)).eval()
        tokens = torch.randint(config.vocab_size, (2, 3, config.token_len))
        numbers = torch.randn(2, 3, 48)
        with torch.no_grad():
            expected, _ = model(tokens, numbers)
        save_checkpoint(model, tmp_path / "best.pt")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"model": asdict(config)})
        )
        restored, _, loaded_config = load_pokedecision(tmp_path, "cpu")
        with torch.no_grad():
            actual, _ = restored(tokens, numbers)
        assert loaded_config == config
        assert torch.equal(actual, expected)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_gru_hidden_state_matches_uninterrupted_sequence():
    torch.manual_seed(11)
    model = PokeDecision(
        PokeDecisionConfig(vocab_size=20, token_len=4, d_model=8,
                           hidden_size=16, numeric_size=8),
        torch.zeros(48), torch.ones(48),
    ).eval()
    tokens, numbers = torch.randint(20, (1, 4, 4)), torch.randn(1, 4, 48)
    with torch.no_grad():
        full, _ = model(tokens, numbers)
        _, hidden = model(tokens[:, :2], numbers[:, :2])
        continuation, _ = model(tokens[:, 2:], numbers[:, 2:], hidden)
        reset, _ = model(tokens[:, :2], numbers[:, :2], None)
    torch.testing.assert_close(continuation, full[:, 2:])
    torch.testing.assert_close(reset, full[:, :2])
