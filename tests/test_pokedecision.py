"""Behavioral regression tests for the W3 masked policy."""

import pytest
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


@pytest.mark.parametrize("value_head,legacy", [(False, False), (False, True), (True, False)])
def test_checkpoint_load_is_bit_identical_on_cpu(tmp_path, value_head, legacy):
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
        d_model=8, hidden_size=16, numeric_size=8, value_head=value_head,
    )
    for arch in ("gru", "mlp"):
        config.arch = arch
        model = PokeDecision(config, torch.zeros(48), torch.ones(48)).eval()
        tokens = torch.randint(config.vocab_size, (2, 3, config.token_len))
        numbers = torch.randn(2, 3, 48)
        with torch.no_grad():
            expected, _ = model(tokens, numbers)
            if value_head:
                policy, expected_value, value_hidden = model.forward_with_value(
                    tokens, numbers
                )
                assert torch.equal(policy, expected)
                assert expected_value.shape == (2, 3)
                if arch == "gru":
                    _, hidden = model(tokens, numbers)
                    assert torch.equal(value_hidden, hidden)
                else:
                    assert value_hidden is None
        save_checkpoint(model, tmp_path / "best.pt")
        recorded_config = asdict(config)
        if legacy:
            recorded_config.pop("value_head")
            checkpoint = torch.load(tmp_path / "best.pt", weights_only=True)
            checkpoint["config"].pop("value_head")
            torch.save(checkpoint, tmp_path / "best.pt")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"model": recorded_config})
        )
        restored, _, loaded_config = load_pokedecision(tmp_path, "cpu")
        with torch.no_grad():
            actual, _ = restored(tokens, numbers)
        assert loaded_config == config
        assert torch.equal(actual, expected)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        if value_head:
            with torch.no_grad():
                _, actual_value, _ = restored.forward_with_value(tokens, numbers)
            assert torch.equal(actual_value, expected_value)
        else:
            assert not any(key.startswith("value_head.") for key in model.state_dict())
            with pytest.raises(ValueError, match="value_head=True"):
                restored.forward_with_value(tokens, numbers)


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


@pytest.mark.parametrize("arch", ["gru", "mlp"])
def test_value_only_update_preserves_frozen_policy(arch):
    config = {
        "arch": arch, "vocab_size": 20, "token_len": 3, "d_model": 8,
        "hidden_size": 16, "numeric_size": 8,
    }
    torch.manual_seed(17)
    policy = PokeDecision(PokeDecisionConfig(**config), torch.zeros(48), torch.ones(48))
    torch.manual_seed(17)
    model = PokeDecision(
        PokeDecisionConfig(**config, value_head=True), torch.zeros(48), torch.ones(48)
    )
    # Compare like-for-like frozen execution: CPU GRU kernels may differ by
    # rounding when one copy retains trainable recurrent parameters.
    policy.requires_grad_(False)
    for name, value in policy.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("value_head."))
    optimizer = torch.optim.SGD(model.value_head.parameters(), lr=0.1)
    tokens, numbers = torch.randint(20, (2, 3, 3)), torch.randn(2, 3, 48)
    with torch.no_grad():
        expected, _ = policy(tokens, numbers)
        actual, _ = model(tokens, numbers)
    assert torch.equal(actual, expected)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    _, value_logits, _ = model.forward_with_value(tokens, numbers)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        value_logits, torch.ones_like(value_logits)
    )
    loss.backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith("value_head."):
            assert torch.equal(value, before[name])
    assert not torch.equal(model.value_head.weight, before["value_head.weight"])
    with torch.no_grad():
        actual, _ = model(tokens, numbers)
    assert torch.equal(actual, expected)
