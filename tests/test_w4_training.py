"""Fast synthetic W4 initialization and provenance contracts."""

import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from pokeai.models.pokedecision import PokeDecision, PokeDecisionConfig, save_checkpoint
from pokeai.training.train_bc import checkpoint_metadata, initialize_model, train_arch


@pytest.fixture
def inputs(tmp_path):
    encoder = {"vocab_size": 2542, "token_len": 87}
    tokenizer = tmp_path / "encoder.json"
    tokenizer.write_text(json.dumps(encoder))
    config = {
        "seed": 20261016, "data_root": str(tmp_path), "dataset_version": "w4",
        "train_tier": "50000", "expected_parameters": 1299789,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "model": {"d_model": 96, "hidden_size": 384, "numeric_size": 128},
        "train": {"device": "cpu", "batch_size": 1, "learning_rate": 0.0005,
                  "weight_decay": 0.0001, "epochs": 1, "patience": 3,
                  "grad_clip": 1.0},
    }
    stats = {"mean": [7.0] * 48, "std": [3.0] * 48}
    return config, stats, encoder


def test_finetune_restores_every_v0_weight_and_normalization(inputs, tmp_path):
    config, stats, encoder = inputs
    torch.manual_seed(13)
    original = PokeDecision(
        PokeDecisionConfig(**(config["model"] | encoder)),
        torch.arange(48).float(), torch.arange(48).float() + 1,
    )
    save_checkpoint(original, tmp_path / "best.pt")
    config["init_checkpoint"] = str(tmp_path / "best.pt")
    restored = initialize_model(config, "gru", stats, encoder)
    assert restored.config == original.config
    assert set(restored.state_dict()) == set(original.state_dict())
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key]), key


def test_scratch_uses_full_train_statistics_and_exact_parameter_count(inputs):
    config, stats, encoder = inputs
    model = initialize_model(config, "gru", stats, encoder)
    assert sum(p.numel() for p in model.parameters()) == 1299789
    assert model.feature_mean.tolist() == stats["mean"]
    assert model.feature_std.tolist() == stats["std"]
    config["expected_parameters"] += 1
    with pytest.raises(ValueError, match="parameters"):
        initialize_model(config, "gru", stats, encoder)


def test_tokenizer_hash_mismatch_raises(inputs, tmp_path):
    config, _, _ = inputs
    config["tokenizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Tokenizer hash mismatch"):
        checkpoint_metadata(config, tmp_path / "encoder.json")


def test_finetune_tokenizer_mismatch_raises(inputs, tmp_path):
    config, _, _ = inputs
    config["init_checkpoint"] = str(tmp_path / "best.pt")
    (tmp_path / "tokenizer.json").write_text("different tokenizer")
    with pytest.raises(ValueError, match="Initialization tokenizer hash mismatch"):
        checkpoint_metadata(config, tmp_path / "encoder.json")


def test_run_records_complete_metadata(inputs, tmp_path):
    import numpy as np
    import yaml

    config, stats, encoder = inputs
    arrays = {
        "tokens": np.zeros((2, 87), dtype=np.int64),
        "numbers": np.zeros((2, 48), dtype=np.float32),
        "masks": np.ones((2, 13), dtype=bool),
        "actions": np.zeros(2, dtype=np.int64),
        "valid": np.ones(2, dtype=bool), "offsets": np.array([0, 2]),
    }
    result = train_arch(config, "gru", arrays, arrays, stats, encoder, tmp_path / "run")
    recorded = yaml.safe_load((tmp_path / "run/config.yaml").read_text())
    required = {"git_commit", "git_dirty", "dataset_version", "train_tier",
                "tokenizer_sha256", "observation_space_version", "init_checkpoint",
                "init_checkpoint_sha256", "source_sha256"}
    assert required <= recorded.keys()
    assert recorded["dataset_version"] == "w4"
    assert recorded["train_tier"] == "50000"
    assert recorded["observation_space_version"] == "DefaultObservationSpace@metamon-0a00a75"
    assert recorded["init_checkpoint"] is None
    assert recorded["init_checkpoint_sha256"] is None
    assert isinstance(recorded["git_dirty"], bool)
    assert len(recorded["git_commit"]) == 40
    assert len(recorded["source_sha256"]) == 64
    assert recorded["model"] == asdict(PokeDecisionConfig(**(config["model"] | encoder)))
    assert result["stopping_epoch"] == 1
    assert result["train_valid_decisions"] == 2
