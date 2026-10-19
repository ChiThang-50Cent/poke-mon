"""Synthetic W5 pool isolation, one-shot holdout and neural pairing checks."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
builder = importlib.import_module("build_teamset")
benchmark = importlib.import_module("evaluate_w5")


def test_pool_selection_deterministic_disjoint(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    paths = []
    for i in range(220):
        path = source / f"{i:04d}.gen9ou_team"
        path.write_text(str(i))
        paths.append(path)
    old = tmp_path / "old"
    old.mkdir()
    (old / "old.gen9ou_team").write_text("219")
    monkeypatch.setattr(builder, "OUTPUT", old)
    monkeypatch.setattr(builder, "normalized_sha256", lambda text: "hash-" + text)
    monkeypatch.setattr(builder, "get_metamon_teams", lambda fmt, src: SimpleNamespace(
        team_files=paths if src == "hl_05_26" else [paths[-2]],
    ))
    excluded = set(builder.random.Random(20261016).sample(paths, 200))
    excluded.update(paths[-2:])
    rejected_path = next(
        p for p in builder.seeded_candidates(paths, 42) if p not in excluded
    )
    monkeypatch.setattr(builder, "validate_team", lambda path: (
        path.read_text(), ["synthetic rejection"] if path == rejected_path else [],
    ))
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **kw: "pinned\n")
    sizes = {"collect": 5, "dev": 3, "heldout": 3}
    for output in (tmp_path / "a", tmp_path / "b"):
        builder.build_w5_pools(output, 42, sizes)
    hashes = []
    for role, size in sizes.items():
        a = tmp_path / "a" / role / "gen9ou/manifest.json"
        b = tmp_path / "b" / role / "gen9ou/manifest.json"
        assert a.read_bytes() == b.read_bytes()
        manifest = json.loads(a.read_text())
        values = {entry["sha256"] for entry in manifest["teams"]}
        assert len(values) == size
        assert "hash-219" not in values and "hash-218" not in values
        assert all(not values & prior for prior in hashes)
        assert manifest["disjointness"]["overlap_with_w4_and_competitive"] == 0
        if role == "collect":
            assert manifest["validation"]["rejected_validation_count"] == 1
        hashes.append(values)
    assert builder.seeded_candidates(paths, 42) == builder.seeded_candidates(paths[::-1], 42)


def test_heldout_requires_final_and_one_shot(tmp_path):
    pool = tmp_path / "heldout"
    (pool / "gen9ou").mkdir(parents=True)
    (pool / "gen9ou/manifest.json").write_text('{"role":"heldout"}')
    from pool_claim import claim_unlocked, pool_lock

    with pool_lock(pool) as lock:
        with pytest.raises(ValueError, match="--final"):
            claim_unlocked(lock, pool, False, {})
        assert not (pool / "final.started.json").exists()
        claim_unlocked(lock, pool, True, {"arms": {"C": {}}})
        with pytest.raises(FileExistsError):
            claim_unlocked(lock, pool, True, {})


def test_neural_opponent_independent_configuration(tmp_path):
    config = benchmark.run_config(
        {"model_dir": "C", "preview_checkpoint": "preview.pt", "greedy": True},
        {"model_dir": "B", "greedy": True},
        pool=tmp_path, battles=20, concurrent=2, seed=7,
    )
    assert config.player == "neural:C" and config.opponent == "neural:B"
    assert config.sample is False and config.opponent_sample is False
    assert config.team_preview_checkpoint == "preview.pt"
    assert config.opponent_team_preview_disabled is True
    assert config.swap_team_pairs and config.alternate_challenger
    from pokeai.env.runner import _player_type

    player = _player_type(config.player, config)
    opponent = _player_type(config.opponent, config)
    from pokeai.agents.neural_player import NeuralPlayer

    assert player.func is NeuralPlayer and opponent.func is NeuralPlayer
    assert player.keywords["model_dir"] == "C"
    assert opponent.keywords["model_dir"] == "B"


def test_default_w4_manifest_byte_compatibility(tmp_path, monkeypatch):
    """Freeze source selection/export order without rerunning server validation."""
    original = builder.OUTPUT
    expected = (original / "manifest.json").read_bytes()
    valid_exports = {
        path.read_text() for path in original.glob("*.gen9ou_team")
    }
    monkeypatch.setattr(builder, "OUTPUT", tmp_path / "rebuilt")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "validate_team", lambda path: (
        path.read_text(), [] if path.read_text() in valid_exports else ["rejected"],
    ))
    builder.build_teamset(["competitive", "hl_05_26"], 200, 20261016)
    assert (builder.OUTPUT / "manifest.json").read_bytes() == expected
