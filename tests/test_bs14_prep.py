"""BS1.4 preparation: bundle verification (config + team files bound), fresh-heldout hashes.

All pools/configs are temporary; no battle runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bs14_manifest
import evaluate_w5
from evaluate_w5 import verify_release

TEAM = (ROOT / "data/teams_w5/dev/gen9ou/hl_05_26-0001.gen9ou_team").read_text()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A temporary frozen bundle: config, pool of 3 teams, manifest (code checks stubbed out)."""
    monkeypatch.setattr(bs14_manifest, "ROOT", tmp_path)
    monkeypatch.setattr(bs14_manifest.M, "ROOT", tmp_path)
    monkeypatch.setattr(bs14_manifest.M, "verify", lambda path: [])
    pool = tmp_path / "pool" / "heldout"
    (pool / "gen9ou").mkdir(parents=True)
    for i in range(3):
        (pool / "gen9ou" / f"t{i}.gen9ou_team").write_text(TEAM + f"\n# {i}\n")
    (pool / "gen9ou" / "manifest.json").write_text(json.dumps({"role": "heldout"}))
    config = {"pool": "pool/heldout", "battles": 2, "reference_arm": "C",
              "arms": {"C": {"greedy": True}}, "opponents": {},
              "run": {"final": True, "release_manifest": str(tmp_path / "manifest.json")}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(config))
    manifest = {
        "benchmark_config": {"file": "cfg.yaml", "sha256": bs14_manifest.M.sha(tmp_path / "cfg.yaml"),
                             "parsed": config},
        "team_pool": {"path": "pool/heldout", "manifest_file": "pool/heldout/gen9ou/manifest.json",
                      "manifest_sha256": bs14_manifest.M.sha(pool / "gen9ou/manifest.json"),
                      "files": bs14_manifest.pool_snapshot(pool)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, pool, config


def _claims(pool: Path):
    return [p for p in pool.iterdir() if p.name != "gen9ou"]


def test_the_intact_bundle_verifies(bundle):
    root, _pool, config = bundle
    verify_release(config, root / "cfg.yaml")


def test_a_changed_config_is_refused_before_any_claim(bundle):
    root, pool, config = bundle
    (root / "cfg.yaml").write_text(yaml.safe_dump({**config, "battles": 4}))
    with pytest.raises(RuntimeError, match="config: passed file"):
        verify_release(config, root / "cfg.yaml")
    assert _claims(pool) == []


def test_another_config_file_is_refused(bundle):
    root, pool, config = bundle
    other = root / "other.yaml"
    other.write_text((root / "cfg.yaml").read_text())
    with pytest.raises(RuntimeError, match="is not the frozen"):
        verify_release(config, other)
    assert _claims(pool) == []


def test_a_config_pointing_to_another_pool_is_refused(bundle):
    root, pool, config = bundle
    other = root / "pool2" / "heldout"
    (other / "gen9ou").mkdir(parents=True)
    changed = {**config, "pool": "pool2/heldout"}
    (root / "cfg.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match="config: passed file"):
        verify_release(changed, root / "cfg.yaml")
    assert _claims(pool) == [] and _claims(other) == []


def test_a_cli_pool_override_is_refused(bundle):
    root, _pool, config = bundle
    other = root / "pool2" / "heldout"
    (other / "gen9ou").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="is not the frozen pool"):
        verify_release(config, root / "cfg.yaml", "pool2/heldout")


def test_an_in_memory_config_that_differs_from_the_file_is_refused(bundle):
    root, _pool, config = bundle
    with pytest.raises(RuntimeError, match="in-memory config differs"):
        verify_release({**config, "battles": 100}, root / "cfg.yaml")


@pytest.mark.parametrize("edit, message", [
    (lambda p: (p / "gen9ou/t1.gen9ou_team").write_text("tampered"), "changed .*t1"),
    (lambda p: (p / "gen9ou/t9.gen9ou_team").write_text(TEAM), "extra file .*t9"),
    (lambda p: (p / "gen9ou/t2.gen9ou_team").unlink(), "missing .*t2"),
    (lambda p: (p / "gen9ou/manifest.json").write_text("{}"), "changed .*manifest.json"),
])
def test_team_file_changes_are_refused_before_any_claim(bundle, edit, message):
    root, pool, config = bundle
    edit(pool)
    with pytest.raises(RuntimeError, match=message):
        verify_release(config, root / "cfg.yaml")
    assert _claims(pool) == []


def test_the_lane_verifies_before_it_claims(bundle, monkeypatch):
    root, pool, _config = bundle
    (pool / "gen9ou/t0.gen9ou_team").write_text("tampered")
    monkeypatch.setattr(evaluate_w5, "claim_pool", lambda *a, **k: pytest.fail("claimed"))
    args = argparse.Namespace(
        config=str(root / "cfg.yaml"), pool=None, battles=None, arm=None, lanes=2, lane=0,
        pairs=[0, 1], out=str(root / "out.json"), final=True)
    with pytest.raises(RuntimeError, match="changed .*t0"):
        asyncio.run(evaluate_w5.evaluate(args))
    assert _claims(pool) == []


def test_the_driver_verifies_before_launching_any_lane(bundle, monkeypatch):
    import bs1_2_run

    root, pool, _config = bundle
    (pool / "gen9ou/t0.gen9ou_team").write_text("tampered")
    launched = []
    monkeypatch.setattr(bs1_2_run.subprocess, "Popen", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(sys, "argv", ["bs1_2_run.py", "--config", str(root / "cfg.yaml")])
    with pytest.raises(RuntimeError, match="changed .*t0"):
        bs1_2_run.main()
    assert launched == [] and _claims(pool) == []


def test_canonical_hash_catches_a_permuted_team_that_the_ordered_hash_misses():
    import build_teamset as b

    blocks = TEAM.strip().split("\n\n")
    assert len(blocks) == 6
    permuted = "\n\n".join(reversed(blocks)) + "\n"
    assert b.normalized_sha256(permuted) != b.normalized_sha256(TEAM)
    assert b.canonical_team_sha256(permuted) == b.canonical_team_sha256(TEAM)


def test_the_real_new_pool_is_unclaimed():
    pool = ROOT / "data/teams_w5_bs14/heldout"
    assert pool.is_dir()
    assert sorted(p.name for p in pool.iterdir()) == ["gen9ou"]
    assert len(list((pool / "gen9ou").glob("*.gen9ou_team"))) == 200
