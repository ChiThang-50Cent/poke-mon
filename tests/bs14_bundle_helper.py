"""Temporary frozen bundle for the pool-claim tests (shared by pytest and the old-tree subprocess)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def make_bundle(root: Path, bm, teams: int = 3, team_text: str = "team") -> tuple[Path, Path, dict]:
    """Create <root>/pool/heldout (teams + manifest), cfg.yaml and manifest.json.

    `bm` is the bs14_manifest module (old or new) whose ROOT/M.ROOT/M.verify the caller already
    pointed at `root` (the code/model checks are not under test here). Paths are absolute."""
    pool = root / "pool" / "heldout"
    (pool / "gen9ou").mkdir(parents=True)
    for i in range(teams):
        (pool / "gen9ou" / f"t{i}.gen9ou_team").write_text(f"{team_text} {i}\n")
    (pool / "gen9ou" / "manifest.json").write_text(json.dumps({"role": "heldout"}))
    config = {"pool": str(pool), "battles": 2, "reference_arm": "C",
              "arms": {"C": {"greedy": True}}, "opponents": {},
              "run": {"final": True, "release_manifest": str(root / "manifest.json")}}
    cfg = root / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(config))
    manifest = {
        "benchmark_config": {"file": "cfg.yaml", "sha256": bm.M.sha(cfg), "parsed": config},
        "team_pool": {"path": str(pool), "manifest_file": str(pool / "gen9ou/manifest.json"),
                      "manifest_sha256": bm.M.sha(pool / "gen9ou/manifest.json"),
                      "files": bm.pool_snapshot(pool)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return pool, cfg, config
