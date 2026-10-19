"""Dress-rehearsal bundle for the final-run machinery (BS1.4-P): runs/rehearsal/<name>/ (gitignored).

  bs_rehearsal_prep.py NAME [--layout 1x4|2x4]

Creates a disposable pool of the 200 DEV teams marked heldout + `"rehearsal": true`, a config copied
from configs/w5_bs14.yaml with ONLY pool / out_prefix (w5-bs14p-NAME) / fail_dir / release_manifest
(inside the rehearsal dir) / battles / block_pairs / schedule seed changed, and that config's bundle
manifest (same tool as the real bundle; needs a clean tree, config and pool are gitignored).

Layout 1x4 (contract): battles 8, blocks [[0,1],[1,2],[2,3],[3,4]] = 1 pair per block. The runner deals
pair k to lane k % lanes, so each block leaves ONE lane with no pair (its window is claimed, 0 battles).
Layout 2x4: battles 16, blocks of 2 pairs, both lanes busy in every block (8 claims, 96 battles).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = "configs/w5_bs14.yaml"
DEV_POOL = "data/teams_w5/dev"
SCHEDULE_SEED = 20262001
LAYOUTS = {
    "1x4": (8, [[0, 1], [1, 2], [2, 3], [3, 4]]),
    "2x4": (16, [[0, 2], [2, 4], [4, 6], [6, 8]]),
}


def prepare(name: str, root: Path = ROOT, layout: str = "1x4", make_manifest: bool = True) -> Path:
    battles, blocks = LAYOUTS[layout]
    base = root / "runs/rehearsal" / name
    if base.exists():
        raise FileExistsError(f"refusing to reuse a rehearsal directory: {base}")
    pool = base / "heldout"
    (pool / "gen9ou").mkdir(parents=True)
    dev = root / DEV_POOL / "gen9ou"
    teams = sorted(dev.glob("*.gen9ou_team"))
    for team in teams:
        shutil.copyfile(team, pool / "gen9ou" / team.name)
    manifest = json.loads((dev / "manifest.json").read_text())
    manifest.update({"role": "heldout", "rehearsal": True})
    (pool / "gen9ou/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    config = yaml.safe_load((root / BASE_CONFIG).read_text())
    rel = base.relative_to(root)
    config["pool"] = str(rel / "heldout")
    config["battles"] = battles
    config["arms"]["C+search"]["search"]["fail_dir"] = str(rel / "new-class")
    run = config["run"]
    run["out_prefix"] = f"w5-bs14p-{name}"
    run["release_manifest"] = str(rel / "manifest.json")
    run["block_pairs"] = blocks
    run["seed"] = SCHEDULE_SEED
    cfg = base / "config.yaml"
    cfg.write_text(f"# REHEARSAL config {name} (layout {layout}); copy of {BASE_CONFIG} with "
                   "pool/out_prefix/fail_dir/release_manifest/battles/block_pairs/seed changed\n"
                   + yaml.safe_dump(config, sort_keys=False))
    if make_manifest:
        subprocess.run([sys.executable, "scripts/bs14_manifest.py", "--config", str(rel / "config.yaml"),
                        "--out", str(rel / "manifest.json")], cwd=root, check=True)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="1x4")
    args = ap.parse_args()
    cfg = prepare(args.name, layout=args.layout)
    print(f"rehearsal ready: python scripts/bs1_2_run.py --config {cfg.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
