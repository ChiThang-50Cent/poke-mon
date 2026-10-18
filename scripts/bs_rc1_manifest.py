"""Freeze manifest of the C + belief search release candidate (tag `w5-bs-rc2`).

`verify(path)` re-hashes everything the manifest froze; a final heldout run refuses to claim the
pool unless it returns no problems."""

from __future__ import annotations

import builtins
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
SCRIPTS_USED = [
    "evaluate_w5.py", "evaluate_w4_5.py", "bs1_2_run.py", "bs1_2_report.py", "bs1_1_report.py",
    "bs1_2b_qa.py", "bs1_2b_golden.py", "bs_rc1_manifest.py",
]
CONFIG = "configs/w5_bs_rc1.yaml"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run(*cmd: str, cwd: Path = ROOT) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()


def live_path_modules() -> list[Path]:
    """Every src/pokeai module imported by the live path (player, search layer, workers)."""
    import importlib

    for name in (
        "pokeai.agents.neural_player", "pokeai.agents.live_search", "pokeai.env.runner",
        "pokeai.cf.recon_search", "pokeai.cf.recon_world", "pokeai.cf.recon_match",
        "pokeai.cf.offline", "pokeai.cf.branching", "pokeai.cf.cf1", "pokeai.cf.belief",
        "pokeai.cf.rollouts", "pokeai.cf.sim", "pokeai.cf.restore_check",
    ):
        importlib.import_module(name)
    files = []
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if f and Path(f).resolve().is_relative_to(ROOT / "src" / "pokeai"):
            files.append(Path(f).resolve())
    return sorted(set(files))


def belief_data_files() -> list[Path]:
    """The usage-statistics files the belief sampler really opens (recorded, not guessed)."""
    opened: list[Path] = []
    real_open = builtins.open

    def spy(file, *a, **k):
        if isinstance(file, (str, Path)) and str(file).endswith((".json", ".npz", ".pkl", ".csv")):
            opened.append(Path(file))
        return real_open(file, *a, **k)

    builtins.open = spy
    try:
        from pokeai.cf.belief import UsagePrior

        UsagePrior()
    finally:
        builtins.open = real_open
    return sorted({p.resolve() for p in opened if p.exists() and "usage-stats" in str(p)})


FILE_SECTIONS = ("c_checkpoint", "preview_checkpoint", "tokenizer", "belief_sampler_data",
                 "code_live_path", "scripts", "sim_driver_js")


def showdown_state() -> dict:
    showdown = ROOT / "third_party/metamon/server/pokemon-showdown"
    dist_files = sorted((showdown / "dist/sim").rglob("*.js"))
    tree = hashlib.sha256()
    for f in dist_files:
        tree.update(str(f.relative_to(showdown)).encode() + b"\0" + f.read_bytes())
    return {
        "repo_commit": run("git", "rev-parse", "HEAD", cwd=showdown),
        "dist_sim_files": len(dist_files), "dist_sim_tree_sha256": tree.hexdigest(),
        "metamon_commit": run("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/metamon"),
    }


def versions() -> dict:
    import numpy
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "node": run("node", "--version")}


def verify(path: Path) -> list[str]:
    """Every mismatch between the frozen manifest and the current tree (empty list = frozen)."""
    manifest = json.loads(path.read_text())
    problems = []
    if manifest["git"]["uncommitted_tracked_files"]:
        problems.append("manifest was generated from a dirty tree")
    for section in FILE_SECTIONS:
        for name, digest in manifest[section].items():
            file = ROOT / name
            if not file.is_file():
                problems.append(f"{section}: missing {name}")
            elif sha(file) != digest:
                problems.append(f"{section}: changed {name}")
    live = {rel(p) for p in live_path_modules()}
    for name in sorted(live - set(manifest["code_live_path"])):
        problems.append(f"code_live_path: unfrozen live module {name}")
    config = manifest["benchmark_config"]
    if sha(ROOT / config["file"]) != config["sha256"]:
        problems.append(f"benchmark_config: changed {config['file']}")
    if sha(ROOT / "uv.lock") != manifest["uv_lock_sha256"]:
        problems.append("uv.lock changed")
    for key, value in showdown_state().items():
        if manifest["showdown"][key] != value:
            problems.append(f"showdown: {key} changed")
    for key, value in versions().items():
        if manifest["versions"][key] != value:
            problems.append(f"versions: {key} {manifest['versions'][key]} -> {value}")
    return problems


def main() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    search = config["arms"]["C+search"]["search"]
    c_dir = ROOT / config["arms"]["C+search"]["model_dir"]
    preview = ROOT / config["arms"]["C+search"]["preview_checkpoint"]
    code = live_path_modules()
    scripts = [ROOT / "scripts" / n for n in SCRIPTS_USED]
    driver = ROOT / "src/pokeai/cf/sim_driver.js"

    status = run("git", "status", "--porcelain", "--untracked-files=no").splitlines()
    manifest = {
        "release_candidate": "RC2 (C + belief search; tag w5-bs-rc2)",
        "git": {"commit": run("git", "rev-parse", "HEAD"), "uncommitted_tracked_files": status,
                "note": "files are hashed below regardless of commit state"},
        "c_checkpoint": {rel(p): sha(p) for p in sorted(c_dir.iterdir()) if p.is_file()},
        "preview_checkpoint": {rel(preview): sha(preview)},
        "tokenizer": {rel(p): sha(p) for p in
                      [c_dir / "tokenizer.json",
                       *sorted((ROOT / "third_party/metamon/metamon/tokenizer").glob("*.json"))]},
        "belief_sampler_data": {rel(p): sha(p) for p in belief_data_files()},
        "code_live_path": {rel(p): sha(p) for p in code},
        "scripts": {rel(p): sha(p) for p in scripts},
        "sim_driver_js": {rel(driver): sha(driver)},
        "showdown": showdown_state(),
        "benchmark_config": {
            "file": CONFIG, "sha256": sha(ROOT / CONFIG), "parsed": config,
            "frozen_search_parameters": {
                "D_s": search["deadline_s"], "M_s": search["margin_s"],
                "min_budget_s": search["min_budget_s"], "opp_model": search["opp_model"],
                "n_worlds": search["n_worlds"], "n_seeds": search["n_seeds"],
                "workers_per_lane": search["workers"], "lanes": config["run"]["lanes"],
                "search_seed": search["seed"], "schedule_seed": config["run"]["seed"],
                "fail_fast": search["fail_fast"]},
        },
        "uv_lock_sha256": sha(ROOT / "uv.lock"),
        "versions": {**versions(), "platform": platform.platform()},
    }
    out = ROOT / "reports/w5_bs_rc1_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print("wrote", out, {k: len(v) for k, v in manifest.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
