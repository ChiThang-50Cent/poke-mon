"""BS1.4 evaluation bundle on RC4 code: everything the RC manifest froze + the exact config file
and the exact fresh heldout team set.

  bs14_manifest.py [--config CFG] [--out MANIFEST]   write the bundle manifest (defaults:
                                                      configs/w5_bs14.yaml, its run.release_manifest)
  verify_bundle(manifest, config_path, config=None, pool=None, claim_config=None) -> problems

A final run refuses the claim unless the config file actually passed equals the manifest's (path +
sha256 + parsed content, pool included) and the pool's team files match the frozen set exactly
(no missing, changed or extra file; team manifest sha256 equal), on top of every RC check."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import bs_rc1_manifest as M
from pool_claim import CLAIM_STARTED, WINDOW_RE, staging_dir

CONFIG = "configs/w5_bs14.yaml"
SCRIPTS = [*M.SCRIPTS_USED, "bs14_manifest.py", "build_teamset.py", "pool_claim.py",
           "bs_watchdog.py"]
LABEL = "BS1.4 evaluation bundle on RC4 code 17e609a"
CODE_COMMIT = "17e609ab902e73586a5c6147644a0d25f9961274"


def pool_snapshot(pool: Path) -> dict:
    """Relative path -> sha256 of every file under <pool>/gen9ou (team files + manifest.json)."""
    base = pool / "gen9ou"
    return {str(p.relative_to(ROOT)): M.sha(p) for p in sorted(base.rglob("*")) if p.is_file()}


def _claim_problem(path: Path, claim_config: dict | None) -> str | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return f"team_pool: claim {path.name} is not complete valid JSON ({exc})"
    if not isinstance(data, dict) or "config" not in data:
        return f"team_pool: claim {path.name} has no config"
    if claim_config is not None and data["config"] != claim_config:
        return f"team_pool: claim {path.name} config differs from the effective config"
    return None


def pool_problems(frozen: dict, pool: Path, claim_config: dict | None = None) -> list[str]:
    """Team files exact; pool root = gen9ou/, an EMPTY final.lock and valid claim artifacts only
    (no generic dotfile or `.final*` whitelist); any orphan in the claim staging dir is refused."""
    problems = []
    if (ROOT / frozen["path"]).resolve() != pool.resolve():
        problems.append(f"team_pool: pool {pool} is not the frozen pool {frozen['path']}")
        return problems
    actual = pool_snapshot(pool)
    for name, digest in frozen["files"].items():
        if name not in actual:
            problems.append(f"team_pool: missing {name}")
        elif actual[name] != digest:
            problems.append(f"team_pool: changed {name}")
    for name in sorted(set(actual) - set(frozen["files"])):
        problems.append(f"team_pool: extra file {name}")
    team = pool / "gen9ou" / "manifest.json"
    if not team.is_file() or M.sha(team) != frozen["manifest_sha256"]:
        problems.append("team_pool: gen9ou/manifest.json changed")
    for entry in sorted(pool.iterdir()):
        if entry.name == "gen9ou" and entry.is_dir():
            continue
        if entry.name == "final.lock" and entry.is_file():
            if entry.stat().st_size:
                problems.append("team_pool: final.lock is not empty")
        elif entry.is_file() and (entry.name == CLAIM_STARTED or WINDOW_RE.fullmatch(entry.name)):
            problem = _claim_problem(entry, claim_config)
            if problem:
                problems.append(problem)
        else:
            problems.append(f"team_pool: unexpected entry {entry.name}")
    staging = staging_dir(pool)
    if staging.exists():
        orphans = sorted(p.name for p in staging.iterdir()) if staging.is_dir() else [staging.name]
        if orphans:
            problems.append(f"team_pool: orphan claim temp file(s) in {staging}: {orphans}; a "
                            "claim crashed mid-write; manual review required (nothing deleted)")
    return problems


def verify_bundle(path: Path, config_path: Path | str, config: dict | None = None,
                  pool: Path | str | None = None, claim_config: dict | None = None) -> list[str]:
    manifest = json.loads(Path(path).read_text())
    problems = M.verify(Path(path))
    frozen_config = manifest["benchmark_config"]
    config_path = Path(config_path)
    if config_path.resolve() != (ROOT / frozen_config["file"]).resolve():
        problems.append(f"config: passed {config_path} is not the frozen {frozen_config['file']}")
    elif not config_path.is_file() or M.sha(config_path) != frozen_config["sha256"]:
        problems.append(f"config: passed file {config_path} changed")
    else:
        parsed = yaml.safe_load(config_path.read_text())
        if parsed != frozen_config["parsed"]:
            problems.append("config: parsed content differs from the manifest")
        if config is not None and config != parsed:
            problems.append("config: the in-memory config differs from the file passed")
    frozen_pool = manifest.get("team_pool")
    if not frozen_pool:
        problems.append("manifest has no team_pool section")
        return problems
    wanted = yaml.safe_load(config_path.read_text()).get("pool") if config_path.is_file() else None
    if wanted != frozen_pool["path"]:
        problems.append(f"team_pool: config pool {wanted} != frozen pool {frozen_pool['path']}")
    problems += pool_problems(
        frozen_pool, ROOT / (pool if pool is not None else frozen_pool["path"]),
        claim_config if claim_config is not None else config)
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config_file = Path(args.config)
    config_rel = M.rel((ROOT / config_file).resolve())
    config = yaml.safe_load((ROOT / config_rel).read_text())
    models = [*config["arms"].values(), *config["opponents"].values()]
    model_dirs = sorted({ROOT / m["model_dir"] for m in models if "model_dir" in m})
    previews = sorted({ROOT / m["preview_checkpoint"] for m in models if "preview_checkpoint" in m})
    search = next((m["search"] for m in config["arms"].values() if "search" in m), None)
    driver = ROOT / "src/pokeai/cf/sim_driver.js"
    pool = ROOT / config["pool"]
    status = M.run("git", "status", "--porcelain", "--untracked-files=no").splitlines()
    manifest = {
        "release_candidate": config["run"].get("label", LABEL),
        "code_commit": config["run"].get("code_commit", CODE_COMMIT),
        "git": {"commit": M.run("git", "rev-parse", "HEAD"), "uncommitted_tracked_files": status,
                "note": "files are hashed below regardless of commit state"},
        "c_checkpoint": {M.rel(p): M.sha(p) for d in model_dirs for p in sorted(d.iterdir())
                         if p.is_file()},
        "preview_checkpoint": {M.rel(p): M.sha(p) for p in previews},
        "tokenizer": {M.rel(p): M.sha(p) for p in
                      [*(d / "tokenizer.json" for d in model_dirs),
                       *sorted((ROOT / "third_party/metamon/metamon/tokenizer").glob("*.json"))]},
        "belief_sampler_data": ({M.rel(p): M.sha(p) for p in M.belief_data_files()}
                                if search else {}),
        "code_live_path": {M.rel(p): M.sha(p) for p in M.live_path_modules()},
        "scripts": {M.rel(ROOT / "scripts" / n): M.sha(ROOT / "scripts" / n) for n in SCRIPTS},
        "sim_driver_js": {M.rel(driver): M.sha(driver)},
        "showdown": M.showdown_state(),
        "benchmark_config": {
            "file": config_rel, "sha256": M.sha(ROOT / config_rel), "parsed": config,
            "frozen_search_parameters": None if search is None else {
                "D_s": search["deadline_s"], "M_s": search["margin_s"],
                "min_budget_s": search["min_budget_s"], "opp_model": search["opp_model"],
                "n_worlds": search["n_worlds"], "n_seeds": search["n_seeds"],
                "workers_per_lane": search["workers"], "lanes": config["run"]["lanes"],
                "search_seed": search["seed"], "schedule_seed": config["run"]["seed"],
                "fail_fast": search["fail_fast"]},
        },
        "team_pool": {
            "path": config["pool"],
            "manifest_file": M.rel(pool / "gen9ou/manifest.json"),
            "manifest_sha256": M.sha(pool / "gen9ou/manifest.json"),
            "files": pool_snapshot(pool),
        },
        "uv_lock_sha256": M.sha(ROOT / "uv.lock"),
        "versions": {**M.versions(), "platform": platform.platform()},
    }
    out = ROOT / (args.out or config["run"]["release_manifest"])
    out.write_text(json.dumps(manifest, indent=1))
    print("wrote", out, {k: len(v) for k, v in manifest.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
