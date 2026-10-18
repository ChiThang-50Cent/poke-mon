"""Fetch, deterministically sample, and validate a fixed Gen9 OU team pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / ".cache" / "metamon"))

from metamon.env.wrappers import TeamSet, get_metamon_teams

SHOWDOWN = ROOT / "third_party/metamon/server/pokemon-showdown/pokemon-showdown"
OUTPUT = ROOT / "data/teams/gen9ou"


def validate_team(path: Path) -> tuple[str, list[str]]:
    """Return the export text and legality errors from the local server's rules."""
    text = path.read_text(encoding="utf-8")
    result = subprocess.run(
        ["node", str(SHOWDOWN), "validate-team", "gen9ou"],
        input=text,
        text=True,
        capture_output=True,
        timeout=60,
        cwd=SHOWDOWN.parent,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"Showdown validator failed for {path}: {result.stderr or result.stdout}"
        )
    if result.returncode:
        errors = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        if not errors:
            raise RuntimeError(f"Showdown rejected {path} without an error message")
        return text, errors
    species = [mon.species or mon.nickname for mon in TeamSet.parse_showdown_team(text)]
    if len(species) != 6 or not all(species):
        return text, ["Expected six parseable species"]
    return text, []


def build_teamset(sources: list[str], sample: int, seed: int) -> None:
    """Validate all competitive teams and a seeded sample of other source pools."""
    rng = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    for source in sources:
        teamset = get_metamon_teams("gen9ou", source)
        paths = sorted(Path(path) for path in teamset.team_files)
        if source != "competitive" and len(paths) > sample:
            paths = sorted(rng.sample(paths, sample))
        selected.extend((source, path) for path in paths)
        print(f"{source}: selected {len(paths)} of {len(teamset.team_files)} teams")
    if not selected:
        raise RuntimeError("No source teams were selected")

    # Each CLI process loads the dex; bound parallelism to avoid memory spikes.
    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(validate_team, [path for _, path in selected]))

    kept: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    manifest: list[dict[str, object]] = []
    exports: dict[str, str] = {}
    for (source, _), (text, errors) in zip(selected, results, strict=True):
        if errors:
            rejected[source] += 1
            reasons.update(errors)
            continue
        kept[source] += 1
        filename = f"{source}-{kept[source]:03d}.gen9ou_team"
        exports[filename] = text
        manifest.append(
            {
                "file": filename,
                "source": source,
                "species": [
                    mon.species or mon.nickname
                    for mon in TeamSet.parse_showdown_team(text)
                ],
            }
        )
    if not manifest:
        raise RuntimeError("All selected teams were rejected; existing pool unchanged")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, text in exports.items():
        (OUTPUT / filename).write_text(text, encoding="utf-8")
    # Remove stale output from previous runs with different sources/sample sizes.
    for path in OUTPUT.glob("*.gen9ou_team"):
        if path.name not in exports:
            path.unlink()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for source in sources:
        print(f"{source}: kept {kept[source]}, rejected {rejected[source]}")
    print(f"Total: kept {len(manifest)}, rejected {sum(rejected.values())}")
    if reasons:
        print("Rejection reasons:")
        for reason, count in reasons.most_common():
            print(f"  {count:4d}  {reason}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}/manifest.json (seed={seed})")


def normalized_sha256(text: str) -> str:
    """Hash the canonical packed representation, not export whitespace."""
    packed = TeamSet.join_team(TeamSet.parse_showdown_team(text))
    return hashlib.sha256(packed.encode()).hexdigest()


def seeded_candidates(paths: list[Path], seed: int) -> list[Path]:
    ordered = sorted(paths)
    random.Random(seed).shuffle(ordered)
    return ordered


def build_w5_pools(
    output: Path, seed: int, sizes: dict[str, int], workers: int = 4,
) -> None:
    """Keep the first N valid unique teams in one deterministic shuffled stream."""
    source = "hl_05_26"
    source_paths = sorted(
        Path(p) for p in get_metamon_teams("gen9ou", source).team_files
    )
    old = list(OUTPUT.glob("*.gen9ou_team"))
    competitive = get_metamon_teams("gen9ou", "competitive").team_files
    excluded = {
        normalized_sha256(Path(p).read_text())
        for p in [*old, *competitive]
    }
    # Explicitly exclude the entire original sampled source selection, including
    # any team that the W4 validator might have rejected.
    old_selection = sorted(random.Random(20261016).sample(source_paths, 200))
    excluded.update(normalized_sha256(p.read_text()) for p in old_selection)
    seen = set(excluded)
    candidates = iter(seeded_candidates(source_paths, seed))
    pool_hashes: dict[str, set[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for role, size in sizes.items():
            target = output / role / "gen9ou"
            if target.exists():
                raise FileExistsError(f"Refusing to replace team pool: {target}")
            entries, exports = [], {}
            rejected: Counter[str] = Counter()
            attempts = 0
            while len(entries) < size:
                batch = []
                while len(batch) < min(workers, size - len(entries)):
                    try:
                        path = next(candidates)
                    except StopIteration as exc:
                        raise RuntimeError("Not enough valid disjoint teams") from exc
                    digest = normalized_sha256(path.read_text())
                    if digest in seen:
                        rejected["excluded_or_duplicate_sha256"] += 1
                        continue
                    seen.add(digest)
                    batch.append((path, digest))
                results = executor.map(validate_team, [p for p, _ in batch])
                for (path, digest), (text, errors) in zip(batch, results, strict=True):
                    attempts += 1
                    if errors:
                        rejected.update(errors)
                        continue
                    filename = f"{source}-{len(entries) + 1:04d}.gen9ou_team"
                    exports[filename] = text
                    entries.append({
                        "file": filename, "source": source,
                        "source_file": path.name, "sha256": digest,
                    })
            hashes = {str(e["sha256"]) for e in entries}
            proof = {
                "normalization": "TeamSet.join_team(TeamSet.parse_showdown_team(text))",
                "excluded_sha256_count": len(excluded),
                "excluded_sha256_set_digest": hashlib.sha256(
                    "\n".join(sorted(excluded)).encode()
                ).hexdigest(),
                "overlap_with_w4_and_competitive": len(hashes & excluded),
                "overlap_with_previous_pools": {
                    name: len(hashes & values) for name, values in pool_hashes.items()
                },
            }
            pool_hashes[role] = hashes
            target.mkdir(parents=True)
            for filename, text in exports.items():
                (target / filename).write_text(text)
            manifest = {
                "schema_version": 1, "role": role, "seed": seed,
                "source": source, "source_count": len(source_paths),
                "selection": "single seeded shuffle; first N valid unique teams",
                "pool_order": list(sizes), "requested": size,
                "validation": {
                    "command": "node pokemon-showdown validate-team gen9ou",
                    "server_git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=SHOWDOWN.parent, text=True
                    ).strip(),
                    "attempted": attempts, "rejected_reasons": dict(rejected),
                    "rejected_validation_count": attempts - size,
                },
                "disjointness": proof, "teams": entries,
            }
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Wrote {role}: {size} validated teams", flush=True)


def canonical_team_sha256(text: str) -> str:
    """Order-independent team hash: sorted multiset of per-Pokemon normalized set hashes.

    Catches a team whose six sets are permuted (the ordered packed hash differs)."""
    mons = TeamSet.parse_showdown_team(text)
    per_mon = sorted(
        hashlib.sha256(TeamSet.join_team([mon]).encode()).hexdigest() for mon in mons
    )
    return hashlib.sha256("\n".join(per_mon).encode()).hexdigest()


def team_files_under(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*_team") if p.is_file())


def hash_pool(paths: list[Path]) -> dict[str, set[str]]:
    texts = [p.read_text() for p in paths]
    return {
        "files": len(paths),
        "ordered": {normalized_sha256(t) for t in texts},
        "canonical": {canonical_team_sha256(t) for t in texts},
    }


def discover_team_dirs(root: Path = ROOT) -> list[Path]:
    """Every directory under data/ holding team files, plus every team dir named by a config."""
    import yaml

    found = {p.parent for p in (root / "data").rglob("*_team") if p.is_file()}
    for config in sorted((root / "configs").glob("*.yaml")):
        data = yaml.safe_load(config.read_text()) or {}
        for key in ("pool", "teams"):
            value = data.get(key)
            if isinstance(value, str) and (root / value).is_dir():
                found.update(p.parent for p in (root / value).rglob("*_team"))
    return sorted(found)


def legacy_excluded_paths() -> tuple[list[Path], list[Path], list[Path]]:
    """The exclusions build_w5_pools applies: W4 pool, competitive set, old sampled selection."""
    source_paths = sorted(Path(p) for p in get_metamon_teams("gen9ou", "hl_05_26").team_files)
    old = sorted(OUTPUT.glob("*.gen9ou_team"))
    competitive = sorted(Path(p) for p in get_metamon_teams("gen9ou", "competitive").team_files)
    old_selection = sorted(random.Random(20261016).sample(source_paths, 200))
    return old, competitive, old_selection


def build_fresh_heldout(
    output: Path, seed: int, exclude_dirs: list[Path], size: int = 200, workers: int = 4,
) -> dict:
    """Build ONLY a fresh heldout pool (no collect/dev rebuild). Every exclusion is loaded from
    disk; the ordered and the order-independent hash must both be disjoint from all of them."""
    source = "hl_05_26"
    target = output / "gen9ou"
    if target.exists():
        raise FileExistsError(f"Refusing to replace team pool: {target}")
    source_paths = sorted(Path(p) for p in get_metamon_teams("gen9ou", source).team_files)
    old, competitive, old_selection = legacy_excluded_paths()
    pools: dict[str, dict] = {}
    for directory in exclude_dirs:
        if not Path(directory).is_dir():
            raise FileNotFoundError(f"excluded directory does not exist: {directory}")
        pools[str(directory)] = hash_pool(team_files_under(Path(directory)))
    pools["competitive"] = hash_pool(competitive)
    pools["old_w4_selection_seed20261016"] = hash_pool(old_selection)
    pools["old_w4_pool_files"] = hash_pool(old)
    for name, pool in pools.items():
        if not pool["files"]:
            raise RuntimeError(f"excluded pool is empty (not loaded?): {name}")
    excl_ordered = set().union(*(p["ordered"] for p in pools.values()))
    excl_canon = set().union(*(p["canonical"] for p in pools.values()))
    seen_ordered, seen_canon = set(), set()
    rejected: Counter[str] = Counter()
    candidates = iter(seeded_candidates(source_paths, seed))
    entries, exports, drawn, validated = [], {}, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(entries) < size:
            batch = []
            while len(batch) < min(workers, size - len(entries)):
                try:
                    path = next(candidates)
                except StopIteration as exc:
                    raise RuntimeError("Not enough valid disjoint teams") from exc
                drawn += 1
                text = path.read_text()
                ordered, canon = normalized_sha256(text), canonical_team_sha256(text)
                if ordered in excl_ordered:
                    rejected["excluded:ordered_sha256"] += 1
                elif canon in excl_canon:
                    rejected["excluded:canonical_sha256_only"] += 1
                elif ordered in seen_ordered:
                    rejected["duplicate:ordered_sha256"] += 1
                elif canon in seen_canon:
                    rejected["duplicate:canonical_sha256_only"] += 1
                else:
                    seen_ordered.add(ordered)
                    seen_canon.add(canon)
                    batch.append((path, ordered, canon))
            results = executor.map(validate_team, [p for p, _, _ in batch])
            for (path, ordered, canon), (text, errors) in zip(batch, results, strict=True):
                validated += 1
                if errors:
                    rejected.update(f"validator:{e}" for e in errors)
                    rejected["validator:teams"] += 1
                    continue
                filename = f"{source}-{len(entries) + 1:04d}.gen9ou_team"
                exports[filename] = text
                entries.append({"file": filename, "source": source, "source_file": path.name,
                                "sha256": ordered, "canonical_sha256": canon})
    target.mkdir(parents=True)
    for filename, text in exports.items():
        (target / filename).write_text(text)
    # independent re-check from the files on disk
    new = hash_pool(team_files_under(target))
    intersections = {
        name: {"files": pool["files"],
               "ordered_sha256": len(new["ordered"] & pool["ordered"]),
               "canonical_sha256": len(new["canonical"] & pool["canonical"])}
        for name, pool in pools.items()
    }
    within = {"teams": new["files"], "distinct_ordered_sha256": len(new["ordered"]),
              "distinct_canonical_sha256": len(new["canonical"])}
    if (new["files"] != size or within["distinct_ordered_sha256"] != size
            or within["distinct_canonical_sha256"] != size
            or any(v["ordered_sha256"] or v["canonical_sha256"] for v in intersections.values())):
        raise RuntimeError(f"disjointness check failed: {intersections} {within}")
    manifest = {
        "schema_version": 1, "role": "heldout", "seed": seed,
        "source": source, "source_count": len(source_paths),
        "selection": "single seeded shuffle; first N valid unique disjoint teams",
        "requested": size,
        "validation": {
            "command": "node pokemon-showdown validate-team gen9ou",
            "server_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SHOWDOWN.parent, text=True).strip(),
            "candidates_drawn": drawn, "validated": validated,
            "rejected_reasons": dict(rejected),
        },
        "disjointness": {
            "normalizations": {
                "ordered_sha256": "TeamSet.join_team(TeamSet.parse_showdown_team(text))",
                "canonical_sha256": "sorted multiset of per-Pokemon join_team([mon]) hashes",
            },
            "excluded_pools": {k: v["files"] for k, v in pools.items()},
            "intersections": intersections, "within_pool": within,
        },
        "teams": entries,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote fresh heldout: {size} teams from {drawn} candidates", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", default=["competitive", "hl_05_26"],
        choices=["competitive", "hl_05_26"],
    )
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--w5", action="store_true")
    parser.add_argument("--w5-fresh-heldout", action="store_true",
                        help="build ONLY a fresh heldout pool (needs --output, --exclude-dirs)")
    parser.add_argument("--exclude-dirs", nargs="+", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "data/teams_w5")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.sample < 1:
        parser.error("--sample must be positive")
    if len(set(args.sources)) != len(args.sources):
        parser.error("--sources must not contain duplicates")
    if args.w5 and args.w5_fresh_heldout:
        parser.error("--w5 and --w5-fresh-heldout are exclusive")
    if args.w5_fresh_heldout:
        if not args.exclude_dirs:
            parser.error("--w5-fresh-heldout requires --exclude-dirs")
        if args.output == ROOT / "data/teams_w5" or args.output.exists():
            parser.error("--output must be a new directory (never an existing W5 pool root)")
        build_fresh_heldout(args.output, args.seed, args.exclude_dirs, 200, args.workers)
    elif args.w5:
        build_w5_pools(
            args.output, args.seed, {"collect": 1000, "dev": 200, "heldout": 200},
            args.workers,
        )
    else:
        build_teamset(args.sources, args.sample, args.seed)


if __name__ == "__main__":
    main()
