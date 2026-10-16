"""Fetch, deterministically sample, and validate a fixed Gen9 OU team pool."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", default=["competitive", "hl_05_26"],
        choices=["competitive", "hl_05_26"],
    )
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.sample < 1:
        parser.error("--sample must be positive")
    if len(set(args.sources)) != len(args.sources):
        parser.error("--sources must not contain duplicates")
    build_teamset(args.sources, args.sample, args.seed)


if __name__ == "__main__":
    main()
