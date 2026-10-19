"""W7: select every remaining Alakazam-family POV from the full pac-base gen9ou archive, locally.

The W6-P batches (`.cache/metamon/w6-pacbase`, `w6-pacbase-b`, 300k POVs) were selected on Colab.
This does the same selection on this box from the downloaded archive
(`jakegrigsby/metamon-parsed-pile`, `pac-base/gen9ou.tar.lz4` + its ratarmount SQLite index):

- pool = POVs whose player is in the Alakazam family (`alakazam`, `alakazamdefault`,
  `pac-mm-alakazam`), the same 952,816-POV pool as W6-P;
- drop every battle already used by the W6-P batches;
- keep one POV per battle (seeded choice when both sides are Alakazam-family);
- shuffle with the seed and cut into batches of --batch-size POVs, written as
  `<out-prefix>-<letter>/gen9ou/*.json.lz4` + `manifest.json` (the layout `w6p_build_pile.py` reads).

The archive is streamed once (lz4 → tar), so only the selected files touch the disk.

    uv run python scripts/w7_select_pile.py --archive .cache/pile-full/pac-base/gen9ou.tar.lz4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import string
import tarfile
from pathlib import Path

import lz4.frame
import numpy as np

NAME = re.compile(
    r"^metamon-gen9ou-(\d+)_Unrated_(.+)_vs_(.+)_\d\d-\d\d-\d{4}-[\d:]+_(WIN|LOSS)\.json\.lz4$")
FAMILIES = ("alakazam", "alakazamdefault", "pac-mm-alakazam")
USED = (".cache/metamon/w6-pacbase", ".cache/metamon/w6-pacbase-b")
POOL_SIZE = 952_816  # W6-P pool, recorded in the batch manifests


def family(user: str) -> str:
    return re.sub(r"[-_]?\d+$", "", user).lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--index", default=None, help="default: <archive without .lz4>.index.sqlite")
    ap.add_argument("--out-prefix", default=".cache/metamon/w7-pacbase")
    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20262420)
    args = ap.parse_args()
    archive = Path(args.archive)
    index = Path(args.index or str(archive).removesuffix(".lz4") + ".index.sqlite")

    names = [n for (n,) in sqlite3.connect(index).execute(
        "select name from files where path = '/gen9ou'")]
    pool: dict[str, list[str]] = {}
    for n in names:
        m = NAME.match(n)
        if m and family(m.group(2)) in FAMILIES:
            pool.setdefault(m.group(1), []).append(n)
    povs = sum(len(v) for v in pool.values())
    if povs != POOL_SIZE:
        raise ValueError(f"pool has {povs} POVs, W6-P recorded {POOL_SIZE}")
    used = set()
    for d in USED:
        for f in Path(d, "gen9ou").glob("*.json.lz4"):
            used.add(NAME.match(f.name).group(1))
    if len(used) != 299_998:
        raise ValueError(f"expected 299,998 used battles, found {len(used)}")
    rng = np.random.default_rng(args.seed)
    battles = sorted(b for b in pool if b not in used)
    chosen = [sorted(pool[b])[rng.integers(len(pool[b]))] for b in battles]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    batches = [chosen[i:i + args.batch_size] for i in range(0, len(chosen), args.batch_size)]
    target = {}
    dirs = []
    for letter, batch in zip(string.ascii_lowercase[2:], batches):  # c, d, e, ...
        d = Path(f"{args.out_prefix}-{letter}")
        (d / "gen9ou").mkdir(parents=True, exist_ok=False)
        dirs.append((d, batch))
        target |= {n: d / "gen9ou" / n for n in batch}
    print(json.dumps({"pool_povs": povs, "pool_battles": len(pool), "used_battles": len(used),
                      "selected": len(chosen), "batches": [len(b) for b in batches]}), flush=True)

    written = 0
    digest = hashlib.sha256()
    with lz4.frame.open(archive, "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
        for member in tar:
            tar.members = []  # streaming mode keeps every TarInfo otherwise (4.5M -> OOM)
            name = Path(member.name).name
            path = target.get(name)
            if path is None or not member.isfile():
                continue
            data = tar.extractfile(member).read()
            path.write_bytes(data)
            digest.update(name.encode())
            written += 1
            if written % 50_000 == 0:
                print(json.dumps({"written": written}), flush=True)
    if written != len(target):
        raise ValueError(f"wrote {written} of {len(target)} selected files")
    for d, batch in dirs:
        (d / "manifest.json").write_text(json.dumps({
            "source": "jakegrigsby/metamon-parsed-pile pac-base/gen9ou.tar.lz4",
            "families": list(FAMILIES), "seed": args.seed, "pool": POOL_SIZE,
            "excluded_battles_from_w6p_batches": len(used), "files": len(batch),
            "one_pov_per_battle": True, "selected_by": "scripts/w7_select_pile.py",
        }, indent=1))
    print(json.dumps({"written": written, "names_sha256": digest.hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
