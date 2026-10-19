"""W7: build a memory-mapped training cache over several encoded pile batches without loading
them all into RAM (the autoresearch `prepare.py` concatenates in memory, which does not fit
for ~950k POVs).

Output layout = `prepare.Split` layout (`train_{offsets,outcomes,tokens,numbers,masks,actions,
valid}.npy`). The val files are symlinked from the autoresearch cache, so `prepare.evaluate`
scores exactly the same fixed 2,000-trajectory sample. Tokens are stored as int16
(vocab 2,542), which halves the disk footprint; `Split.batch` casts them to int64.

    uv run python scripts/w7_build_cache.py --out .cache/epn-pile953 \
        --train datasets/w6p datasets/w6p_b datasets/w7_c datasets/w7_d datasets/w7_e datasets/w7_f
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("tokens", "numbers", "masks", "actions", "valid")
VAL_FROM = Path(".cache/autoresearch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="dataset dirs with train.npz")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    encoder = Path("datasets/w4/encoder.json").read_bytes()
    sources = [Path(d) for d in args.train]
    for d in sources:
        if (d / "encoder.json").read_bytes() != encoder:
            raise ValueError(f"{d}: encoder differs from production")
    zips = [np.load(d / "train.npz") for d in sources]
    totals = [int(z["offsets"][-1]) for z in zips]
    n = sum(totals)
    offsets, outcomes, base = [np.zeros(1, np.int64)], [], 0
    for z, total in zip(zips, totals):
        offsets.append(z["offsets"][1:] + base)
        outcomes.append(z["outcomes"])
        base += total
    np.save(out / "train_offsets.npy", np.concatenate(offsets))
    np.save(out / "train_outcomes.npy", np.concatenate(outcomes))
    for key in KEYS:
        first = zips[0][key]
        dtype = np.int16 if key == "tokens" else first.dtype
        dest = np.lib.format.open_memmap(out / f"train_{key}.npy", mode="w+", dtype=dtype,
                                         shape=(n, *first.shape[1:]))
        del first
        start = 0
        for z, total in zip(zips, totals):
            part = z[key]
            if key == "tokens" and part.max() > np.iinfo(np.int16).max:
                raise ValueError("token id does not fit int16")
            dest[start:start + total] = part
            start += total
            del part
        dest.flush()
        del dest
        print(key, "done", flush=True)
    for f in sorted(VAL_FROM.glob("val_*.npy")):
        (out / f.name).symlink_to(f.resolve())
    (out / "manifest.json").write_text(json.dumps({
        "train_sources": [str(d) for d in sources], "decisions": n,
        "trajectories": int(len(np.load(out / "train_offsets.npy")) - 1),
        "val": "symlinked from .cache/autoresearch (fixed w6p_b/val sample)"}, indent=1))
    print("trajectories", len(np.load(out / "train_offsets.npy")) - 1, "decisions", n)


if __name__ == "__main__":
    main()
