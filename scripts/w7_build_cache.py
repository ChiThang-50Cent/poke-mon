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
    ap.add_argument("--kakuna", nargs="*", default=[], metavar="DATASET=LABELS.npz",
                    help="w7_kakuna_label.py outputs for a prefix of a source's train split; "
                         "written as train_kakuna.npy (float16 [decisions, 13], NaN = unlabeled)")
    ap.add_argument("--labels-only", action="store_true",
                    help="add train_kakuna.npy to an existing cache built from the same --train")
    args = ap.parse_args()
    out = Path(args.out)
    sources = [Path(d) for d in args.train]
    if args.labels_only:
        if json.loads((out / "manifest.json").read_text())["train_sources"] != args.train:
            raise ValueError("--train differs from the existing cache")
    else:
        out.mkdir(parents=True, exist_ok=False)
    encoder = Path("datasets/w4/encoder.json").read_bytes()
    for d in sources:
        if (d / "encoder.json").read_bytes() != encoder:
            raise ValueError(f"{d}: encoder differs from production")
    zips = [np.load(d / "train.npz") for d in sources]
    totals = [int(z["offsets"][-1]) for z in zips]
    n = sum(totals)
    if args.kakuna:
        write_kakuna(out, args.train, zips, totals, dict(kv.split("=", 1) for kv in args.kakuna))
    if args.labels_only:
        return
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


def write_kakuna(out: Path, names: list[str], zips, totals, labels: dict[str, str]) -> None:
    unknown = set(labels) - set(names)
    if unknown:
        raise ValueError(f"labels for sources not in --train: {sorted(unknown)}")
    dest = np.lib.format.open_memmap(out / "train_kakuna.npy", mode="w+", dtype=np.float16,
                                     shape=(sum(totals), 13))
    dest[:] = np.nan
    start, report = 0, {}
    for name, z, total in zip(names, zips, totals):
        if name in labels:
            lab = np.load(labels[name])
            k = len(lab["battle_ids"])
            if not (np.array_equal(lab["battle_ids"], z["battle_ids"][:k])
                    and np.array_equal(lab["offsets"], z["offsets"][:k + 1])):
                raise ValueError(f"{labels[name]} is not a prefix of {name}/train.npz")
            dest[start:start + int(lab["offsets"][-1])] = lab["probs"]
            report[name] = {"labels": labels[name], "trajectories": k,
                            "decisions": int(lab["offsets"][-1])}
        start += total
    dest.flush()
    (out / "kakuna_manifest.json").write_text(json.dumps(report, indent=1))
    print("kakuna", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
