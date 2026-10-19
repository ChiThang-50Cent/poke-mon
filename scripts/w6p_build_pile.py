"""W6-P: encode the Metamon self-play subset (pac-base gen9ou, Alakazam family, 100k POVs).

Source: `.cache/metamon/w6-pacbase/` (manifest.json + gen9ou/*.json.lz4, built on Colab from
jakegrigsby/metamon-parsed-pile). One POV per battle in this subset; split 95/5 by a hash of
the battle id. Output: datasets/w6p/{train,val}.npz (W4 array layout + `outcomes`, `family`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

import numpy as np

import pokeai  # noqa: F401  (metamon import order)
from pokeai.data.dataset import native_dataset
from pokeai.data.features import ObservationEncoder
from pokeai.data.w3 import _EncoderSpace, encode_trajectory

NAME = re.compile(
    r"^metamon-gen9ou-(\d+)_Unrated_(.+)_vs_(.+)_\d\d-\d\d-\d{4}-[\d:]+_(WIN|LOSS)\.json\.lz4$")
_NATIVE = None


def family(user: str) -> str:
    return re.sub(r"[-_]?\d+$", "", user).lower()


def split_of(battle: str) -> str:
    return "val" if int(hashlib.sha256(battle.encode()).hexdigest(), 16) % 20 == 0 else "train"


def _encode(filename: str):
    return encode_trajectory(_NATIVE, filename)


def main() -> None:
    global _NATIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".cache/metamon/w6-pacbase")
    parser.add_argument("--out", default="datasets/w6p")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = ObservationEncoder()
    encoder.save(out / "encoder.json")
    if (out / "encoder.json").read_bytes() != Path("datasets/w4/encoder.json").read_bytes():
        raise ValueError("Encoder differs from W4: train/serve parity broken")
    _NATIVE = native_dataset(root, human=True)
    files = sorted(_NATIVE.filenames)
    if len(files) != len(list((root / "gen9ou").glob("*.json.lz4"))):
        raise ValueError("Native index differs from extracted files")
    rows = []
    for f in files:
        m = NAME.match(Path(f).name)
        if m is None:
            raise ValueError(f"Unexpected filename: {f}")
        rows.append({"file": f, "battle_id": m.group(1), "family": family(m.group(2)),
                     "win": m.group(4) == "WIN"})
    if len({r["battle_id"] for r in rows}) != len(rows):
        raise ValueError("Expected one POV per battle in this subset")
    _NATIVE.observation_space = _EncoderSpace(encoder)
    report = {"source": json.loads((root / "manifest.json").read_text()), "splits": {}}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for split in ("train", "val"):
            part = [r for r in rows if split_of(r["battle_id"]) == split]
            keys = ("tokens", "numbers", "masks", "actions", "valid")
            chunks = {k: [] for k in keys}
            offsets, reasons = [0], Counter()
            for encoded, excluded in pool.imap(_encode, (r["file"] for r in part), chunksize=64):
                for k in keys:
                    chunks[k].append(encoded[k])
                reasons.update(excluded)
                offsets.append(offsets[-1] + len(encoded["actions"]))
            arrays = {k: np.concatenate(v) for k, v in chunks.items()}
            del chunks
            arrays |= {
                "offsets": np.asarray(offsets, dtype=np.int64),
                "battle_ids": np.asarray([r["battle_id"] for r in part]),
                "family": np.asarray([r["family"] for r in part]),
                "outcomes": np.asarray([r["win"] for r in part], dtype=np.float32),
            }
            np.savez(out / f"{split}.npz", **arrays)
            unknown = int((arrays["tokens"] == encoder.unknown_id).sum())
            report["splits"][split] = {
                "trajectories": len(part), "decisions": len(arrays["actions"]),
                "valid_decisions": int(arrays["valid"].sum()), "excluded_by_reason": dict(reasons),
                "unknown_token_rate": unknown / arrays["tokens"].size,
                "all_finite": bool(np.isfinite(arrays["numbers"]).all()),
                "win_rate": float(arrays["outcomes"].mean()),
                "families": dict(Counter(r["family"] for r in part)),
                "action_class_histogram": np.bincount(
                    arrays["actions"][arrays["valid"]], minlength=13).tolist(),
            }
            print(json.dumps({split: report["splits"][split]}), flush=True)
            del arrays
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
