"""W8 E1: encoder v2 matchup extras for an already-encoded pile batch.

Re-reads the raw Metamon trajectories of a `w6p_build_pile.py` dataset in exactly the same
order (sorted native filenames, same battle-hash split), encodes every decision with
`ObservationEncoder(extras="matchup-v0")` (the live-play code path) and writes
`<dataset>/extras_{train,val}.npy` (float16 [decisions, 66]). Alignment is asserted per
trajectory: decision count and the 48 v1 numbers must equal the dataset's arrays, so the rows
(and the Kakuna labels keyed on them) stay aligned.

    uv run python scripts/w8_build_extras.py --root .cache/metamon/w6-pacbase \
        --dataset datasets/w6p --workers 12
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

import pokeai  # noqa: F401  (metamon import order)
from pokeai.data.dataset import native_dataset
from pokeai.data.features import ObservationEncoder
from pokeai.data.matchup import EXTRAS_DIM, EXTRAS_VERSION
from pokeai.data.w3 import _EncoderSpace, encode_trajectory

_NATIVE = None


def _encode(filename: str) -> tuple[np.ndarray, np.ndarray]:
    numbers = encode_trajectory(_NATIVE, filename)[0]["numbers"]
    return numbers[:, :48], numbers[:, 48:].astype(np.float16)


def main() -> None:
    global _NATIVE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="raw batch dir (manifest.json + gen9ou/)")
    ap.add_argument("--dataset", required=True, help="w6p_build_pile.py output for --root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()
    root, dataset = Path(args.root), Path(args.dataset)
    sys.path.insert(0, str(Path(__file__).parent))
    import w6p_build_pile as pile

    report = json.loads((dataset / "report.json").read_text())
    if report["source"] != json.loads((root / "manifest.json").read_text()):
        raise ValueError("--root manifest differs from the dataset's source")
    _NATIVE = native_dataset(root, human=True)
    _NATIVE.observation_space = _EncoderSpace(ObservationEncoder(extras=EXTRAS_VERSION))
    files = sorted(_NATIVE.filenames)
    battle = {f: pile.NAME.match(Path(f).name).group(1) for f in files}
    out_report = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for split in args.splits:
            part = [f for f in files if pile.split_of(battle[f]) == split]
            with np.load(dataset / f"{split}.npz") as z:
                offsets, numbers = z["offsets"], z["numbers"]
                ids = z["battle_ids"]
            if len(part) != len(offsets) - 1 or [battle[f] for f in part] != ids.tolist():
                raise ValueError(f"{split}: trajectory order differs from the dataset")
            extras = np.empty((int(offsets[-1]), EXTRAS_DIM), dtype=np.float16)
            for i, (v1, ex) in enumerate(pool.imap(_encode, part, chunksize=64)):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                if len(v1) != hi - lo or not np.array_equal(v1, numbers[lo:hi]):
                    raise ValueError(f"{split}: misaligned trajectory {part[i]}")
                extras[lo:hi] = ex
                if i % 20000 == 0:
                    print(split, i, "/", len(part), flush=True)
            if not np.isfinite(extras).all():
                raise ValueError(f"{split}: non-finite extras")
            np.save(dataset / f"extras_{split}.npy", extras)
            out_report[split] = {"trajectories": len(part), "decisions": len(extras),
                                 "pad_fraction": float((extras == -2).mean())}
            print(json.dumps({split: out_report[split]}), flush=True)
    (dataset / "extras.json").write_text(json.dumps(
        {"extras": EXTRAS_VERSION, "dim": EXTRAS_DIM, "splits": out_report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
