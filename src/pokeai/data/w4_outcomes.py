"""Filename-derived W4 value labels, sealed to complete ordered trajectories.

Labels use prepare_entries/temporal_split only. The original NPZ has no POV
field: sorted source order alone is insufficient proof. Building additionally
compares every raw decision's own/opponent HP, opponents remaining and action
against NPZ (no observation encoding), then seals every stored array with SHA256.
By default only train/val are opened. Final evaluation must explicitly opt in
to test-only labels in a separate outcomes_holdout.json; training stays sealed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import lz4.frame
import numpy as np

from pokeai.data.w3 import load_split
from pokeai.data.w4 import prepare_entries, temporal_split

_STEPS = ("tokens", "numbers", "masks", "actions", "valid")
_META = ("battle_ids", "ratings", "months", "offsets")


def _digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _alignment(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {key: _digest(arrays[key]) for key in (*_STEPS, *_META)}


def _validate_rows(rows: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    for key, field in (
        ("battle_ids", "battle_id"), ("ratings", "rating"), ("months", "month")
    ):
        if not np.array_equal(arrays[key], [row[field] for row in rows]):
            raise ValueError(f"Outcome source/NPZ {key} order mismatch")
    offsets = arrays["offsets"]
    if (
        offsets.ndim != 1
        or len(offsets) != len(rows) + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or any(len(arrays[key]) != offsets[-1] for key in _STEPS)
    ):
        raise ValueError("Outcome NPZ trajectory offsets mismatch")


def _prove_pov(path: Path, arrays: dict[str, np.ndarray], index: int) -> None:
    """Compare direct stored fields, never instantiate an encoder or dataset."""
    raw = json.loads(lz4.frame.decompress(path.read_bytes()))
    start, stop = arrays["offsets"][index:index + 2]
    states = raw["states"][:-1]
    actions = np.asarray(raw["actions"][:-1], dtype=arrays["actions"].dtype)
    if len(states) != stop - start or not np.array_equal(
        actions, arrays["actions"][start:stop]
    ):
        raise ValueError(f"Outcome POV/action alignment mismatch: {path.name}")
    # DefaultObservationSpace: remaining/6, own active 15, moves 12,
    # switch HP 5, opponent active 15 => opponent HP at index 33.
    expected = np.asarray([
        [state["opponents_remaining"] / 6.0,
         state["player_active_pokemon"]["hp_pct"],
         state["opponent_active_pokemon"]["hp_pct"]]
        for state in states
    ], dtype=np.float32).reshape(-1, 3)
    actual = arrays["numbers"][start:stop][:, [0, 1, 33]]
    if not np.array_equal(expected, actual):
        raise ValueError(f"Outcome POV/HP alignment mismatch: {path.name}")


def build_outcomes(
    root: str | Path = "datasets/w4",
    source_root: str | Path = ".cache/metamon/w4-human",
    *,
    allow_holdout: bool = False,
) -> dict[str, Any]:
    """Write sealed labels, with test-only access requiring explicit opt-in.

    Initial POV proof checks all-step raw actions and HP against existing NPZ.
    Full-array digests subsequently reject any ordering/content change, including
    swapping same-battle POVs. No label or filename metadata is a model input.
    """
    root, source_root = Path(root), Path(source_root)
    splits = ("test",) if allow_holdout else ("train", "val")
    filename = "outcomes_holdout.json" if allow_holdout else "outcomes.json"
    filenames = sorted(str(path) for path in (source_root / "gen9ou").glob("*.json.lz4"))
    entries, _ = prepare_entries(filenames)
    assignments = temporal_split(entries)
    frozen = json.loads((root / "split_ids.json").read_text())
    sidecar: dict[str, Any] = {
        "version": 1,
        "label_source": "prepare_entries filename WIN=1, LOSS=0; temporal_split",
        "alignment_evidence": {
            "source_order": "prepare_entries sorts file; build_w4_dataset uses ordered pool.imap",
            "initial_pov_proof": "all decision actions, own HP, opponent HP, opponents_remaining",
            "sealed_arrays": [*_STEPS, *_META],
            "turns": "unavailable: DefaultObservationSpace numbers[48] contains no turn feature",
            "holdout": (
                "explicit final-evaluation opt-in; test-only NPZ and raw POV proof"
                if allow_holdout else
                "test filenames only for splitting; no test NPZ or raw bodies opened"
            ),
        },
        "splits": {},
    }
    for split in splits:
        if assignments[split] != frozen[split]:
            raise ValueError(f"Outcome frozen {split} battle IDs mismatch")
        selected = set(assignments[split])
        rows = [row for row in entries if row["battle_id"] in selected]
        arrays = load_split(split, root)
        _validate_rows(rows, arrays)
        for index, row in enumerate(rows):
            _prove_pov(Path(row["file"]), arrays, index)
        sidecar["splits"][split] = {
            "alignment": _alignment(arrays),
            "outcomes": [float(row["result"] == "WIN") for row in rows],
            "trajectories": [
                {
                    "filename": Path(row["file"]).name,
                    "battle_id": row["battle_id"],
                    "pov_username": row["pov_username"],
                    "result": row["result"],
                    "rating": row["rating"],
                    "month": row["month"],
                    "decisions": int(arrays["offsets"][i + 1] - arrays["offsets"][i]),
                    "turns": None,
                }
                for i, row in enumerate(rows)
            ],
        }
        del arrays
    (root / filename).write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def load_outcomes(
    root: str | Path, split: str, arrays: dict[str, np.ndarray],
    *, allow_holdout: bool = False,
) -> np.ndarray:
    """Return sealed labels; test requires opt-in and its separate sidecar."""
    if split not in {"train", "val", "test"} or (split == "test" and not allow_holdout):
        raise ValueError("Outcome labels are restricted to train and val")
    holdout = split == "test"
    filename = "outcomes_holdout.json" if holdout else "outcomes.json"
    expected_splits = {"test"} if holdout else {"train", "val"}
    sidecar = json.loads((Path(root) / filename).read_text())
    if sidecar.get("version") != 1 or set(sidecar["splits"]) != expected_splits:
        raise ValueError("Unsupported outcome sidecar or holdout contamination")
    stored = sidecar["splits"][split]
    rows = stored["trajectories"]
    identities, _ = prepare_entries([row["filename"] for row in rows])
    fields = ("battle_id", "pov_username", "result", "rating", "month")
    if len(identities) != len(rows) or any(
        identity["file"] != row["filename"]
        or any(identity[field] != row[field] for field in fields)
        for identity, row in zip(identities, rows)
    ):
        raise ValueError("Outcome filename/POV identity order mismatch")
    _validate_rows(rows, arrays)
    if _alignment(arrays) != stored["alignment"]:
        raise ValueError(f"Outcome {split} NPZ identity/order hash mismatch")
    labels = np.asarray(stored["outcomes"], dtype=np.float32)
    expected = np.asarray([float(row["result"] == "WIN") for row in rows])
    if (
        labels.shape != (len(rows),)
        or any(row["result"] not in {"WIN", "LOSS"} for row in rows)
        or not np.array_equal(labels, expected)
    ):
        raise ValueError("Outcome filename result/label mismatch")
    return labels
