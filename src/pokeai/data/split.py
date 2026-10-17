"""Deterministic battle-grouped splits, independent of trajectory ordering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def split_battles(
    entries: list[dict[str, Any]], seed: int = 42
) -> dict[str, list[str]]:
    ids = sorted(
        {row["battle_id"] for row in entries},
        key=lambda battle: hashlib.sha256(f"{seed}:{battle}".encode()).digest(),
    )
    n = len(ids)
    n_test = max(1, int(n * 0.1)) if n >= 3 else 0
    n_val = max(1, int(n * 0.1)) if n >= 3 else 0
    return {
        "test": ids[:n_test],
        "val": ids[n_test : n_test + n_val],
        "train": ids[n_test + n_val :],
    }


def write_dataset_index(
    entries: list[dict[str, Any]], output: Path, seed: int = 42
) -> dict[str, list[str]]:
    splits = split_battles(entries, seed)
    assignments = {battle: split for split, ids in splits.items() for battle in ids}
    output.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (output / f"{split}_ids.json").write_text(json.dumps(ids, indent=2) + "\n")
    rows = [dict(row, split=assignments[row["battle_id"]]) for row in entries]
    (output / "local_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return splits
