"""W3 trajectory interface over native Metamon local and human data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from metamon.data import MetamonDataset, ParsedReplayDataset
from metamon.interface import (
    DefaultActionSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)

SPACE_VERSIONS = {
    "observation_space_version": "DefaultObservationSpace@metamon-0a00a75",
    "action_space": "DefaultActionSpace",
    "reward_function": "DefaultShapedReward",
}



def native_dataset(root: Path, *, human: bool = False) -> MetamonDataset:
    cls = ParsedReplayDataset if human else MetamonDataset
    return cls(
        dset_root=str(root),
        formats=["gen9ou"],
        observation_space=DefaultObservationSpace(),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        shuffle=False,
    )


def resolve_run_path(run: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return run / path


def local_entries(run: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run / "manifest.json").read_text())
    rows = []
    for entry in manifest["trajectory_files"]:
        file = resolve_run_path(run, entry["file"])
        meta_path = resolve_run_path(run, entry["meta"])
        meta = json.loads(meta_path.read_text())
        rows.append(
            {
                "file": str(file),
                "meta": str(meta_path),
                "battle_id": entry["battle_id"],
                "side": entry["player_side"],
                "pov_username": file.name.split("_")[2],
                "n_steps": meta["n_steps"],
                "source": "local",
                "root": str(run / "trajectories"),
            }
        )
    return rows



def human_entries(root: Path) -> list[dict[str, Any]]:
    native = native_dataset(root, human=True)
    rows = []
    for filename in sorted(native.filenames):
        name = Path(filename).name
        parts = name.removesuffix(".lz4").removesuffix(".json").split("_")
        _, actions, _, _ = native.load_filename(filename)
        rows.append(
            {
                "file": filename,
                "meta": None,
                "battle_id": parts[0],
                "side": None,
                "pov_username": "_".join(parts[2:-4]),
                "n_steps": len(actions["chosen"]),
                "source": "human",
                "root": str(root),
            }
        )
    return rows


class TrajectoryDataset:
    """One decision-aligned trajectory per item; terminal observations excluded.

    Missing/unsupported decisions remain visible and must be excluded from BC loss.
    Local masks are exact engine masks; human masks are replay-derived candidates.

    Human native files retain POV usernames but not original server sides:
    ``player_side`` is None for human data, p1/p2 for local data.
    ``pov_username`` identifies the perspective without inventing a server side.
    """

    def __init__(
        self, index: str | Path | list[dict[str, Any]], split: str | None = None
    ) -> None:
        rows = (
            [json.loads(line) for line in Path(index).read_text().splitlines() if line]
            if isinstance(index, (str, Path))
            else index
        )
        self.entries = [r for r in rows if split is None or r.get("split") == split]
        self._datasets: dict[tuple[str, str], MetamonDataset] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[index]
        key = (row["root"], row["source"])
        if key not in self._datasets:
            self._datasets[key] = native_dataset(Path(key[0]), human=key[1] == "human")
            indexed = {
                str(Path(filename).resolve())
                for filename in self._datasets[key].filenames
            }
            required = {
                str(Path(entry["file"]).resolve())
                for entry in self.entries
                if (entry["root"], entry["source"]) == key
            }
            if not required <= indexed:
                del self._datasets[key]
                raise ValueError("Trajectory filename rejected by native Metamon index")
        observations, action_info, rewards, dones = self._datasets[key].load_filename(
            row["file"]
        )
        actions = np.asarray(action_info["chosen"], dtype=np.int64)
        missing = np.asarray(action_info["missing"], dtype=np.bool_)
        count = len(actions)
        if row["source"] == "local":
            meta = json.loads(Path(row["meta"]).read_text())
            masks = np.asarray(meta["action_masks"], dtype=np.bool_).reshape(count, 13)
            actions = np.asarray(meta["chosen_actions"], dtype=np.int64)
            missing = missing | (actions < 0)
            result = meta["final_result"]
        else:
            masks = np.zeros((count, 13), dtype=np.bool_)
            for step, legal in enumerate(action_info["legal"]):
                for action in legal:
                    if 0 <= action < 13:
                        masks[step, action] = True
            result = "win" if "_WIN.json" in row["file"] else "loss"
        return {
            "observations": {k: np.asarray(v[:-1]) for k, v in observations.items()},
            "actions": actions,
            "action_masks": masks,
            "rewards": rewards,
            "dones": dones,
            "missing": missing,
            "battle_result": result,
            "battle_id": row["battle_id"],
            "player_side": row["side"],
            "pov_username": row["pov_username"],
            "source": row["source"],
        }


def default_human_root() -> Path:
    return Path(os.environ.get("METAMON_CACHE_DIR", ".cache/metamon")) / "w2-human"
