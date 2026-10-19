"""W6-A opponent-action labels for W4 learner rows, from the other POV of the same battle.

The label at learner row t is the action the opponent chose at the same simultaneous
decision. It is a training target only: model inputs stay the existing W4 arrays, which
the replay parser builds from the learner's pre-action state (no part of the opponent's
turn-t action is in them). The other POV is opened only to decode the target.

Alignment: the parser inserts forced-switch subturns per side, so POVs are paired by
the order of their non-forced states, and a battle is rejected unless every pair agrees
on both active Pokémon. Forced-switch learner rows get no label.

Label space (learner view): "move:<name>" or "switch:<teampreview name>", plus a
separate Tera flag on move labels. Switch targets use the learner's public team-preview
name, so the switch support is known at decision time.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import lz4.frame
import numpy as np
from metamon.interface import (
    UniversalState,
    consistent_move_order,
    consistent_pokemon_order,
)

from pokeai.data.w3 import load_split
from pokeai.data.w4 import prepare_entries
from pokeai.data.w4_outcomes import load_outcomes

MOVE_UNK, SWITCH_UNK = "move:<unk>", "switch:<unk>"
PREVIEW = 6


class AlignmentError(ValueError):
    pass


def decode_action(state: dict[str, Any], index: int) -> tuple[str, str, bool] | None:
    """Invert UniversalAction for a stored state: (kind, name, tera); None if unrevealed."""
    if index < 0:
        return None
    universal = UniversalState.from_dict(copy.deepcopy(state))
    if 4 <= index <= 8:
        switches = consistent_pokemon_order(universal.available_switches)
        if index - 4 >= len(switches):
            raise AlignmentError(f"switch slot {index} beyond {len(switches)} options")
        return "switch", switches[index - 4].name, False
    slot = index - 9 if index >= 9 else index
    moves = consistent_move_order(universal.player_active_pokemon.moves)
    if slot >= len(moves):
        raise AlignmentError(f"move slot {index} beyond {len(moves)} moves")
    return "move", moves[slot].name, index >= 9


def preview_name(name: str, preview: list[str]) -> str | None:
    """Map an in-battle forme name to the opponent's team-preview entry."""
    if name in preview:
        return name
    matches = [p for p in preview if name.startswith(p) or p.startswith(name)]
    return max(matches, key=len) if matches else None


def pair_labels(
    learner: dict[str, Any], opponent: dict[str, Any]
) -> tuple[list[tuple[str, int] | None], Counter]:
    """One (class, tera) or None per learner row (states[:-1]); raises on misalignment."""
    rows = learner["states"][:-1]
    mine = [i for i, s in enumerate(rows) if not s["forced_switch"]]
    theirs = [j for j, s in enumerate(opponent["states"][:-1]) if not s["forced_switch"]]
    if len(mine) != len(theirs):
        raise AlignmentError(f"non-forced rows differ: {len(mine)} vs {len(theirs)}")
    labels: list[tuple[str, int] | None] = [None] * len(rows)
    qa: Counter = Counter()
    for i, j in zip(mine, theirs):
        state, other = rows[i], opponent["states"][j]
        if (
            state["player_active_pokemon"]["name"] != other["opponent_active_pokemon"]["name"]
            or state["opponent_active_pokemon"]["name"] != other["player_active_pokemon"]["name"]
        ):
            raise AlignmentError(f"active Pokémon disagree at rows {i}/{j}")
        decoded = decode_action(other, opponent["actions"][j])
        if decoded is None:
            qa["unrevealed"] += 1
            continue
        kind, name, tera = decoded
        following = learner["states"][i + 1]
        if kind == "move":
            qa["move"] += 1
            qa["move_eq_next_prev_move"] += following["opponent_prev_move"]["name"] == name
            qa["move_eq_current_prev_move"] += state["opponent_prev_move"]["name"] == name
            labels[i] = (f"move:{name}", int(tera))
            continue
        qa["switch"] += 1
        qa["switch_eq_next_active"] += following["opponent_active_pokemon"]["name"] == name
        target = preview_name(name, state["opponent_teampreview"])
        if target is None:
            qa["switch_not_in_preview"] += 1
            continue
        labels[i] = (f"switch:{target}", -1)
    return labels, qa


def _load(path: Path) -> dict[str, Any]:
    return json.loads(lz4.frame.decompress(path.read_bytes()))


def _battle_job(job: tuple[str, str | None]) -> dict[str, Any]:
    path, partner = job
    learner = _load(Path(path))
    rows = learner["states"][:-1]
    context = [
        (
            list(s["opponent_teampreview"]),
            preview_name(s["opponent_active_pokemon"]["name"], s["opponent_teampreview"]),
        )
        for s in rows
    ]
    if partner is None:
        return {"labels": [None] * len(rows), "context": context, "status": "single_pov"}
    try:
        labels, qa = pair_labels(learner, _load(Path(partner)))
    except AlignmentError as error:
        return {"labels": [None] * len(rows), "context": context,
                "status": "rejected", "reason": str(error)}
    return {"labels": labels, "context": context, "status": "paired", "qa": dict(qa)}


def _jobs(split: str, root: Path, source: Path) -> list[tuple[str, str | None]]:
    sidecar = json.loads((root / "outcomes.json").read_text())
    rows = sidecar["splits"][split]["trajectories"]
    entries, _ = prepare_entries(
        sorted(str(p) for p in (source / "gen9ou").glob("*.json.lz4"))
    )
    by_battle: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_battle.setdefault(entry["battle_id"], []).append(entry)
    jobs = []
    for row in rows:
        pair = by_battle[row["battle_id"]]
        own = [e for e in pair if e["pov_username"] == row["pov_username"]]
        other = [e for e in pair if e["pov_username"] != row["pov_username"]]
        if len(own) != 1 or Path(own[0]["file"]).name != row["filename"]:
            raise ValueError(f"Trajectory identity mismatch: {row['filename']}")
        jobs.append((own[0]["file"], other[0]["file"] if other else None))
    return jobs


def build_labels(
    root: str | Path = "datasets/w4",
    source: str | Path = ".cache/metamon/w4-human",
    output: str | Path = "datasets/w6a",
    workers: int = 8,
) -> dict[str, Any]:
    """Write {train,val}.npz label arrays aligned to W4 rows, plus labels.json.

    The class vocabulary is built from train only. W4 test is spent and never opened.
    """
    root, source, output = Path(root), Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val"):
        with Pool(workers) as pool:
            results[split] = pool.map(_battle_job, _jobs(split, root, source), chunksize=64)
    counts = Counter(
        label[0] for result in results["train"] for label in result["labels"] if label
    )
    vocab = [MOVE_UNK, SWITCH_UNK] + sorted(counts, key=lambda c: (-counts[c], c))
    index = {name: i for i, name in enumerate(vocab)}

    def class_id(name: str) -> int:
        return index.get(name, index[SWITCH_UNK if name.startswith("switch:") else MOVE_UNK])

    report: dict[str, Any] = {
        "version": 1,
        "label_source": "other POV of the same battle, paired by non-forced row order; "
                        "learner rows/inputs unchanged; W4 test never opened",
        "vocab": vocab,
        "train_class_counts": dict(counts),
        "splits": {},
    }
    for split in ("train", "val"):
        arrays = load_split(split, root)
        load_outcomes(root, split, arrays)  # proves NPZ order == outcomes.json filenames
        n = int(arrays["offsets"][-1])
        opp_class = np.full(n, -1, np.int32)
        opp_tera = np.full(n, -1, np.int8)
        opp_preview = np.full((n, PREVIEW), -1, np.int32)
        opp_active = np.full(n, -1, np.int32)
        status: Counter = Counter()
        qa: Counter = Counter()
        reasons: Counter = Counter()
        for t, result in enumerate(results[split]):
            start, stop = int(arrays["offsets"][t]), int(arrays["offsets"][t + 1])
            if len(result["labels"]) != stop - start:
                raise ValueError(f"{split} trajectory {t}: row count mismatch")
            status[result["status"]] += 1
            qa.update(result.get("qa", {}))
            if "reason" in result:
                reasons[result["reason"].split(":")[0]] += 1
            for k, (label, (preview, active)) in enumerate(
                zip(result["labels"], result["context"])
            ):
                row = start + k
                ids = [class_id(f"switch:{p}") for p in preview[:PREVIEW]]
                opp_preview[row, : len(ids)] = ids
                if active is not None:
                    opp_active[row] = class_id(f"switch:{active}")
                if label is not None:
                    opp_class[row] = class_id(label[0])
                    opp_tera[row] = label[1]
        np.savez_compressed(
            output / f"{split}.npz", opp_class=opp_class, opp_tera=opp_tera,
            opp_preview=opp_preview, opp_active=opp_active,
        )
        labelled = opp_class >= 0
        report["splits"][split] = {
            "outcomes_alignment": json.loads((root / "outcomes.json").read_text())[
                "splits"][split]["alignment"],
            "rows": n,
            "labelled_rows": int(labelled.sum()),
            "labelled_valid_rows": int((labelled & arrays["valid"]).sum()),
            "trajectories": dict(status),
            "rejection_reasons": dict(reasons),
            "qa": dict(qa),
            "unk_labels": int(np.isin(opp_class, [0, 1]).sum()),
            "sha256": hashlib.sha256((output / f"{split}.npz").read_bytes()).hexdigest(),
        }
    (output / "labels.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_labels(
    split: str, arrays: dict[str, np.ndarray],
    root: str | Path = "datasets/w4", labels: str | Path = "datasets/w6a",
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Label arrays for a W4 split, refusing anything not aligned to these exact rows."""
    if split not in {"train", "val"}:
        raise ValueError("W6-A labels exist for train and val only (W4 test is spent)")
    labels = Path(labels)
    report = json.loads((labels / "labels.json").read_text())
    stored = report["splits"][split]
    load_outcomes(root, split, arrays)
    if stored["outcomes_alignment"] != json.loads(
        (Path(root) / "outcomes.json").read_text()
    )["splits"][split]["alignment"]:
        raise ValueError(f"W6-A {split} labels were built for different W4 arrays")
    path = labels / f"{split}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != stored["sha256"]:
        raise ValueError(f"W6-A {split} label file changed")
    with np.load(path) as data:
        result = {key: data[key] for key in data.files}
    if len(result["opp_class"]) != int(arrays["offsets"][-1]):
        raise ValueError(f"W6-A {split} label rows mismatch")
    return result, report["vocab"]
