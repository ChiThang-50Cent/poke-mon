"""W6-B: join search-augmented trajectories to their `battles.jsonl` search rows.

A learner step's teacher label lives in `battles.jsonl[battle]["search"][behavior.search_row]`.
The join must be exact: one row per recorded step, same turn, the sent action equals both the
native trajectory action and the row's action, and C's choice equals the row's `c_action`.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from pokeai.data.dataset import resolve_run_path
from pokeai.data.validate import read_native

CONFIDENT_Z = 1.96


def override_z(row: dict[str, Any]) -> float | None:
    """Paired z of the executed override vs a_C (inf for a zero-variance positive delta)."""
    if row.get("category") != "search_override":
        return None
    stat = next(s for s in row["stats"] if s["slot"] == row["action"])
    return math.inf if stat["se"] == 0 else stat["delta"] / stat["se"]


def trajectory_errors(
    native: dict, meta: dict, rows: list[dict[str, Any]] | None
) -> tuple[list[str], list[dict[str, Any]]]:
    """Join errors for one POV, plus the joined learner rows (in step order)."""
    behaviors = meta["behavior"]
    searched = [b for b in behaviors if b and "executed_index" in b]
    if not meta.get("learner"):
        if searched or meta.get("search_augmented"):
            return ["opponent POV carries search fields"], []
        return [], []
    if meta.get("search_augmented") is not True:
        return ["learner POV not marked search_augmented"], []
    if rows is None:
        return ["battle has no search rows"], []
    errors, joined, used = [], [], set()
    for i, (behavior, action) in enumerate(zip(behaviors, native["actions"][:-1], strict=True)):
        k = behavior.get("search_row")
        if type(k) is not int or not 0 <= k < len(rows) or k in used:
            errors.append(f"{i}: bad search_row {k!r}")
            continue
        used.add(k)
        row = rows[k]
        executed = behavior["executed_index"]
        if row.get("turn") != meta["decision_turns"][i]:
            errors.append(f"{i}: turn {meta['decision_turns'][i]} != row turn {row.get('turn')}")
        if executed != action:
            errors.append(f"{i}: executed_index {executed} != native action {action}")
        if executed < 0:
            if row.get("category") != "not_represented":
                errors.append(f"{i}: no model choice but row category {row.get('category')}")
        else:
            if row.get("c_action") != behavior["chosen_index"]:
                errors.append(f"{i}: row c_action {row.get('c_action')} != chosen_index")
            if row.get("action") != executed:
                errors.append(f"{i}: row action {row.get('action')} != executed {executed}")
            if (row.get("category") == "search_override") != (executed != behavior["chosen_index"]):
                errors.append(f"{i}: override category disagrees with executed vs chosen")
        joined.append(row)
    if used != set(range(len(rows))):
        errors.append(f"{len(rows) - len(used)} search rows not joined to any step")
    return errors, joined


def search_join(run: str | Path) -> dict[str, Any]:
    """Validate every POV of a search-augmented run and summarize label yield and cost."""
    run = Path(run)
    battles = {
        b["battle_id"]: b
        for b in map(json.loads, (run / "battles.jsonl").read_text().splitlines())
    }
    manifest = json.loads((run / "manifest.json").read_text())
    errors: list[str] = []
    categories: Counter = Counter()
    by_opponent: dict[str, Counter] = {}
    decision_s: list[float] = []
    learner_povs = 0
    for entry in manifest["trajectory_files"]:
        meta = json.loads(resolve_run_path(run, entry["meta"]).read_text())
        native = read_native(resolve_run_path(run, entry["file"]))
        battle = battles[entry["battle_id"]]
        issues, joined = trajectory_errors(native, meta, battle.get("search"))
        errors.extend(f"{entry['battle_id']}/{entry['player_side']}: {e}" for e in issues)
        if not meta.get("learner"):
            continue
        learner_povs += 1
        opponent = by_opponent.setdefault(str(meta["opponent_type"]), Counter())
        opponent["battles"] += 1
        for row in joined:
            categories[row["category"]] += 1
            opponent["decisions"] += 1
            z = override_z(row)
            if z is not None:
                opponent["overrides"] += 1
                opponent["confident"] += z > CONFIDENT_Z
            if "t_choice_returned" in row and "t_request" in row:
                decision_s.append(row["t_choice_returned"] - row["t_request"])
    decision_s.sort()
    return {
        "errors": errors,
        "learner_trajectories": learner_povs,
        "categories": dict(categories),
        "by_opponent": {k: dict(v) for k, v in by_opponent.items()},
        "decision_s": {
            "n": len(decision_s),
            "mean": sum(decision_s) / len(decision_s) if decision_s else None,
            "max": decision_s[-1] if decision_s else None,
        },
    }
