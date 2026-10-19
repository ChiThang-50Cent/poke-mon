"""Fail-closed W5.1 native self-play QA and recurrent behavior replay.

This validator reads only the collected run and the checkpoints identified by its
sidecars. It never opens training/evaluation splits or starts simulator battles.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from metamon.interface import UniversalState

from pokeai.data.dataset import native_dataset, resolve_run_path
from pokeai.data.validate import (
    invalid_actions,
    leakage_errors,
    read_native,
    schema_errors,
)
from pokeai.models.pokedecision import load_pokedecision


def behavior_errors(data: dict, meta: dict) -> list[str]:
    """Check the recorded distribution, not merely the selected action."""
    errors = []
    behaviors = meta.get("behavior")
    actions = data.get("actions", [])[:-1]
    if not isinstance(behaviors, list) or len(behaviors) != len(actions):
        return ["Missing/unaligned behavior records"]
    for i, (action, behavior) in enumerate(zip(actions, behaviors)):
        prefix = f"behavior[{i}]"
        if not isinstance(behavior, dict):
            errors.append(f"{prefix}: not an object")
            continue
        required = {
            "probabilities",
            "chosen_index",
            "log_probability",
            "mode",
            "temperature",
            "mask",
            "policy_version",
            "checkpoint_sha256",
            "value",
            "fallback_reason",
        }
        if required - behavior.keys():
            errors.append(
                f"{prefix}: missing fields {sorted(required - behavior.keys())}"
            )
        # Search runs record C's choice in chosen_index and the sent action in executed_index.
        searched = "executed_index" in behavior
        executed = behavior["executed_index"] if searched else behavior.get("chosen_index")
        if executed != action:
            kind = "executed" if searched else "chosen"
            errors.append(f"{prefix}: {kind} index differs from native action")
        if searched and type(behavior.get("search_row")) is not int:
            errors.append(f"{prefix}: search step without search_row")
        action = behavior.get("chosen_index")
        masks = meta.get("action_masks", [])
        mask = behavior.get("mask")
        if (
            not isinstance(mask, list)
            or len(mask) != 13
            or any(type(x) is not int or x not in (0, 1) for x in mask)
            or i >= len(masks)
            or mask != masks[i]
        ):
            errors.append(f"{prefix}: invalid/different mask")
            continue
        if behavior.get("mode") not in {"sample", "argmax"}:
            errors.append(f"{prefix}: invalid mode")
        temperature = behavior.get("temperature")
        if not _finite(temperature) or temperature <= 0:
            errors.append(f"{prefix}: invalid temperature")
        for key in ("policy_version", "checkpoint_sha256"):
            if not meta.get(key) or behavior.get(key) != meta.get(key):
                errors.append(f"{prefix}: {key} differs/missing")
        if (
            "protocol_retry" in behavior
            and type(behavior["protocol_retry"]) is not bool
        ):
            errors.append(f"{prefix}: protocol_retry must be boolean")
        fallback = behavior.get("fallback_reason")
        if fallback:
            if (
                action != -1
                or any(mask)
                or behavior.get("probabilities") is not None
                or behavior.get("log_probability") is not None
                or behavior.get("value") is not None
                or fallback != meta.get("unsupported_reasons", [])[i]
            ):
                errors.append(f"{prefix}: inconsistent fallback")
            continue
        probabilities = behavior.get("probabilities")
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != 13
            or not all(_finite(p) and 0 <= p <= 1 for p in probabilities)
        ):
            errors.append(f"{prefix}: invalid probabilities")
            continue
        p = np.asarray(probabilities)
        if not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
            errors.append(f"{prefix}: probabilities do not sum to one")
        if np.any(p[np.asarray(mask) == 0] != 0):
            errors.append(f"{prefix}: probability outside mask")
        if type(action) is not int or not 0 <= action < 13 or not mask[action]:
            errors.append(f"{prefix}: chosen action outside mask")
            continue
        if searched and (type(executed) is not int or not 0 <= executed < 13 or not mask[executed]):
            errors.append(f"{prefix}: executed action outside mask")
        logp = behavior.get("log_probability")
        if (
            p[action] <= 0
            or not _finite(logp)
            or not np.isclose(logp, math.log(p[action]), atol=1e-5, rtol=0)
        ):
            errors.append(f"{prefix}: chosen log probability differs")
        if behavior.get("mode") == "argmax" and action != int(p.argmax()):
            errors.append(f"{prefix}: argmax choice differs")
        value = behavior.get("value")
        if value is not None and (not _finite(value) or not 0 <= value <= 1):
            errors.append(f"{prefix}: invalid value")
    preview = meta.get("team_preview")
    if not isinstance(preview, dict):
        errors.append("Missing team preview")
    else:
        if (
            preview.get("chosen_index") != -1
            or preview.get("fallback_reason") != "team_preview"
        ):
            errors.append("Preview must be separate from 13-slot behavior")
        probabilities = preview.get("probabilities")
        if probabilities is not None:
            p = np.asarray(probabilities, dtype=float).reshape(-1)
            if (
                len(p) != 6
                or not np.isfinite(p).all()
                or (p < 0).any()
                or not np.isclose(p.sum(), 1, atol=1e-6)
            ):
                errors.append("Invalid preview probabilities")
    return errors


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def replay_parity(
    data: dict,
    meta: dict,
    model: Any,
    encoder: Any,
    device: str = "cpu",
    decision_sink: Any = None,
) -> dict:
    """Replay each POV from its own initial memory, undoing rejected requests.

    A fallback does not call the model. A protocol retry is compared normally but
    its resulting memory is discarded before the corrected request is replayed.
    """
    errors = []
    hidden = None
    unknown = tokens_seen = compared = 0
    max_difference = 0.0
    with torch.inference_mode():
        for i, (state, behavior) in enumerate(
            zip(data["states"][:-1], meta["behavior"])
        ):
            encoded = encoder.encode(UniversalState.from_dict(copy.deepcopy(state)))
            unknown += int((encoded["tokens"] == encoder.unknown_id).sum())
            tokens_seen += int(encoded["tokens"].size)
            if behavior["fallback_reason"]:
                continue
            tokens = torch.as_tensor(encoded["tokens"], device=device).long()
            numbers = torch.as_tensor(encoded["numbers"], device=device).float()
            tokens, numbers = tokens[None, None], numbers[None, None]
            # Retain an independent snapshot even for models that mutate memory.
            before = hidden.clone() if hidden is not None else None
            if model.config.value_head:
                logits, value, next_hidden = model.forward_with_value(
                    tokens, numbers, hidden
                )
                expected_value = float(value[0, 0].sigmoid().item())
                if not _finite(behavior["value"]) or not np.isclose(
                    expected_value, behavior["value"], atol=2e-5, rtol=1e-5
                ):
                    errors.append(f"{i}: value parity")
            else:
                logits, next_hidden = model(tokens, numbers, hidden)
                if behavior["value"] is not None:
                    errors.append(f"{i}: value recorded for policy-only model")
            mask = torch.as_tensor(behavior["mask"], device=device).bool()
            scores = logits[0, 0].masked_fill(~mask, -torch.inf)
            expected = (
                torch.softmax(scores.float() / behavior["temperature"], -1)
                .cpu()
                .numpy()
            )
            actual = np.asarray(behavior["probabilities"])
            difference = float(np.max(np.abs(expected - actual)))
            max_difference = max(max_difference, difference)
            if not np.allclose(expected, actual, atol=2e-5, rtol=1e-5):
                errors.append(f"{i}: probability parity")
            compared += 1
            if decision_sink is not None:
                decision_sink(i, behavior, next_hidden)
            hidden = before if behavior.get("protocol_retry", False) else next_hidden
    return {
        "errors": errors,
        "compared_decisions": compared,
        "unknown_tokens": unknown,
        "tokens": tokens_seen,
        "max_probability_difference": max_difference,
    }


def validate_selfplay(
    run: str | Path,
    report_path: str | Path = "reports/w5_1_selfplay_validation.json",
    *,
    min_parity_trajectories: int = 200,
    device: str = "cpu",
) -> dict:
    """Validate every saved POV; PASS requires >=200 neural replay trajectories.

    ``min_parity_trajectories`` may raise, but cannot lower, the acceptance gate.
    Every neural POV is replayed, including frozen neural opponents, not just the
    learner. Results are written even when an artifact cannot be loaded.
    """
    run = Path(run)
    report: dict[str, Any] = {
        "stage": "W5.1",
        "overall_status": "FAIL",
        "errors": [],
        "native_dataset_load": "FAIL",
        "trajectories": 0,
        "neural_parity_trajectories": 0,
        "neural_parity_by_side": {},
        "compared_decisions": 0,
        "unknown_tokens": 0,
        "tokens": 0,
        "max_probability_difference": 0.0,
        "opponent_statistics": {},
        "battle_count": 0,
        "battle_error_count": 0,
        "model_illegal_actions": 0,
    }
    errors = report["errors"]
    loaded: dict[tuple[str, str], tuple[dict, dict]] = {}
    models = {}
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "trajectories": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "lengths": [],
            "actions": Counter(),
            "entropy": [],
            "fallbacks": Counter(),
        }
    )
    sides: Counter = Counter()
    try:
        manifest = json.loads((run / "manifest.json").read_text())
        battles = [
            json.loads(line)
            for line in (run / "battles.jsonl").read_text().splitlines()
            if line
        ]
        report["battle_count"] = len(battles)
        report["battle_error_count"] = sum(
            battle.get("error") is not None for battle in battles
        )
        summary = json.loads((run / "summary.json").read_text())
        neural_summary = summary.get("neural")
        if not isinstance(neural_summary, dict) or not neural_summary:
            errors.append("Missing neural summary")
        else:
            for role, stats in neural_summary.items():
                if (
                    isinstance(stats, dict)
                    and type(stats.get("illegal_actions")) is int
                ):
                    report["model_illegal_actions"] += stats["illegal_actions"]
                if (
                    not isinstance(stats, dict)
                    or type(stats.get("illegal_actions")) is not int
                    or stats["illegal_actions"] != 0
                ):
                    errors.append(f"Nonzero/missing neural illegal_actions: {role}")
        by_id = {b["battle_id"]: b for b in battles}
        entries = manifest["trajectory_files"]
        if (
            not battles
            or len(by_id) != len(battles)
            or manifest.get("missing")
            or manifest.get("expected_trajectories") != 2 * len(battles)
            or manifest.get("saved_trajectories") != len(entries)
            or len(entries) != 2 * len(battles)
        ):
            errors.append("Incomplete/duplicate battles or manifest counts")
        for b in battles:
            if b.get("error") is not None or b.get("winner") not in {
                "player",
                "opponent",
                "tie",
            }:
                errors.append(f"Incomplete/error battle: {b['battle_id']}")
        for entry in entries:
            identity = (entry["battle_id"], entry["player_side"])
            try:
                data = read_native(resolve_run_path(run, entry["file"]))
                meta = json.loads(resolve_run_path(run, entry["meta"]).read_text())
                if identity in loaded or identity != (
                    meta.get("battle_id"),
                    meta.get("player_side"),
                ):
                    errors.append(f"Duplicate/mismatched POV: {identity}")
                loaded[identity] = data, meta
                issues = schema_errors(data, meta)
                if invalid_actions(data, meta):
                    issues.append("Invalid action(s)")
                battle = by_id[identity[0]]
                winner = battle.get("winner")
                expected = (
                    "tie"
                    if winner == "tie"
                    else "win"
                    if (winner == "player") == meta.get("learner")
                    else "loss"
                )
                terminal = data["states"][-1]
                if (
                    meta.get("final_result") != expected
                    or bool(terminal.get("battle_won")) != (expected == "win")
                    or bool(terminal.get("battle_lost")) != (expected == "loss")
                ):
                    issues.append("Terminal result mismatch")
                if type(meta.get("learner")) is not bool or not meta.get(
                    "opponent_type"
                ):
                    issues.append("Missing learner/opponent provenance")
                neural = bool(meta.get("model_dir"))
                if meta.get("learner") and not neural:
                    issues.append("Learner lacks model_dir")
                if neural:
                    issues.extend(behavior_errors(data, meta))
                    model_dir = Path(meta["model_dir"])
                    digest = hashlib.sha256(
                        (model_dir / "best.pt").read_bytes()
                    ).hexdigest()
                    if digest != meta.get("checkpoint_sha256"):
                        issues.append("model_dir checkpoint digest mismatch")
                    if not issues:
                        key = str(model_dir.resolve())
                        if key not in models:
                            models[key] = load_pokedecision(model_dir, device)[:2]
                        parity = replay_parity(data, meta, *models[key], device=device)
                        issues.extend(parity["errors"])
                        if parity["compared_decisions"] == 0:
                            issues.append("Neural trajectory has no model decisions")
                        else:
                            report["neural_parity_trajectories"] += 1
                            sides[identity[1]] += 1
                        for name in ("compared_decisions", "unknown_tokens", "tokens"):
                            report[name] += parity[name]
                        report["max_probability_difference"] = max(
                            report["max_probability_difference"],
                            parity["max_probability_difference"],
                        )
                errors.extend(f"{identity}: {issue}" for issue in issues)
                group = groups[str(meta.get("opponent_type", "unknown"))]
                group["trajectories"] += 1
                if meta.get("learner"):
                    result_key = {
                        "win": "wins",
                        "loss": "losses",
                        "tie": "ties",
                    }[meta["final_result"]]
                    group[result_key] += 1
                    group["learner_decisions"] = (
                        group.get("learner_decisions", 0) + len(data["actions"]) - 1
                    )
                group["lengths"].append(len(data["actions"]) - 1)
                group["actions"].update(str(a) for a in data["actions"][:-1])
                for behavior in meta.get("behavior", []):
                    if not isinstance(behavior, dict):
                        continue
                    if behavior.get("fallback_reason"):
                        group["fallbacks"][str(behavior["fallback_reason"])] += 1
                    elif behavior.get("probabilities") is not None:
                        p = np.asarray(behavior["probabilities"], dtype=float)
                        if np.isfinite(p).all() and (p >= 0).all():
                            positive = p[p > 0]
                            entropy = float(-(positive * np.log(positive)).sum())
                            group["entropy"].append(entropy)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                RuntimeError,
                AssertionError,
            ) as exc:
                errors.append(f"{identity}: {type(exc).__name__}: {exc}")
        for battle_id in by_id:
            for side in ("p1", "p2"):
                if (battle_id, side) not in loaded:
                    errors.append(f"Missing POV: {battle_id}/{side}")
        for (battle_id, side), (data, meta) in loaded.items():
            other = loaded.get((battle_id, "p2" if side == "p1" else "p1"))
            try:
                errors.extend(
                    f"{battle_id}/{side}: {e}"
                    for e in leakage_errors(data, meta, *(other or (None, None)))
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                errors.append(f"{battle_id}/{side}: leakage check: {exc}")
        native = native_dataset(run / "trajectories")
        indexed = {str(Path(f).resolve()) for f in native.filenames}
        expected_files = {
            str(resolve_run_path(run, e["file"]).resolve()) for e in entries
        }
        if indexed != expected_files or not indexed:
            raise ValueError("Native dataset index differs from manifest")
        for filename in native.filenames:
            observations, action_info, rewards, dones = native.load_filename(filename)
            n = len(action_info["chosen"])
            if (
                n == 0
                or len(observations["numbers"]) != n + 1
                or len(rewards) != n
                or len(dones) != n
                or not dones[-1]
                or not np.isfinite(rewards).all()
            ):
                raise ValueError(f"Invalid native loaded trajectory: {filename}")
        report["native_dataset_load"] = "PASS"
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AssertionError,
    ) as exc:
        errors.append(f"Run/native loading: {type(exc).__name__}: {exc}")
    report["trajectories"] = len(loaded)
    report["neural_parity_by_side"] = dict(sides)
    if report["neural_parity_trajectories"] < max(200, min_parity_trajectories):
        errors.append("Offline parity requires at least 200 neural POV trajectories")
    if not sides["p1"] or not sides["p2"]:
        errors.append("Offline parity requires BOTH neural POVs")
    report["unknown_token_rate"] = (
        report["unknown_tokens"] / report["tokens"] if report["tokens"] else None
    )
    for opponent, group in groups.items():
        group["actions"] = dict(group["actions"])
        group["fallbacks"] = dict(group["fallbacks"])
        learner_battles = group["wins"] + group["losses"] + group["ties"]
        group["learner_battles"] = learner_battles
        group["learner_win_rate"] = (
            group["wins"] / learner_battles if learner_battles else None
        )
        for raw, name in (("lengths", "length"), ("entropy", "policy_entropy_nats")):
            values = group.pop(raw)
            group[name] = {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "p50": float(np.median(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
            }
        report["opponent_statistics"][opponent] = group
    if not errors and report["native_dataset_load"] == "PASS":
        report["overall_status"] = "PASS"
    report["status"] = report["overall_status"]
    report["error_count"] = len(errors)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/w5_1_selfplay_validation.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = validate_selfplay(args.run, args.report, device=args.device)
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "errors": len(report["errors"]),
            }
        )
    )
    raise SystemExit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
