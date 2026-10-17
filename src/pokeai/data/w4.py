"""Recent human replay selection and temporal BC datasets using the W3 encoder."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pokeai.data.dataset import native_dataset
from pokeai.data.download import parse_replay_name
from pokeai.data.features import ObservationEncoder
from pokeai.data.w3 import _EncoderSpace, _rating_summary, encode_trajectory, load_split

_NATIVE: Any = None


def canonical_battle_id(name: str) -> str:
    """Strip only the upstream numeric prefix, retaining the actual battle ID."""
    battle = re.sub(r"^\d+-", "", Path(name).name.split("_", 1)[0])
    if not re.fullmatch(r"gen9ou-\d+", battle):
        raise ValueError(f"Invalid Gen9 OU battle ID: {name}")
    return battle


def prepare_entries(
    filenames: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate POVs before rejecting battles with incompatible outcomes."""
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    prefixed = 0
    duplicate_battles: set[str] = set()
    conflicting_results: set[str] = set()
    source_results: dict[str, list[str]] = defaultdict(list)
    for filename in sorted(
        filenames,
        key=lambda f: (bool(re.match(r"^\d+-", Path(f).name)), Path(f).name, f),
    ):
        name = Path(filename).name
        parts = name.removesuffix(".json.lz4").split("_")
        battle = canonical_battle_id(name)
        prefixed += bool(re.match(r"^\d+-", name))
        rating, month = parse_replay_name(name)
        if rating is None or rating < 1300:
            raise ValueError(f"Unexpected unfiltered replay: {name}")
        players = "_".join(parts[2:-2])
        if players.count("_vs_") != 1:
            raise ValueError(f"Invalid replay player delimiter: {name}")
        username, opponent = players.split("_vs_")
        result = parts[-1]
        if not username or not opponent or result not in {"WIN", "LOSS"}:
            raise ValueError(f"Invalid replay metadata: {name}")
        source_results[battle].append(result)
        key = battle, username
        row = {
            "file": filename,
            "battle_id": battle,
            "rating": rating,
            "month": month,
            "pov_username": username,
            "result": result,
        }
        if key in selected:
            duplicate_battles.add(battle)
            if selected[key]["result"] != result:
                conflicting_results.add(battle)
            if selected[key]["month"] != month:
                raise ValueError(f"Conflicting duplicate battle months: {battle}")
        else:
            selected[key] = row
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected.values():
        grouped[row["battle_id"]].append(row)
    inconsistent = sorted(
        conflicting_results
        | {
            b
            for b, rows in grouped.items()
            if len(rows) > 2
            or (len(rows) == 2 and len({r["result"] for r in rows}) != 2)
        }
    )
    bad = set(inconsistent)
    entries = sorted(
        (r for r in selected.values() if r["battle_id"] not in bad),
        key=lambda r: r["file"],
    )
    return entries, {
        "source_files": len(filenames),
        "prefixed_files": prefixed,
        "unprefixed_files": len(filenames) - prefixed,
        "duplicate_files_removed": len(filenames) - len(selected),
        "duplicate_battles": len(duplicate_battles),
        "duplicate_battles_both_povs": sum(
            len(grouped[b]) == 2 for b in duplicate_battles
        ),
        "duplicate_battles_single_pov": sum(
            len(grouped[b]) == 1 for b in duplicate_battles
        ),
        "pre_dedup_single_file_battles": sum(
            len(results) == 1 for results in source_results.values()
        ),
        "pre_dedup_two_file_same_result_battles": sum(
            len(results) == 2 and len(set(results)) == 1
            for results in source_results.values()
        ),
        "inconsistent_battle_ids": inconsistent,
        "inconsistent_battles": len(inconsistent),
        "inconsistent_trajectories_removed": sum(len(grouped[b]) for b in bad),
        "single_pov_battles": sum(
            len(rows) == 1 for b, rows in grouped.items() if b not in bad
        ),
    }


def temporal_split(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    assignments: dict[str, tuple[str, str]] = {}
    for row in entries:
        month = row["month"]
        if "2025-01" <= month <= "2025-12":
            split = "train"
        elif "2026-01" <= month <= "2026-02":
            split = "val"
        elif "2026-03" <= month <= "2026-05":
            split = "test"
        else:
            raise ValueError(f"Month outside temporal dataset: {month}")
        battle = canonical_battle_id(row["battle_id"])
        if battle in assignments and assignments[battle] != (split, month):
            raise ValueError(f"Conflicting battle months: {battle}")
        assignments[battle] = split, month
    return {
        s: sorted(b for b, (assigned, _) in assignments.items() if assigned == s)
        for s in ("train", "val", "test")
    }


def build_train_tiers(
    battle_ids: np.ndarray,
    valid_counts: np.ndarray,
    seed: int = 42,
    thresholds: tuple[int, ...] = (50000, 250000, 1000000),
) -> dict[str, Any]:
    """Seeded battle prefixes: all POVs stay together and tiers are nested."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battle in enumerate(battle_ids):
        grouped[str(battle)].append(index)
    order = sorted(
        grouped, key=lambda b: (hashlib.sha256(f"{seed}:{b}".encode()).hexdigest(), b)
    )
    indices: list[int] = []
    count = 0
    tiers: dict[str, Any] = {}
    targets = sorted(set(thresholds))
    for battle_index, battle in enumerate(order):
        indices.extend(grouped[battle])
        count += int(valid_counts[grouped[battle]].sum())
        for target in targets:
            if count >= target and str(target) not in tiers:
                tiers[str(target)] = {
                    "trajectory_indices": sorted(indices),
                    "valid_decisions": count,
                    "battles": battle_index + 1,
                    "trajectories": len(indices),
                }
    if any(str(t) not in tiers for t in targets):
        raise ValueError(
            f"Train data has only {count} valid decisions; tiers require {max(targets)}"
        )
    tiers["full"] = {
        "trajectory_indices": list(range(len(battle_ids))),
        "valid_decisions": count,
        "battles": len(order),
        "trajectories": len(battle_ids),
    }
    return tiers


def load_train_tier(
    tier: str, root: str | Path = "datasets/w4"
) -> dict[str, np.ndarray]:
    """Materialize a stored train tier, preserving complete trajectories."""
    root = Path(root)
    indices = json.loads((root / "train_tiers.json").read_text())["tiers"][tier][
        "trajectory_indices"
    ]
    arrays = load_split("train", root)
    offsets = arrays["offsets"]
    steps = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])
    result = {
        k: v[steps]
        for k, v in arrays.items()
        if k in {"tokens", "numbers", "masks", "actions", "valid"}
    }
    result.update({k: arrays[k][indices] for k in ("battle_ids", "ratings", "months")})
    result["offsets"] = np.concatenate(
        ([0], np.cumsum([offsets[i + 1] - offsets[i] for i in indices]))
    )
    return result


def _encode(filename: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    return encode_trajectory(_NATIVE, filename)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_w4_dataset(
    root: str | Path = ".cache/metamon/w4-human",
    output: str | Path = "datasets/w4",
    report_path: str | Path = "reports/w4_dataset_validation.json",
    seed: int = 42,
    workers: int = 4,
    w3_root: str | Path = "datasets/w3",
) -> dict[str, Any]:
    """Encode all retained recent POVs in parallel, then validate frozen splits."""
    global _NATIVE
    started = time.monotonic()
    root, output, w3_root = Path(root), Path(output), Path(w3_root)
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = (root / "stats.json").read_bytes()
    source = json.loads(source_bytes)
    encoder = ObservationEncoder()
    encoder.save(output / "encoder.json")
    encoder_equal = (output / "encoder.json").read_bytes() == (
        w3_root / "encoder.json"
    ).read_bytes()
    _NATIVE = native_dataset(root, human=True)
    discovered = {str(p.resolve()) for p in (root / "gen9ou").glob("*.json.lz4")}
    indexed = {str(Path(p).resolve()) for p in _NATIVE.filenames}
    if discovered != indexed:
        raise ValueError(
            f"Native index mismatch: {len(discovered - indexed)} missing, {len(indexed - discovered)} extra"
        )
    entries, selection = prepare_entries(sorted(_NATIVE.filenames))
    splits = temporal_split(entries)
    assignments = {b: s for s, ids in splits.items() for b in ids}
    for row in entries:
        row["split"] = assignments[row["battle_id"]]
    _write_json(output / "split_ids.json", splits)
    _NATIVE.observation_space = _EncoderSpace(encoder)
    reports = {}
    tiers: dict[str, Any] = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for split in ("train", "val", "test"):
            rows = [r for r in entries if r["split"] == split]
            chunks: dict[str, list[np.ndarray]] = {
                k: [] for k in ("tokens", "numbers", "masks", "actions", "valid")
            }
            offsets = [0]
            reasons: Counter[str] = Counter()
            valid_counts = []
            for encoded, excluded in pool.imap(
                _encode, (r["file"] for r in rows), chunksize=32
            ):
                for key, values in chunks.items():
                    values.append(encoded[key])
                reasons.update(excluded)
                offsets.append(offsets[-1] + len(encoded["actions"]))
                valid_counts.append(int(encoded["valid"].sum()))
            arrays = {k: np.concatenate(v) for k, v in chunks.items()}
            del chunks
            arrays.update(
                offsets=np.asarray(offsets, dtype=np.int64),
                battle_ids=np.asarray([r["battle_id"] for r in rows]),
                ratings=np.asarray([r["rating"] for r in rows], dtype=np.int32),
                months=np.asarray([r["month"] for r in rows]),
            )
            np.savez_compressed(output / f"{split}.npz", **arrays)
            unknown = int((arrays["tokens"] == encoder.unknown_id).sum())
            reports[split] = {
                "trajectories": len(rows),
                "battles": len(splits[split]),
                "decisions": len(arrays["actions"]),
                "valid_decisions": int(arrays["valid"].sum()),
                "excluded_decisions": int((~arrays["valid"]).sum()),
                "excluded_by_reason": dict(reasons),
                "reason_counts_overlap": True,
                "all_finite": bool(np.isfinite(arrays["numbers"]).all()),
                "unknown_tokens": unknown,
                "total_tokens": arrays["tokens"].size,
                "unknown_token_rate": unknown / arrays["tokens"].size,
                "ratings": _rating_summary([r["rating"] for r in rows]),
                "rating_buckets": dict(
                    sorted(
                        Counter(f"{r['rating'] // 100 * 100}+" for r in rows).items()
                    )
                ),
                "months": dict(sorted(Counter(r["month"] for r in rows).items())),
                "action_class_histogram": np.bincount(
                    arrays["actions"][arrays["valid"]], minlength=13
                ).tolist(),
            }
            if split == "train":
                numbers = arrays["numbers"].astype(np.float64)
                raw_std = numbers.std(axis=0)
                _write_json(
                    output / "feature_stats.json",
                    {
                        "mean": numbers.mean(axis=0).tolist(),
                        "std": np.where(raw_std > 0, raw_std, 1.0).tolist(),
                        "source_split": "train",
                        "decisions": len(numbers),
                        "constant_features": np.flatnonzero(raw_std == 0).tolist(),
                    },
                )
                del numbers
                tiers = build_train_tiers(
                    arrays["battle_ids"], np.asarray(valid_counts), seed
                )
                _write_json(
                    output / "train_tiers.json",
                    {
                        "seed": seed,
                        "selection": "sha256 seeded battle prefixes",
                        "tiers": tiers,
                    },
                )
            print(
                f"{split}: {len(rows)} trajectories, {reports[split]['valid_decisions']} valid decisions",
                flush=True,
            )
            del arrays
    overlap = {
        f"{a}_{b}": len(set(splits[a]) & set(splits[b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    old = json.loads((w3_root / "split_ids.json").read_text())
    old_overlap = {
        f"{a}_w3_{b}": len(set(ids) & {canonical_battle_id(i) for i in old_ids})
        for a, ids in splits.items()
        for b, old_ids in old.items()
    }
    invariants = {
        "native_index_complete": discovered == indexed,
        "encoder_equal_w3": encoder_equal,
        "zero_split_overlap": not any(overlap.values()),
        "zero_w3_overlap": not any(old_overlap.values()),
        "finite_features": all(r["all_finite"] for r in reports.values()),
        "nonempty_splits": all(r["valid_decisions"] > 0 for r in reports.values()),
        "tiers_meet_thresholds": all(
            tiers[str(t)]["valid_decisions"] >= t for t in (50000, 250000, 1000000)
        ),
        "source_archive_complete": source["completed_full_archive"],
        "source_files_match": selection["source_files"] == source["kept"],
        "source_months_match": dict(
            Counter(parse_replay_name(Path(f).name)[1] for f in _NATIVE.filenames)
        )
        == source["months"],
    }
    report = {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "seed": seed,
        "invariants": invariants,
        "wall_time_s": round(time.monotonic() - started, 3),
        "workers": workers,
        "filters": {
            "format": "gen9ou",
            "min_rating": 1300,
            "months": "2025-01..2026-05",
        },
        "data_source": {
            "url": source["source"],
            "etag": source["etag"],
            "stats_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "selection": selection,
        "native_index_files": len(indexed),
        "encoder": {
            "sha256": hashlib.sha256(
                (output / "encoder.json").read_bytes()
            ).hexdigest(),
            "equals_w3": encoder_equal,
        },
        "battle_id_overlap": overlap,
        "w3_battle_id_overlap": old_overlap,
        "splits": reports,
        "month_by_split": {
            m: {s: reports[s]["months"].get(m, 0) for s in reports}
            for m in sorted(source["months"])
        },
        "tiers": {
            k: {
                field: value
                for field, value in v.items()
                if field != "trajectory_indices"
            }
            for k, v in tiers.items()
        },
    }
    _write_json(Path(report_path), report)
    return report
