"""Frozen battle-grouped human BC arrays and auditable validation report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from metamon.interface import UniversalState

from pokeai.data.dataset import local_entries, native_dataset
from pokeai.data.download import parse_replay_name
from pokeai.data.features import ObservationEncoder
from pokeai.data.split import split_battles


class _EncoderSpace:
    def __init__(self, encoder: ObservationEncoder) -> None:
        self.encoder = encoder

    def reset(self) -> None:
        self.encoder.space.reset()

    def state_to_obs(self, state: UniversalState) -> dict[str, np.ndarray]:
        return self.encoder.encode(state)


def decision_validity(
    actions: np.ndarray, masks: np.ndarray, missing: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Return loss eligibility and overlapping diagnostic reason counts."""
    in_range = (actions >= 0) & (actions < 13)
    chosen_legal = masks[np.arange(len(actions)), np.clip(actions, 0, 12)]
    empty = ~masks.any(axis=-1)
    valid = ~missing & in_range & chosen_legal & ~empty
    return valid, {
        "missing": int(missing.sum()),
        "action_minus_one": int((actions == -1).sum()),
        "action_out_of_range": int(((actions < -1) | (actions >= 13)).sum()),
        "chosen_not_in_mask": int((in_range & ~chosen_legal).sum()),
        "empty_mask": int(empty.sum()),
    }


def encode_trajectory(
    native: Any, filename: str
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Shared W3/W4 native loading, decision alignment and legality contract."""
    observations, action_info, _, _ = native.load_filename(filename)
    actions = np.asarray(action_info["chosen"], dtype=np.int64)
    masks = np.zeros((len(actions), 13), dtype=bool)
    for step, legal in enumerate(action_info["legal"]):
        for action in legal:
            if 0 <= action < 13:
                masks[step, action] = True
    missing = np.asarray(action_info["missing"], dtype=bool)
    valid, excluded = decision_validity(actions, masks, missing)
    return {
        "tokens": np.asarray(observations["tokens"][:-1], dtype=np.int32),
        "numbers": np.asarray(observations["numbers"][:-1], dtype=np.float32),
        "masks": masks,
        "actions": actions,
        "valid": valid,
    }, excluded


def load_split(split: str, root: str | Path = "datasets/w3") -> dict[str, np.ndarray]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    with np.load(Path(root) / f"{split}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _rating_summary(ratings: list[int]) -> dict[str, Any]:
    return {
        "min": min(ratings),
        "max": max(ratings),
        "mean": float(np.mean(ratings)),
        "deciles": np.quantile(ratings, np.arange(1, 10) / 10).tolist(),
    }


def _live_gap(run: Path, encoder: ObservationEncoder) -> dict[str, Any]:
    entries = local_entries(run)[:4]
    native = native_dataset(run / "trajectories")
    native.observation_space = _EncoderSpace(encoder)
    unknown = total = decisions = 0
    for row in entries:
        observations, _, _, _ = native.load_filename(row["file"])
        tokens = np.asarray(observations["tokens"][:-1])
        unknown += int((tokens == encoder.unknown_id).sum())
        total += tokens.size
        decisions += len(tokens)
    return {
        "source_run": str(run),
        "backend": "poke-env UniversalState.from_Battle",
        "trajectories": len(entries),
        "decisions": decisions,
        "unknown_tokens": unknown,
        "total_tokens": total,
        "unknown_token_rate": unknown / total if total else 0.0,
        "note": "Measured only; backend sim2sim differences are not corrected.",
    }


def build_w3_dataset(
    root: str | Path = ".cache/metamon/w3-human",
    output: str | Path = "datasets/w3",
    report_path: str | Path = "reports/w3_dataset_validation.json",
    seed: int = 42,
    live_run: str | Path = "runs/20261016T074108-cd71eb26",
) -> dict[str, Any]:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample_bytes = (root / "sample.json").read_bytes()
    sample = json.loads(sample_bytes)
    encoder = ObservationEncoder()
    native = native_dataset(root, human=True)
    entries = []
    for filename in sorted(native.filenames):
        rating, month = parse_replay_name(Path(filename).name)
        if rating is None or rating < 1300:
            raise ValueError(f"Unexpected unfiltered trajectory: {filename}")
        entries.append(
            {
                "file": filename,
                "battle_id": Path(filename).name.split("_", 1)[0],
                "rating": rating,
                "month": month,
            }
        )
    splits = split_battles(entries, seed)
    assignments = {battle: split for split, ids in splits.items() for battle in ids}
    for row in entries:
        row["split"] = assignments[row["battle_id"]]
    (output / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in entries)
    )
    (output / "split_ids.json").write_text(json.dumps(splits, indent=2) + "\n")
    encoder.save(output / "encoder.json")

    # Explicitly prove the wrapper keeps Metamon's native IDs (except -1 -> UNK).
    raw = native._load_json(entries[0]["file"])
    compared = 0
    for state_dict in raw["states"][:8]:
        state = UniversalState.from_dict(state_dict)
        expected = encoder.space.state_to_obs(state)
        encoded = encoder.encode(state)
        native_tokens = np.asarray(expected["text_tokens"])
        expected_tokens = np.where(
            native_tokens == -1, encoder.unknown_id, native_tokens
        )
        if not np.array_equal(encoded["tokens"], expected_tokens):
            raise ValueError("Encoder IDs diverge from TokenizedObservationSpace")
        if not np.array_equal(encoded["numbers"], expected["numbers"]):
            raise ValueError("Encoder numbers diverge from TokenizedObservationSpace")
        compared += 1
    native.observation_space = _EncoderSpace(encoder)
    reports = {}
    all_finite = True
    for split in ("train", "val", "test"):
        rows = [row for row in entries if row["split"] == split]
        chunks: dict[str, list[np.ndarray]] = {
            key: [] for key in ("tokens", "numbers", "masks", "actions", "valid")
        }
        offsets = [0]
        reasons: Counter[str] = Counter()
        for row in rows:
            encoded, excluded = encode_trajectory(native, row["file"])
            reasons.update(excluded)
            for key, values in chunks.items():
                values.append(encoded[key])
            offsets.append(offsets[-1] + len(encoded["actions"]))
        arrays = {key: np.concatenate(values) for key, values in chunks.items()}
        arrays.update(
            offsets=np.asarray(offsets, dtype=np.int64),
            battle_ids=np.asarray([row["battle_id"] for row in rows]),
            ratings=np.asarray([row["rating"] for row in rows], dtype=np.int32),
        )
        np.savez_compressed(output / f"{split}.npz", **arrays)
        finite = bool(np.isfinite(arrays["numbers"]).all())
        all_finite &= finite
        unknown = int((arrays["tokens"] == encoder.unknown_id).sum())
        reports[split] = {
            "trajectories": len(rows),
            "battles": len(splits[split]),
            "decisions": len(arrays["actions"]),
            "valid_decisions": int(arrays["valid"].sum()),
            "excluded_decisions": int((~arrays["valid"]).sum()),
            "excluded_by_reason": dict(reasons),
            "reason_counts_overlap": True,
            "all_finite": finite,
            "unknown_tokens": unknown,
            "total_tokens": arrays["tokens"].size,
            "unknown_token_rate": unknown / arrays["tokens"].size,
            "ratings": _rating_summary([row["rating"] for row in rows]),
            "months": dict(Counter(row["month"] for row in rows)),
            "action_class_histogram": np.bincount(
                arrays["actions"][arrays["valid"]], minlength=13
            ).tolist(),
        }
        if split == "train":
            numbers = arrays["numbers"].astype(np.float64)
            mean = numbers.mean(axis=0)
            raw_std = numbers.std(axis=0)
            std = np.where(raw_std > 0, raw_std, 1.0)
            stats = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "source_split": "train",
                "decisions": len(numbers),
                "constant_features": np.flatnonzero(raw_std == 0).tolist(),
            }
            (output / "feature_stats.json").write_text(
                json.dumps(stats, indent=2) + "\n"
            )
        print(
            f"{split}: {reports[split]['trajectories']} trajectories, "
            f"{reports[split]['valid_decisions']} valid decisions",
            flush=True,
        )
    overlap = {
        f"{a}_{b}": len(set(splits[a]) & set(splits[b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    report = {
        "status": (
            "PASS"
            if (
                not any(overlap.values())
                and all_finite
                and reports["train"]["valid_decisions"] >= 50000
            )
            else "FAIL"
        ),
        "seed": seed,
        "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "filters": {
            "format": "gen9ou",
            "min_rating": 1300,
            "outcomes": "both",
            "source": "all downloaded pilot trajectories",
            "download_skipped": sample["skipped"],
        },
        "data_source": {
            "url": sample["source"],
            "revision": sample["revision"],
            "etag": sample["etag"],
            "sample_json_sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(
                (output / "index.jsonl").read_bytes()
            ).hexdigest(),
        },
        "encoder": encoder.metadata(),
        "native_encoding_equivalence_states": compared,
        "battle_id_overlap": overlap,
        "splits": reports,
        "train_serve_gap": _live_gap(Path(live_run), encoder),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
