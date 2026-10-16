"""Outcome identity tests use synthetic raw fields, never an encoder or holdout."""

import json
from pathlib import Path

import lz4.frame
import numpy as np
import pytest

from pokeai.data.w4 import prepare_entries, temporal_split
from pokeai.data.w4_outcomes import build_outcomes, load_outcomes


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source, root = tmp_path / "source", tmp_path / "data"
    (source / "gen9ou").mkdir(parents=True)
    root.mkdir()
    for month, battle in (("01-15-2025", 1), ("01-15-2026", 2), ("03-15-2026", 3)):
        for pov, opponent, result, hp in (
            ("alice", "bob", "WIN", .75), ("bob", "alice", "LOSS", .25)
        ):
            path = source / "gen9ou" / (
                f"gen9ou-{battle}_1500_{pov}_vs_{opponent}_{month}_{result}.json.lz4"
            )
            raw = {
                "states": [{
                    "opponents_remaining": 6,
                    "player_active_pokemon": {"hp_pct": hp},
                    "opponent_active_pokemon": {"hp_pct": 1 - hp},
                }] * 3,
                "actions": [1, 2, -1],
            }
            # A holdout body cannot be decoded; construction must not open it.
            path.write_bytes(
                b"never open holdout" if battle == 3 else
                lz4.frame.compress(json.dumps(raw).encode())
            )
    entries, _ = prepare_entries([str(p) for p in (source / "gen9ou").iterdir()])
    splits = temporal_split(entries)
    (root / "split_ids.json").write_text(json.dumps(splits))
    for split in ("train", "val"):
        rows = [r for r in entries if r["battle_id"] in splits[split]]
        numbers = np.zeros((4, 48), dtype=np.float32)
        numbers[:, 0] = 1
        numbers[:, 1] = [.75, .75, .25, .25]
        numbers[:, 33] = [.25, .25, .75, .75]
        np.savez(
            root / f"{split}.npz",
            tokens=np.zeros((4, 87), dtype=np.int32), numbers=numbers,
            masks=np.ones((4, 13), dtype=bool), actions=np.array([1, 2, 1, 2]),
            valid=np.ones(4, dtype=bool), offsets=np.array([0, 2, 4]),
            battle_ids=np.asarray([r["battle_id"] for r in rows]),
            ratings=np.array([1500, 1500]), months=np.asarray([r["month"] for r in rows]),
        )
    return root, source


def _load(root: Path, split: str = "train") -> dict[str, np.ndarray]:
    with np.load(root / f"{split}.npz") as data:
        return {key: data[key] for key in data.files}


def test_outcomes_builds_only_train_val_with_pov_evidence(tmp_path: Path) -> None:
    root, source = _fixture(tmp_path)
    sidecar = build_outcomes(root, source)
    assert set(sidecar["splits"]) == {"train", "val"}
    assert "gen9ou-3" not in (root / "outcomes.json").read_text()
    for split in ("train", "val"):
        labels = load_outcomes(root, split, _load(root, split))
        np.testing.assert_array_equal(labels, [1., 0.])
        assert labels.dtype == np.float32
        rows = sidecar["splits"][split]["trajectories"]
        assert [r["pov_username"] for r in rows] == ["alice", "bob"]
        assert [r["result"] for r in rows] == ["WIN", "LOSS"]
        assert all(r["rating"] == 1500 and r["turns"] is None for r in rows)
    with pytest.raises(ValueError, match="restricted"):
        load_outcomes(root, "test", {})


@pytest.mark.parametrize("field", ["battle_ids", "ratings", "months", "offsets"])
def test_build_rejects_metadata_mismatch(tmp_path: Path, field: str) -> None:
    root, source = _fixture(tmp_path)
    arrays = _load(root)
    arrays[field][0] = 99 if arrays[field].dtype.kind in "iu" else "wrong"
    np.savez(root / "train.npz", **arrays)
    with pytest.raises(ValueError, match="mismatch"):
        build_outcomes(root, source)
    assert not (root / "outcomes.json").exists()


def test_build_rejects_same_battle_same_rating_swapped_povs(tmp_path: Path) -> None:
    root, source = _fixture(tmp_path)
    arrays = _load(root)
    arrays["numbers"] = arrays["numbers"][[2, 3, 0, 1]]
    np.savez(root / "train.npz", **arrays)
    with pytest.raises(ValueError, match="POV/HP"):
        build_outcomes(root, source)


@pytest.mark.parametrize("field", ["tokens", "numbers", "masks", "actions", "valid"])
def test_load_rejects_any_changed_sealed_array(tmp_path: Path, field: str) -> None:
    root, source = _fixture(tmp_path)
    build_outcomes(root, source)
    arrays = _load(root)
    arrays[field].flat[0] = not arrays[field].flat[0] if arrays[field].dtype == bool else 9
    with pytest.raises(ValueError, match="hash mismatch"):
        load_outcomes(root, "train", arrays)


def test_load_rejects_swapped_pov_and_changed_label(tmp_path: Path) -> None:
    root, source = _fixture(tmp_path)
    build_outcomes(root, source)
    arrays = _load(root)
    arrays["numbers"] = arrays["numbers"][[2, 3, 0, 1]]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_outcomes(root, "train", arrays)
    path = root / "outcomes.json"
    sidecar = json.loads(path.read_text())
    sidecar["splits"]["train"]["outcomes"].reverse()
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="result/label mismatch"):
        load_outcomes(root, "train", _load(root))


def test_load_rejects_same_battle_identity_order_change(tmp_path: Path) -> None:
    root, source = _fixture(tmp_path)
    build_outcomes(root, source)
    path = root / "outcomes.json"
    sidecar = json.loads(path.read_text())
    sidecar["splits"]["train"]["trajectories"].reverse()
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="identity order mismatch"):
        load_outcomes(root, "train", _load(root))


def test_build_rejects_raw_action_mismatch(tmp_path: Path) -> None:
    root, source = _fixture(tmp_path)
    arrays = _load(root)
    arrays["actions"][0] = 3
    np.savez(root / "train.npz", **arrays)
    with pytest.raises(ValueError, match="POV/action"):
        build_outcomes(root, source)
