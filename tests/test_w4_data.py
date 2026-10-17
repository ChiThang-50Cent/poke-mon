"""Synthetic checks for canonical replay groups, temporal holdouts, and tiers."""

import json
from pathlib import Path

import numpy as np
import pytest

from pokeai.data.w4 import (
    build_train_tiers,
    canonical_battle_id,
    load_train_tier,
    prepare_entries,
    temporal_split,
)


def _filename(
    battle: str,
    pov: str = "alice",
    opponent: str = "bob",
    result: str = "WIN",
    month: str = "2025-01",
    rating: int = 1500,
) -> str:
    year, number = month.split("-")
    return (
        f"gen9ou/{battle}_{rating}_{pov}_vs_{opponent}_"
        f"{number}-15-{year}_{result}.json.lz4"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gen9ou-123456", "gen9ou-123456"),
        ("1-gen9ou-123456", "gen9ou-123456"),
        ("0027-gen9ou-123456", "gen9ou-123456"),
        ("gen9ou-123457", "gen9ou-123457"),
    ],
)
def test_canonical_battle_id_removes_only_numeric_prefix(
    name: str, expected: str
) -> None:
    assert canonical_battle_id(name) == expected


def test_prepare_entries_prefers_unprefixed_then_filename() -> None:
    # A lexically earlier prefixed copy must not displace an unprefixed POV.
    alice = _filename("gen9ou-101", pov="alice_name", rating=1400)
    bob = _filename("2-gen9ou-101", "bob", "alice_name", "LOSS")
    filenames = [
        _filename("1-gen9ou-101", pov="alice_name", rating=1300),
        _filename("gen9ou-101", pov="alice_name", rating=1600),
        alice,
        _filename("3-gen9ou-101", "bob", "alice_name", "LOSS"),
        bob,
    ]
    entries, stats = prepare_entries(filenames)
    reversed_entries, reversed_stats = prepare_entries(list(reversed(filenames)))
    assert entries == reversed_entries
    assert stats == reversed_stats
    assert {row["file"] for row in entries} == {alice, bob}
    assert {row["battle_id"] for row in entries} == {"gen9ou-101"}
    assert {row["pov_username"] for row in entries} == {"alice_name", "bob"}
    assert {row["result"] for row in entries} == {"WIN", "LOSS"}
    assert {row["month"] for row in entries} == {"2025-01"}
    assert stats["duplicate_files_removed"] == 3
    alice_row = next(row for row in entries if row["pov_username"] == "alice_name")
    assert alice_row["rating"] == 1400
    assert Path(alice_row["file"]).name.endswith("_WIN.json.lz4")


@pytest.mark.parametrize("result", ["WIN", "LOSS"])
def test_prepare_entries_removes_whole_battle_for_inconsistent_povs(
    result: str,
) -> None:
    retained = _filename("gen9ou-202")
    entries, stats = prepare_entries(
        [
            _filename("gen9ou-201", result=result),
            _filename("4-gen9ou-201", "bob", "alice", result),
            retained,
        ]
    )
    assert [row["file"] for row in entries] == [retained]
    assert stats["inconsistent_battles"] == 1
    assert set(stats["inconsistent_battle_ids"]) == {"gen9ou-201"}


def test_prepare_entries_checks_outcomes_before_discarding_duplicates() -> None:
    entries, stats = prepare_entries(
        [
            _filename("gen9ou-301"),
            _filename("1-gen9ou-301", result="LOSS"),
            _filename("gen9ou-301", "bob", "alice", "LOSS"),
            _filename("gen9ou-302"),
        ]
    )
    assert {row["battle_id"] for row in entries} == {"gen9ou-302"}
    assert set(stats["inconsistent_battle_ids"]) == {"gen9ou-301"}


def test_selection_reconciles_pre_and_post_dedup_counts() -> None:
    _, stats = prepare_entries(
        [
            _filename("gen9ou-310"),
            _filename("1-gen9ou-310"),
            _filename("gen9ou-311"),
            _filename("gen9ou-311", "bob", "alice", "WIN"),
            _filename("gen9ou-312"),
        ]
    )
    assert stats["pre_dedup_two_file_same_result_battles"] == 2
    assert stats["pre_dedup_single_file_battles"] == 1
    assert stats["duplicate_battles_single_pov"] == 1
    assert stats["duplicate_battles_both_povs"] == 0
    assert stats["inconsistent_battles"] == 1
    assert stats["single_pov_battles"] == 2


def test_temporal_split_keeps_prefix_variants_and_both_povs_together() -> None:
    months = [
        ("2025-01", "train"),
        ("2025-12", "train"),
        ("2026-01", "val"),
        ("2026-02", "val"),
        ("2026-03", "test"),
        ("2026-05", "test"),
    ]
    filenames = []
    expected = {"train": [], "val": [], "test": []}
    for index, (month, split) in enumerate(months, start=401):
        battle = f"gen9ou-{index}"
        expected[split].append(battle)
        filenames.extend(
            [
                _filename(battle, month=month),
                _filename(f"1-{battle}", month=month),
                _filename(f"2-{battle}", "bob", "alice", "LOSS", month),
            ]
        )
    entries, _ = prepare_entries(filenames)
    splits = temporal_split(entries)
    assert splits == expected
    assert temporal_split(list(reversed(entries))) == splits
    assigned = [battle for battles in splits.values() for battle in battles]
    assert len(assigned) == len(set(assigned)) == len(months)
    assert len(entries) == 2 * len(months)
    for row in entries:
        assert sum(row["battle_id"] in ids for ids in splits.values()) == 1


def test_temporal_split_rejects_conflicting_months_for_same_battle() -> None:
    entries = [
        {"battle_id": "gen9ou-501", "month": "2025-12"},
        {"battle_id": "gen9ou-501", "month": "2026-01"},
    ]
    with pytest.raises(ValueError):
        temporal_split(entries)


@pytest.mark.parametrize("month", ["2024-12", "2026-06", "2027-01", "2025-13"])
def test_temporal_split_rejects_out_of_window_months(month: str) -> None:
    with pytest.raises(ValueError):
        temporal_split([{"battle_id": "gen9ou-502", "month": month}])


def test_train_tiers_are_deterministic_nested_whole_battle_groups() -> None:
    # Each battle contributes two unequal POVs; thresholds must count both.
    battle_ids = np.repeat([f"gen9ou-{index}" for index in range(12)], 2)
    valid_counts = np.tile([40_000, 60_000], 12)
    tiers = build_train_tiers(battle_ids, valid_counts, seed=42)
    repeated = build_train_tiers(battle_ids, valid_counts, seed=42)
    assert set(tiers) == {"50000", "250000", "1000000", "full"}
    previous: set[int] = set()
    for name in ("50000", "250000", "1000000", "full"):
        tier = tiers[name]
        indices = np.asarray(tier["trajectory_indices"], dtype=np.int64)
        np.testing.assert_array_equal(indices, repeated[name]["trajectory_indices"])
        selected = set(indices.tolist())
        assert len(selected) == len(indices)
        assert previous <= selected
        assert all(0 <= index < len(battle_ids) for index in selected)
        selected_battles = set(battle_ids[indices].tolist())
        for battle in selected_battles:
            assert set(np.flatnonzero(battle_ids == battle).tolist()) <= selected
        assert tier["valid_decisions"] == int(valid_counts[indices].sum())
        assert tier["trajectories"] == len(indices)
        assert tier["battles"] == len(selected_battles)
        if name != "full":
            assert tier["valid_decisions"] >= int(name)
        previous = selected
    assert previous == set(range(len(battle_ids)))
    assert tiers["full"]["valid_decisions"] == 1_200_000


def test_train_tiers_accept_custom_decision_thresholds() -> None:
    battle_ids = np.array(["gen9ou-601", "gen9ou-602", "gen9ou-601", "gen9ou-603"])
    valid_counts = np.array([2, 7, 3, 11])
    tiers = build_train_tiers(battle_ids, valid_counts, seed=9, thresholds=(5, 12))
    assert set(tiers) == {"5", "12", "full"}
    smaller = set(tiers["5"]["trajectory_indices"])
    larger = set(tiers["12"]["trajectory_indices"])
    assert smaller <= larger <= set(tiers["full"]["trajectory_indices"])
    assert tiers["5"]["valid_decisions"] >= 5
    assert tiers["12"]["valid_decisions"] >= 12
    assert tiers["full"]["valid_decisions"] == 23


def test_load_train_tier_preserves_decision_and_trajectory_alignment(
    tmp_path: Path,
) -> None:
    offsets = np.array([0, 2, 5, 6])
    arrays = {
        "tokens": np.arange(12).reshape(6, 2),
        "numbers": np.arange(18, dtype=np.float32).reshape(6, 3),
        "masks": np.ones((6, 13), dtype=bool),
        "actions": np.arange(6),
        "valid": np.ones(6, dtype=bool),
        "offsets": offsets,
        "battle_ids": np.array(["gen9ou-1", "gen9ou-2", "gen9ou-1"]),
        "ratings": np.array([1300, 1400, 1500]),
        "months": np.array(["2025-01", "2025-02", "2025-01"]),
    }
    np.savez_compressed(tmp_path / "train.npz", **arrays)
    (tmp_path / "train_tiers.json").write_text(
        json.dumps({"tiers": {"small": {"trajectory_indices": [0, 2]}}})
    )
    subset = load_train_tier("small", tmp_path)
    np.testing.assert_array_equal(subset["offsets"], [0, 2, 3])
    for key in ("tokens", "numbers", "masks", "actions", "valid"):
        np.testing.assert_array_equal(subset[key], arrays[key][[0, 1, 5]])
    for key in ("battle_ids", "ratings", "months"):
        np.testing.assert_array_equal(subset[key], arrays[key][[0, 2]])


def test_train_tiers_reject_insufficient_decisions() -> None:
    with pytest.raises(ValueError, match="only 1 valid decisions"):
        build_train_tiers(np.array(["gen9ou-1"]), np.array([1]))
