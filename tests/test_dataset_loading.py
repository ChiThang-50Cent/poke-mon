"""Decision-aligned W3 loading and deterministic battle-level split coverage."""

import json
from pathlib import Path

import numpy as np

from pokeai.data import TrajectoryDataset
from pokeai.data.dataset import human_entries, local_entries, native_dataset
from pokeai.data.split import split_battles, write_dataset_index


def test_local_and_parsed_replay_interface(synthetic_run):
    run, human_root = synthetic_run
    for rows in (local_entries(run), human_entries(human_root)):
        sample = TrajectoryDataset(rows)[0]
        assert set(sample) == {
            "observations",
            "actions",
            "action_masks",
            "rewards",
            "dones",
            "missing",
            "battle_result",
            "battle_id",
            "player_side",
            "pov_username",
            "source",
        }
        assert sample["observations"]["numbers"].shape == (2, 48)
        assert sample["actions"].shape == (2,)
        assert sample["action_masks"].shape == (2, 13)
        assert sample["action_masks"].dtype == np.bool_
        assert sample["rewards"].shape == sample["dones"].shape == (2,)
        assert sample["missing"].shape == (2,)
        assert sample["dones"].tolist() == [False, True]
        assert sample["action_masks"][0, sample["actions"][0]]
        assert isinstance(sample["pov_username"], str)
        assert (
            sample["player_side"] is None
            if sample["source"] == "human"
            else sample["player_side"] in {"p1", "p2"}
        )


def test_native_loader_sees_no_sidecars(synthetic_run):
    run, _ = synthetic_run
    native = native_dataset(run / "trajectories")
    assert len(native) == 2
    assert all(name.endswith(".json.lz4") for name in native.filenames)


def test_missing_human_action_preserved(synthetic_run):
    _, human_root = synthetic_run
    import lz4.frame

    rows = human_entries(human_root)
    path = Path(rows[0]["file"])
    data = json.loads(lz4.frame.decompress(path.read_bytes()))
    data["actions"][0] = -1
    path.write_bytes(lz4.frame.compress(json.dumps(data).encode()))
    sample = TrajectoryDataset(rows)[0]
    assert sample["missing"][0]
    assert sample["actions"][0] == -1


def test_split_by_battle_and_seed(tmp_path):
    rows = [
        {"battle_id": f"battle-{battle}", "side": side, "source": source}
        for battle in range(30)
        for side in ("p1", "p2")
        for source in ("local", "human")
    ]
    splits = split_battles(rows, 17)
    assert splits == split_battles(list(reversed(rows)), 17)
    assert splits != split_battles(rows, 18)
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    write_dataset_index(rows, tmp_path, 17)
    index = [
        json.loads(line)
        for line in (tmp_path / "local_index.jsonl").read_text().splitlines()
    ]
    for battle in range(30):
        assert (
            len(
                {
                    row["split"]
                    for row in index
                    if row["battle_id"] == f"battle-{battle}"
                }
            )
            == 1
        )
    for split in ("train", "val", "test"):
        assert json.loads((tmp_path / f"{split}_ids.json").read_text()) == splits[split]
