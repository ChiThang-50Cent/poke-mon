"""Battle-grouping, BC exclusions, and shared encoder behavior."""

import json
from pathlib import Path

import lz4.frame
import numpy as np
from metamon.interface import UniversalState

from pokeai.data.features import ObservationEncoder
from pokeai.data.split import split_battles
from pokeai.data.w3 import decision_validity


def test_split_groups_both_povs_and_is_order_independent() -> None:
    rows = [
        {"battle_id": f"gen9ou-{i}", "side": side}
        for i in range(100)
        for side in ("p1", "p2")
    ]
    splits = split_battles(rows, seed=42)
    assert splits == split_battles(list(reversed(rows)), seed=42)
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 80,
        "val": 10,
        "test": 10,
    }
    assignments = {battle: split for split, ids in splits.items() for battle in ids}
    assert len(assignments) == 100
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(splits[a]) & set(splits[b])
    grouped = {}
    for row in rows:
        grouped.setdefault(row["battle_id"], []).append(assignments[row["battle_id"]])
    assert all(len(set(povs)) == 1 and len(povs) == 2 for povs in grouped.values())


def test_validity_excludes_all_unusable_decisions() -> None:
    actions = np.array([0, 0, -1, 1, 0, 13, -2, 12])
    masks = np.zeros((8, 13), dtype=bool)
    masks[:, 0] = True
    masks[4] = False
    masks[7, 12] = True
    missing = np.array([False, True, False, False, False, False, False, False])
    valid, reasons = decision_validity(actions, masks, missing)
    np.testing.assert_array_equal(
        valid, [True, False, False, False, False, False, False, True]
    )
    assert reasons == {
        "missing": 1,
        "action_minus_one": 1,
        "action_out_of_range": 2,
        "chosen_not_in_mask": 2,
        "empty_mask": 1,
    }


def test_encoder_round_trip_and_native_token_equivalence(
    synthetic_run: tuple[Path, Path], tmp_path: Path
) -> None:
    run, _ = synthetic_run
    entry = json.loads((run / "manifest.json").read_text())["trajectory_files"][0]
    data = json.loads(lz4.frame.decompress(Path(entry["file"]).read_bytes()))
    state = UniversalState.from_dict(data["states"][0])
    encoder = ObservationEncoder()
    path = tmp_path / "encoder.json"
    encoder.save(path)
    restored = ObservationEncoder.load(path)
    before = encoder.encode(state)
    after = restored.encode(state)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    native = encoder.space.state_to_obs(state)
    np.testing.assert_array_equal(
        before["tokens"],
        np.where(
            native["text_tokens"] == -1, encoder.unknown_id, native["text_tokens"]
        ),
    )
    assert before["tokens"].shape == (87,)
    assert before["tokens"].dtype == np.int64
    assert before["numbers"].shape == (48,)
    assert before["numbers"].dtype == np.float32
    # An unseen online string must not grow the frozen vocabulary.
    state.player_active_pokemon.name = "not-a-real-pokemon-token"
    assert encoder.unknown_id in encoder.encode(state)["tokens"]
    assert encoder.vocab_size == len(encoder.tokenizer) + 1
