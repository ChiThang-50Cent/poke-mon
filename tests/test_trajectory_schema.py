"""Schema and fail-closed report mutation tests."""

import json
from pathlib import Path

import lz4.frame
import pytest

from pokeai.data.validate import read_native, schema_errors, validate_run


def first_entry(run: Path) -> dict:
    return json.loads((run / "manifest.json").read_text())["trajectory_files"][0]


def test_native_schema_and_report(synthetic_run):
    run, human = synthetic_run
    entry = first_entry(run)
    assert (
        schema_errors(
            read_native(Path(entry["file"])),
            json.loads(Path(entry["meta"]).read_text()),
        )
        == []
    )
    report = validate_run(run, human)
    assert report["overall_status"] == "PASS", report["errors"]
    assert report["total_decision_steps"] == 4
    assert report["saved_player_trajectories"] == 2


@pytest.mark.parametrize(
    "mutation", ["illegal", "missing_pov", "result", "mask", "missing_state"]
)
def test_corrupt_trajectory_fails(synthetic_run, mutation):
    run, human = synthetic_run
    entry = first_entry(run)
    file, meta_path = Path(entry["file"]), Path(entry["meta"])
    data = read_native(file)
    meta = json.loads(meta_path.read_text())
    if mutation == "illegal":
        data["actions"][0] = 12
        meta["chosen_actions"][0] = 12
    elif mutation == "missing_pov":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["trajectory_files"].pop()
        (run / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "result":
        meta["final_result"] = "loss"
    elif mutation == "mask":
        meta["action_masks"][0] = [1]
    elif mutation == "missing_state":
        data["states"].pop(0)
    file.write_bytes(lz4.frame.compress(json.dumps(data).encode()))
    meta_path.write_text(json.dumps(meta))
    report = validate_run(run, human)
    assert report["overall_status"] == "FAIL"
    if mutation == "illegal":
        assert report["invalid_action_count"] == 1
    elif mutation == "result":
        assert report["terminal_result_mismatches"] > 0
    elif mutation == "missing_state":
        assert report["missing_observation_count"] > 0
    else:
        assert report["schema_validation"] == "FAIL"


def test_missing_manifest_fails(tmp_path):
    assert validate_run(tmp_path)["overall_status"] == "FAIL"
