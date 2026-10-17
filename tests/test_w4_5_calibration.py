"""Fast synthetic calibration contracts, without models, server, or benchmark."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "w4_5_calibration",
    Path(__file__).resolve().parents[1] / "scripts/w4_5_calibration.py",
)
calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibration)


def record(index, values=(0.2, 0.4), winner="opponent"):
    return {
        "battle_id": f"test-{index}", "matchup_index": index,
        "winner": winner, "error": None, "values": {"player": list(values)},
        "player_team_sha256": f"own-{index}",
        "opponent_team_sha256": f"other-{index}",
    }


def test_weighted_metrics_do_not_average_predictions_before_scoring():
    trajectories = [(np.array([0.0, 1.0, 1.0]), 1), (np.array([0.0]), 0)]
    decision = calibration._Distribution(trajectories, "all", False)
    trajectory = calibration._Distribution(trajectories, "all", True)
    d = decision.report(np.random.default_rng(7), 30)
    t = trajectory.report(np.random.default_rng(7), 30)
    assert d["brier"] == pytest.approx(1 / 4)
    assert t["brier"] == pytest.approx(1 / 6)
    assert t["mean_probability"] == pytest.approx(1 / 3)
    assert t["std_probability"] == pytest.approx(np.sqrt(2 / 9))
    assert t["roc_auc"] == pytest.approx(5 / 6)
    assert t["mean_absolute_p_minus_half"] == .5
    assert sum(row["weight_fraction"] for row in t["reliability"]) == pytest.approx(1)
    json.dumps(t, allow_nan=False)


def test_phase_boundaries_first_last_and_empty_late_phase():
    trajectories = [(np.arange(33) / 40, 1), (np.array([.2]), 0)]
    expected = {"all": 34, "0-5": 7, "6-15": 10,
                "16-30": 15, "31+": 2, "first": 2, "last": 2}
    for phase, count in expected.items():
        dist = calibration._Distribution(trajectories, phase, False)
        assert len(dist.p) == count
    empty = calibration._Distribution([(np.array([.1]), 0)], "31+", True)
    report = empty.report(np.random.default_rng(1), 30)
    assert report["brier"] is None
    assert report["ci95"]["brier"]["defined_replicates"] == 0
    json.dumps(report, allow_nan=False)


def test_seeded_cluster_bootstrap_and_one_class_auc():
    dist = calibration._Distribution([(np.array([.1, .3]), 0)], "all", False)
    first = dist.report(np.random.default_rng(9), 30)
    assert first == dist.report(np.random.default_rng(9), 30)
    assert first["ci95"]["roc_auc"]["defined_replicates"] == 0
    assert first["ci95"]["brier"]["lower"] == pytest.approx(.05)
    assert first["ci95"]["brier"]["upper"] == pytest.approx(.05)


def test_paired_brier_difference_aligns_schedule_and_direction():
    groups = {"opponent": {
        "C": [record(0, (.1,)), record(1, (.9,), "player")],
        "D": [record(1, (.6,), "player"), record(0, (.4,))],
    }}
    result = calibration._brier_difference(groups, np.random.default_rng(5), 30)
    assert result["paired_non_tie_battles"] == 2
    assert result["estimate"]["decision_weighted"] == pytest.approx(-.15)
    assert result["estimate"]["trajectory_equally_weighted"] == pytest.approx(-.15)
    assert result["ci95"]["decision_weighted"]["lower"] == pytest.approx(-.15)
    groups["opponent"]["D"][0]["player_team_sha256"] = "different"
    with pytest.raises(ValueError, match="pairing mismatch"):
        calibration._brier_difference(groups, np.random.default_rng(5), 30)


def test_invalid_missing_values_fail_and_ties_are_counted():
    with pytest.raises(ValueError, match="Missing live values"):
        calibration._trajectories([record(0, ())])
    with pytest.raises(ValueError, match="finite"):
        calibration._trajectories([record(0, (float("nan"),))])
    with pytest.raises(ValueError, match="Duplicate"):
        calibration._trajectories([record(0), record(0)])
    trajectories, excluded = calibration._trajectories([record(0, winner="tie")])
    assert not trajectories
    assert excluded == {"ties": 1}
    with pytest.raises(ValueError, match="5000"):
        calibration.calibration_report({}, samples=4999)


def test_loading_full_benchmark_paths(tmp_path):
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(record(0)) + "\n")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"opponents": {"GymLeader": {"arms": {
        arm: {"battles_path": "full.jsonl"} for arm in ("C", "D")
    }}}}))
    groups, sources = calibration.load_benchmark(benchmark)
    assert groups["GymLeader"]["C"] == [record(0)]
    assert sources["GymLeader"]["D"] == str(path)
