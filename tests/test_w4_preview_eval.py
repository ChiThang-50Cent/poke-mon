"""Fast synthetic contracts for paired preview evaluation (no server/model load)."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_w4_team_preview",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_w4_team_preview.py",
)
eval_preview = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_preview)


def record(index, winner="player"):
    return {
        "matchup_index": index, "winner": winner, "error": None,
        "player_team": f"own-{index}", "opponent_team": f"opp-{index}",
        "player_team_sha256": f"own-hash-{index}",
        "opponent_team_sha256": f"opp-hash-{index}",
        "team_preview": {"source": "model", "chosen_lead": "greattusk"},
    }


def paired(a, t, **kwargs):
    return eval_preview.paired_bootstrap(
        a, t, battles=len(a), seed=17, resamples=2000, **kwargs,
    )


def test_bootstrap_aligns_by_index_not_completion_order():
    a = [record(0, "opponent"), record(1, "tie")]
    t = [record(1), record(0)]
    result = paired(a, t)
    assert result["valid"]
    assert result["win_rate_difference"] == 1.0
    assert result["bootstrap_95"] == [1.0, 1.0]
    assert result == paired(a, list(reversed(t)))


def test_bootstrap_is_seeded_and_counts_ties_as_nonwins():
    a = [record(i, "tie") for i in range(5)]
    t = [record(i, "player" if i < 2 else "opponent") for i in range(5)]
    result = paired(a, t)
    assert result == paired(a, t)
    assert result["win_rate_difference"] == pytest.approx(0.4)
    assert result["bootstrap_95"][0] <= 0.4 <= result["bootstrap_95"][1]


@pytest.mark.parametrize("field", ["player_team", "opponent_team",
                                   "player_team_sha256", "opponent_team_sha256"])
def test_pairing_rejects_changed_team_or_contents(field):
    a = [record(0)]
    t = copy.deepcopy(a)
    t[0][field] = "different"
    result = paired(a, t)
    assert not result["valid"]
    assert not result["team_pairs_equal"]
    assert result["completed_pairs"] == 0


def test_pairing_rejects_duplicate_index_and_missing_hash():
    a = [record(0), record(1)]
    t = [record(0), record(0)]
    assert not paired(a, t)["valid"]
    t = copy.deepcopy(a)
    del t[0]["player_team_sha256"]
    assert not paired(a, t)["valid"]


def test_error_pair_never_becomes_a_loss_or_a_valid_comparison():
    a = [record(0)]
    t = [record(0)]
    t[0]["error"] = "TimeoutError"
    result = paired(a, t)
    assert result["team_pairs_equal"]
    assert not result["valid"]
    assert result["completed_pairs"] == 0
    assert result["win_rate_difference"] is None
    assert result["bootstrap_95"] is None


def test_arm_requires_exact_counts_and_zero_errors_and_illegals():
    records = [record(0), record(1, "tie")]
    summary = {
        "attempted": 2, "completed": 2, "error_count": 0,
        "wins": 1, "losses": 0, "ties": 1,
        "neural": {"player": {"illegal_actions": 0}},
    }
    result = eval_preview.arm_report(summary, records, 2)
    assert result["valid"]
    assert result["win_rate"] == 0.5
    assert result["lead_source_counts"] == {"model": 2}
    assert result["top_lead_species"] == [{"species": "greattusk", "count": 2}]
    for field in ("attempted", "completed", "error_count", "wins"):
        mutated = copy.deepcopy(summary)
        mutated[field] += 1
        assert not eval_preview.arm_report(mutated, records, 2)["valid"]
    summary["neural"]["player"]["illegal_actions"] = 1
    assert not eval_preview.arm_report(summary, records, 2)["valid"]
    summary["neural"]["player"]["illegal_actions"] = 0
    records[0]["error"] = "protocol error"
    assert not eval_preview.arm_report(summary, records, 2)["valid"]


def test_cli_defaults_and_smoke_overrides():
    args = eval_preview.parse_args([])
    assert args.battles == 500
    assert args.concurrent == 8
    assert args.bootstrap_resamples >= 2000
    assert args.out == "reports/w4_team_preview.json"
    smoke = eval_preview.parse_args([
        "--battles", "20", "--opponents", "Grunt", "--out", "smoke.json",
        "--seed", "7", "--bootstrap-resamples", "2000", "--concurrent", "2",
    ])
    assert smoke.battles == 20 and smoke.opponents == ["Grunt"]
    assert smoke.seed == 7 and smoke.out == "smoke.json"


@pytest.mark.parametrize("argv", [
    ["--concurrent", "9"], ["--battles", "0"],
    ["--bootstrap-resamples", "1999"], ["--opponents", "Grunt", "Grunt"],
])
def test_cli_rejects_invalid_configuration(argv):
    with pytest.raises(SystemExit):
        eval_preview.parse_args(argv)


def test_prior_report_is_fingerprinted_without_changing_invalid_evidence(tmp_path):
    path = tmp_path / "run1.json"
    raw = json.dumps({
        "valid": False, "wall_time_s": 123.5,
        "opponents": {
            "Grunt": {"arms": {
                "A": {"valid": False, "error_count": 3, "illegal_actions": 1},
                "T": {"valid": False, "error_count": 1, "illegal_actions": 0},
            }},
        },
    }).encode()
    path.write_bytes(raw)
    metadata = eval_preview.prior_report_metadata(path)
    assert metadata["path"] == str(path)
    assert metadata["sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["valid"] is False
    assert metadata["errors"] == 4
    assert metadata["illegal_actions"] == 1
    assert metadata["wall_time_s"] == 123.5
    assert path.read_bytes() == raw
    args = eval_preview.parse_args(["--prior-report", str(path)])
    assert args.prior_report == str(path)


def test_expected_trapped_retries_are_reported_separately_from_illegal_actions():
    evidence = {"player": [{"matchup_index": 0, "error": "trapped"}],
                "opponent": []}
    summary = {
        "attempted": 1, "completed": 1, "error_count": 0,
        "wins": 1, "losses": 0, "ties": 0,
        "neural": {"player": {"illegal_actions": 0}},
        "protocol_retry_counts": {"player": 1, "opponent": 0},
        "protocol_retry_evidence": evidence,
    }
    result = eval_preview.arm_report(summary, [record(0)], 1)
    assert result["valid"]
    assert result["illegal_actions"] == 0
    assert result["protocol_retries"] == {"player": 1, "opponent": 0}
    assert result["protocol_retry_evidence"] == evidence
    assert result["trapped_rejected_choices"] == 1
