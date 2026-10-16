"""Mutations must fail behavior QA, including recurrent replay corruption."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pokeai.data.features import ObservationEncoder
from pokeai.data.selfplay_validate import (
    behavior_errors,
    replay_parity,
    validate_selfplay,
)
from pokeai.data.validate import read_native


@pytest.fixture
def selfplay_run(synthetic_run):
    run, human = synthetic_run
    (run / "summary.json").write_text(
        json.dumps({"neural": {"player": {"illegal_actions": 0}}})
    )
    entries = json.loads((run / "manifest.json").read_text())["trajectory_files"]
    for entry in entries:
        path = Path(entry["meta"])
        meta = json.loads(path.read_text())
        meta.update(learner=meta["player_side"] == "p1", opponent_type="test")
        path.write_text(json.dumps(meta))
    return run, human


def behavior(action=0, *, fallback=None, retry=False):
    return {
        "probabilities": None if fallback else [1.0] + [0.0] * 12,
        "chosen_index": -1 if fallback else action,
        "log_probability": None if fallback else 0.0,
        "mode": "sample",
        "temperature": 1.0,
        "mask": [0] * 13 if fallback else [1] + [0] * 12,
        "policy_version": "test",
        "checkpoint_sha256": "a" * 64,
        "value": None,
        "fallback_reason": fallback,
        "protocol_retry": retry,
    }


def sample():
    return (
        {"actions": [0, -1]},
        {
            "behavior": [behavior()],
            "policy_version": "test",
            "checkpoint_sha256": "a" * 64,
            "action_masks": [[1] + [0] * 12],
            "unsupported_reasons": [None],
            "team_preview": {
                "chosen_index": -1,
                "fallback_reason": "team_preview",
                "probabilities": [1 / 6] * 6,
            },
        },
    )


def test_valid_behavior_distribution():
    assert behavior_errors(*sample()) == []


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("chosen_index", 1, "chosen index"),
        ("mask", [1] * 13, "mask"),
        ("probabilities", [0.5] + [0.0] * 12, "sum to one"),
        ("probabilities", [0.5, 0.5] + [0.0] * 11, "outside mask"),
        ("probabilities", [float("nan")] + [0.0] * 12, "invalid probabilities"),
        ("probabilities", [-0.1, 1.1] + [0.0] * 11, "invalid probabilities"),
        ("log_probability", -1.0, "log probability"),
        ("temperature", 0.0, "temperature"),
        ("mode", "random", "mode"),
        ("policy_version", "other", "policy_version"),
        ("checkpoint_sha256", "other", "checkpoint_sha256"),
        ("value", 2.0, "value"),
        ("protocol_retry", "true", "protocol_retry"),
    ],
)
def test_behavior_mutations(field, value, expected):
    data, meta = sample()
    meta["behavior"][0][field] = value
    assert any(expected in error for error in behavior_errors(data, meta))


def test_argmax_mutation():
    data, meta = sample()
    meta["action_masks"] = [[1, 1] + [0] * 11]
    meta["behavior"][0].update(
        mask=meta["action_masks"][0],
        mode="argmax",
        probabilities=[0.25, 0.75] + [0.0] * 11,
        log_probability=math.log(0.25),
    )
    assert any("argmax" in e for e in behavior_errors(data, meta))


def test_fallback_mutation():
    data, meta = sample()
    data["actions"][0] = -1
    meta["action_masks"] = [[0] * 13]
    meta["unsupported_reasons"] = ["struggle"]
    meta["behavior"] = [behavior(fallback="struggle")]
    assert behavior_errors(data, meta) == []
    meta["behavior"][0]["probabilities"] = [1.0] + [0.0] * 12
    assert any("fallback" in e for e in behavior_errors(data, meta))


def test_preview_and_alignment_mutations():
    data, meta = sample()
    meta["team_preview"]["chosen_index"] = 0
    assert any("Preview" in e for e in behavior_errors(data, meta))
    meta["behavior"] = []
    assert any("unaligned" in e for e in behavior_errors(data, meta))


class RecurrentProbe:
    """Known memory-dependent logits make unintended resets observable."""

    config = SimpleNamespace(value_head=False)

    def __init__(self):
        self.calls = 0

    def __call__(self, tokens, numbers, hidden):
        self.calls += 1
        previous = 0.0 if hidden is None else float(hidden.item())
        logits = torch.zeros((1, 1, 13))
        logits[0, 0, 0] = previous
        return logits, torch.tensor(previous + 1)


def replay_sample(synthetic_run):
    run, _ = synthetic_run
    entry = json.loads((run / "manifest.json").read_text())["trajectory_files"][0]
    state = read_native(Path(entry["file"]))["states"][0]
    # Memory before model steps is 0, 1 (rejected), 1 (corrected), 2.
    records = []
    for memory, fallback, retry in (
        (0, None, False),
        (None, "struggle", False),
        (1, None, True),
        (1, None, False),
        (2, None, False),
    ):
        b = behavior(fallback=fallback, retry=retry)
        if not fallback:
            p = torch.softmax(torch.tensor([float(memory), 0.0]), 0).tolist()
            b.update(
                mask=[1, 1] + [0] * 11,
                probabilities=p + [0.0] * 11,
                log_probability=math.log(p[0]),
            )
        records.append(b)
    return {"states": [copy.deepcopy(state) for _ in range(6)]}, {"behavior": records}


def test_parity_preserves_memory_skips_fallback_and_rolls_back_retry(synthetic_run):
    data, meta = replay_sample(synthetic_run)
    model = RecurrentProbe()
    result = replay_parity(data, meta, model, ObservationEncoder())
    assert result["errors"] == []
    assert result["compared_decisions"] == 4
    assert model.calls == 4
    assert result["tokens"] == 5 * 87


def test_parity_feature_sink_observes_continuous_post_decision_memory(synthetic_run):
    data, meta = replay_sample(synthetic_run)
    observed = []

    def collect(index, behavior_record, hidden):
        observed.append((index, behavior_record["protocol_retry"], hidden.item()))

    result = replay_parity(
        data, meta, RecurrentProbe(), ObservationEncoder(), decision_sink=collect
    )
    assert result["errors"] == []
    assert observed == [(0, False, 1), (2, True, 2), (3, False, 2), (4, False, 3)]


def test_parity_retry_flag_mutation(synthetic_run):
    data, meta = replay_sample(synthetic_run)
    meta["behavior"][2]["protocol_retry"] = False
    result = replay_parity(data, meta, RecurrentProbe(), ObservationEncoder())
    assert any("probability parity" in e for e in result["errors"])


def test_parity_distribution_mutation(synthetic_run):
    data, meta = replay_sample(synthetic_run)
    meta["behavior"][4]["probabilities"] = [0.5, 0.5] + [0.0] * 11
    result = replay_parity(data, meta, RecurrentProbe(), ObservationEncoder())
    assert any("4: probability parity" in e for e in result["errors"])


def test_report_cannot_pass_without_full_parity(selfplay_run, tmp_path):
    run, _ = selfplay_run
    report_path = tmp_path / "qa.json"
    report = validate_selfplay(run, report_path, min_parity_trajectories=0)
    assert report["overall_status"] == "FAIL"
    assert any("200" in e for e in report["errors"])
    assert any("BOTH" in e for e in report["errors"])
    assert json.loads(report_path.read_text())["overall_status"] == "FAIL"


def test_missing_pov_mutation(selfplay_run, tmp_path):
    run, _ = selfplay_run
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trajectory_files"].pop()
    manifest_path.write_text(json.dumps(manifest))
    report = validate_selfplay(run, tmp_path / "qa.json")
    assert any("Missing POV" in e for e in report["errors"])
    assert report["overall_status"] == "FAIL"


def test_checkpoint_digest_mutation(selfplay_run, tmp_path):
    run, _ = selfplay_run
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "best.pt").write_bytes(b"mutated-checkpoint")
    entries = json.loads((run / "manifest.json").read_text())["trajectory_files"]
    for entry in entries:
        path = Path(entry["meta"])
        meta = json.loads(path.read_text())
        meta.update(
            model_dir=str(checkpoint),
            checkpoint_sha256="a" * 64,
            policy_version="test",
            learner=True,
            opponent_type="neural",
            behavior=[behavior(), behavior()],
            team_preview={
                "chosen_index": -1,
                "fallback_reason": "team_preview",
                "probabilities": None,
            },
        )
        path.write_text(json.dumps(meta))
    report = validate_selfplay(run, tmp_path / "qa.json")
    assert report["overall_status"] == "FAIL"
    assert any("checkpoint digest mismatch" in e for e in report["errors"])


def test_parity_hidden_reset_mutation(synthetic_run):
    class ResetProbe(RecurrentProbe):
        def __call__(self, tokens, numbers, hidden):
            return super().__call__(tokens, numbers, None)

    data, meta = replay_sample(synthetic_run)
    result = replay_parity(data, meta, ResetProbe(), ObservationEncoder())
    assert any("probability parity" in e for e in result["errors"])


def test_parity_value_mutation(synthetic_run):
    class ValueProbe(RecurrentProbe):
        config = SimpleNamespace(value_head=True)

        def forward_with_value(self, tokens, numbers, hidden):
            logits, next_hidden = self(tokens, numbers, hidden)
            return logits, torch.zeros((1, 1)), next_hidden

    data, meta = replay_sample(synthetic_run)
    for record in meta["behavior"]:
        if not record["fallback_reason"]:
            record["value"] = 0.5
    assert replay_parity(data, meta, ValueProbe(), ObservationEncoder())["errors"] == []
    meta["behavior"][4]["value"] = 0.9
    result = replay_parity(data, meta, ValueProbe(), ObservationEncoder())
    assert any("4: value parity" in e for e in result["errors"])


def test_illegal_action_summary_mutation(selfplay_run, tmp_path):
    run, _ = selfplay_run
    (run / "summary.json").write_text(
        json.dumps({"neural": {"player": {"illegal_actions": 1}}})
    )
    report = validate_selfplay(run, tmp_path / "qa.json")
    assert any("illegal_actions" in e for e in report["errors"])
    assert report["status"] == report["overall_status"] == "FAIL"
    assert report["battle_count"] == 1
    assert report["battle_error_count"] == 0
    assert report["model_illegal_actions"] == 1
    assert report["error_count"] == len(report["errors"])


def test_learner_p2_and_learner_only_stats(selfplay_run, tmp_path):
    run, _ = selfplay_run
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["trajectory_files"]:
        side = "p2" if entry["player_side"] == "p1" else "p1"
        entry["player_side"] = side
        path = Path(entry["meta"])
        meta = json.loads(path.read_text())
        meta["player_side"] = side
        path.write_text(json.dumps(meta))
    manifest_path.write_text(json.dumps(manifest))
    report = validate_selfplay(run, tmp_path / "qa.json")
    assert not any("Terminal result mismatch" in e for e in report["errors"])
    stats = report["opponent_statistics"]["test"]
    assert stats["learner_battles"] == 1
    assert stats["learner_win_rate"] == 1
    assert stats["learner_decisions"] == 2
    assert stats["trajectories"] == 2


def test_battle_error_count_mutation(selfplay_run, tmp_path):
    run, _ = selfplay_run
    path = run / "battles.jsonl"
    battle = json.loads(path.read_text())
    battle["error"] = "protocol error"
    path.write_text(json.dumps(battle) + "\n")
    report = validate_selfplay(run, tmp_path / "qa.json")
    assert report["battle_count"] == report["battle_error_count"] == 1
    assert report["model_illegal_actions"] == 0
    assert report["status"] == "FAIL"
    assert report["error_count"] == len(report["errors"])
