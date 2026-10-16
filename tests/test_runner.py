"""Check battle metrics offline and the local battle protocol end to end."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from pokeai.env.runner import RunConfig, run_battles, summarize, wilson_interval


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(50, 100) == pytest.approx((0.40383153, 0.59616847))
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-15)
    assert high == pytest.approx(0.2775328)
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7224672)
    assert high == pytest.approx(1.0)


@pytest.mark.parametrize("wins,total", [(-1, 1), (2, 1), (0, -1)])
def test_wilson_rejects_invalid_counts(wins: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(wins, total)


def test_summary_excludes_errors_from_outcome_rates() -> None:
    records = [
        {"winner": "player", "error": None, "decisions_player": 20},
        {"winner": "opponent", "error": None, "decisions_player": 30},
        {"winner": "tie", "error": None, "decisions_player": 40},
        {"winner": None, "error": "timeout", "decisions_player": 10},
    ]
    summary = summarize(records, 2.0, {"concurrent": 2})
    assert summary["attempted"] == 4
    assert summary["completed"] == 3
    assert summary["error_count"] == 1
    assert summary["wins"] == summary["losses"] == summary["ties"] == 1
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["lose_rate"] == pytest.approx(1 / 3)
    assert summary["tie_rate"] == pytest.approx(1 / 3)
    assert summary["win_rate_wilson_95"] == list(wilson_interval(1, 3))
    assert summary["games_per_s"] == 1.5
    assert summary["decisions_player"] == 100
    assert summary["decisions_per_s"] == 50
    assert summary["config"] == {"concurrent": 2}


def test_empty_summary() -> None:
    summary = summarize([], 0.0, {})
    assert summary["attempted"] == summary["completed"] == 0
    assert summary["games_per_s"] == summary["decisions_per_s"] == 0
    assert summary["win_rate_wilson_95"] == [0.0, 1.0]


@pytest.mark.parametrize(
    "options",
    [
        {"battles": 0},
        {"concurrent": 0},
        {"timeout_s": -1},
        {"timeout_s": float("inf")},
        {"timeout_s": float("nan")},
    ],
)
def test_invalid_config(options: dict) -> None:
    with pytest.raises(ValueError):
        RunConfig(**options)


@pytest.mark.parametrize("save_trajectories", [False, True])
def test_timeouts_are_logged_and_run_continues(
    tmp_path, monkeypatch, save_trajectories
) -> None:
    from pokeai.env import runner, teams

    async def stop_listening() -> None:
        return None

    class StuckBaseline:
        decisions = 0

        def __init__(self, **kwargs) -> None:
            self.battles = {}
            self.ps_client = SimpleNamespace(stop_listening=stop_listening)

        def update_team(self, team: str) -> None:
            pass

        def randomize(self) -> None:
            pass

        async def battle_against(self, opponent, n_battles: int) -> None:
            self.last_server_error = (
                "|error|[Invalid choice] Can't switch: "
                "You have to pass to a fainted Pokémon"
            )
            await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_counted_baseline", lambda name: StuckBaseline)
    monkeypatch.setattr(
        teams,
        "load_teamset",
        lambda path: SimpleNamespace(
            yield_team=lambda: "Garchomp||||earthquake",
            most_recent_team_file="test.gen9ou_team",
        ),
    )
    summary = asyncio.run(
        runner.run_battles(RunConfig(
            battles=3, concurrent=2, timeout_s=0.01, out=str(tmp_path),
            save_trajectories=save_trajectories,
        ))
    )
    records = [
        json.loads(line)
        for line in (tmp_path / summary["run_id"] / "battles.jsonl")
        .read_text().splitlines()
    ]
    assert len(records) == summary["error_count"] == 3
    assert summary["completed"] == 0
    assert all(record["error"].startswith("TimeoutError:") for record in records)
    assert all("You have to pass to a fainted Pokémon" in r["error"] for r in records)
    assert all(record["winner"] is None for record in records)
    output = tmp_path / summary["run_id"]
    if save_trajectories:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["expected_trajectories"] == 6
        assert manifest["saved_trajectories"] == 0
        assert len(manifest["missing"]) == 6
        assert not (output / "trajectories").exists()
    else:
        assert not (output / "manifest.json").exists()


def test_revival_request_filters_healthy_reserves() -> None:
    from pokeai.env.runner import _prepare_revival_request

    healthy = SimpleNamespace(fainted=False)
    fainted = SimpleNamespace(fainted=True)
    battle = SimpleNamespace(
        reviving=True, available_switches=[healthy, fainted, healthy]
    )
    original_list = battle.available_switches
    _prepare_revival_request(battle)
    assert battle.available_switches == [fainted]
    assert battle.available_switches is original_list


def test_ordinary_switch_request_keeps_healthy_reserves() -> None:
    from pokeai.env.runner import _prepare_revival_request

    healthy = SimpleNamespace(fainted=False)
    battle = SimpleNamespace(reviving=False, available_switches=[healthy])
    _prepare_revival_request(battle)
    assert battle.available_switches == [healthy]


@pytest.mark.server
def test_real_battles(tmp_path) -> None:
    summary = asyncio.run(
        run_battles(
            RunConfig(battles=4, concurrent=2, timeout_s=60, out=str(tmp_path))
        )
    )
    output = tmp_path / summary["run_id"]
    records = [
        json.loads(line)
        for line in (output / "battles.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    assert summary["completed"] == 4
    assert summary["error_count"] == 0
    assert len({record["battle_id"] for record in records}) == 4
    for record in records:
        assert record["error"] is None
        assert record["winner"] in {"player", "opponent", "tie"}
        assert record["turns"] > 0
        assert record["decisions_player"] > 0
        assert record["duration_s"] > 0
        assert record["player_team"].endswith(".gen9ou_team")
        assert record["opponent_team"].endswith(".gen9ou_team")
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.server
def test_real_trajectory_collection(tmp_path) -> None:
    import lz4.frame
    from metamon.data import MetamonDataset
    from metamon.interface import (
        DefaultActionSpace,
        DefaultObservationSpace,
        DefaultShapedReward,
    )

    summary = asyncio.run(run_battles(RunConfig(
        player="Grunt", opponent="GymLeader", battles=4, concurrent=2,
        timeout_s=60, out=str(tmp_path), save_trajectories=True,
    )))
    output = tmp_path / summary["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["completed"] == 4
    assert manifest["expected_trajectories"] == manifest["saved_trajectories"] == 8
    assert manifest["missing"] == []
    records = {
        item["battle_id"]: item
        for item in map(json.loads, (output / "battles.jsonl").read_text().splitlines())
    }
    for entry in manifest["trajectory_files"]:
        meta = json.loads((output / entry["meta"]).read_text())
        with lz4.frame.open(output / entry["file"], "rb") as file:
            native = json.loads(file.read())
        assert len(native["states"]) == len(native["actions"]) == meta["n_steps"] + 1
        assert native["actions"][-1] == -1
        assert all(
            action == -1 or mask[action]
            for action, mask in zip(meta["chosen_actions"], meta["action_masks"])
        )
        assert all(
            (action == -1) == (reason is not None)
            for action, reason in zip(
                meta["chosen_actions"], meta["unsupported_reasons"]
            )
        )
        terminal = native["states"][-1]
        assert terminal["battle_won"] == (meta["final_result"] == "win")
        assert terminal["battle_lost"] == (meta["final_result"] == "loss")
        assert records[meta["battle_id"]]["winner"] in {"player", "opponent", "tie"}
        assert all(event[1] != "request" for event in meta["public_events"])
    dataset = MetamonDataset(
        dset_root=str(output / "trajectories"),
        formats=["gen9ou"],
        observation_space=DefaultObservationSpace(),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
    )
    assert len(dataset) == 8
    observations, actions, rewards, dones = dataset[0]
    assert len(observations["numbers"]) == len(actions["chosen"]) + 1
    assert rewards.shape == dones.shape == (len(actions["chosen"]),)
    assert dones[-1]
