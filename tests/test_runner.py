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


@pytest.fixture
def scheduled_runner(tmp_path, monkeypatch):
    """Exercise runner lanes without a server or policy inference."""
    from pokeai.env import runner, teams

    files = []
    for index in range(5):
        path = tmp_path / f"team-{index}.gen9ou_team"
        path.write_text(f"packed-{index}")
        files.append(str(path))
    samples = []

    def sample():
        samples.append(True)
        return "default-packed"

    teamset = SimpleNamespace(
        team_files=list(reversed(files)),
        parse_showdown_team=lambda text: text,
        join_team=lambda team: team,
        yield_team=sample,
        most_recent_team_file="default.gen9ou_team",
    )
    monkeypatch.setattr(teams, "load_teamset", lambda path: teamset)

    async def stop_listening():
        return None

    class Player:
        def __init__(self, **kwargs):
            self.battles = {}
            self.decisions = 0
            self.team_preview_records = {}
            self.ps_client = SimpleNamespace(stop_listening=stop_listening)

        def update_team(self, team):
            self.team = team.yield_team()

        def randomize(self):
            pass

        async def battle_against(self, opponent, n_battles):
            # Different team-dependent delays change lane completion order.
            await asyncio.sleep(0.001 if self.team.endswith("0") else 0)
            tag = f"fake-{id(self)}-{len(self.battles)}"
            self.battles[tag] = SimpleNamespace(
                battle_tag=tag, turn=1, finished=True, won=True, lost=False,
            )
            self.team_preview_records[tag] = {
                "source": "random_default", "chosen_lead": "pikachu",
                "top_probability": None, "order": "/team 123456",
            }
            self.decisions = 1

    monkeypatch.setattr(runner, "_counted_baseline", lambda name: Player)

    def run(**kwargs):
        summary = asyncio.run(runner.run_battles(RunConfig(
            battles=12, out=str(tmp_path / "runs"), **kwargs,
        )))
        records = [
            json.loads(line)
            for line in (tmp_path / "runs" / summary["run_id"] / "battles.jsonl")
            .read_text().splitlines()
        ]
        return summary, sorted(records, key=lambda record: record["matchup_index"])

    return run, samples, teamset


def test_seeded_schedule_is_paired_across_arms_and_concurrency(scheduled_runner):
    import random

    from pokeai.env.runner import _build_schedule

    run, samples, teamset = scheduled_runner
    state = random.getstate()
    _build_schedule(teamset, RunConfig(schedule_seed=72))
    assert random.getstate() == state
    summary, first = run(schedule_seed=72, concurrent=1, player="GymLeader")
    teamset.team_files.reverse()
    _, second = run(schedule_seed=72, concurrent=4, player="Grunt")
    _, different = run(schedule_seed=73, concurrent=2)
    fields = (
        "matchup_index", "player_team", "opponent_team",
        "player_team_sha256", "opponent_team_sha256",
    )

    def pairs(records):
        return [tuple(record[field] for field in fields) for record in records]

    assert pairs(first) == pairs(second)
    assert pairs(first) != pairs(different)
    assert [record["matchup_index"] for record in first] == list(range(12))
    assert samples == []
    assert summary["completed"] == 12
    assert summary["lead_source_counts"] == {"random_default": 12}
    assert all(record["team_preview"]["chosen_lead"] == "pikachu" for record in first)


def test_default_schedule_keeps_original_teamset_sampling(scheduled_runner):
    from pokeai.env.runner import _build_schedule

    run, samples, _ = scheduled_runner
    # No file enumeration or parsing is required on the default path.
    assert _build_schedule(object(), RunConfig()) is None
    summary, records = run(concurrent=3)
    assert len(samples) == 24
    assert summary["config"]["schedule_seed"] is None
    assert all(record["player_team"] == "default.gen9ou_team" for record in records)
    assert all(record["opponent_team"] == "default.gen9ou_team" for record in records)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_preview_checkpoint_loaded_once_and_shared(
    scheduled_runner, monkeypatch, cuda_available
):
    import torch
    from metamon.backend.team_preview.preview import TeamPreviewModel

    from pokeai.env import runner

    run, _, _ = scheduled_runner
    shared_model = object()
    loads = []
    player_models = []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return shared_model

    original_player_type = runner._player_type

    def player_type(name, config, team_preview_model):
        player_models.append(team_preview_model)
        return original_player_type("GymLeader", config, team_preview_model)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(TeamPreviewModel, "load_from_checkpoint", load)
    monkeypatch.setattr(runner, "_player_type", player_type)
    summary, _ = run(
        player="neural:left", opponent="neural:right", concurrent=4,
        team_preview_checkpoint="preview.pt", team_preview_argmax=True,
        device="cpu", schedule_seed=9,
    )
    assert summary["completed"] == 12
    assert loads == [("preview.pt", {
        "device": "cuda" if cuda_available else "cpu", "use_argmax": True,
    })]
    assert player_models == [shared_model, shared_model]


@pytest.mark.parametrize("error_kind", ["Unavailable", "Invalid"])
@pytest.mark.parametrize("collect", [False, True])
def test_hidden_trap_retry_keeps_protocol_evidence(
    monkeypatch, error_kind, collect
):
    import metamon.baselines

    from pokeai.env.runner import _counted_baseline

    tag = "battle-gen9ou-retry"
    battle = SimpleNamespace(
        battle_tag=tag, turn=4,
        active_pokemon=SimpleNamespace(species="corviknight", ability="pressure"),
        opponent_active_pokemon=SimpleNamespace(species="magnezone", ability=None),
    )

    class Baseline:
        def __init__(self):
            self.sent = []
            self.retry_handshakes = 0

        def choose_move(self, battle):
            return SimpleNamespace(message=battle._last_request["choice"])

        async def _handle_battle_message(self, messages):
            for message in messages[1:]:
                if message[1] == "request":
                    battle._last_request = json.loads(message[2])
                    self.sent.append(self.choose_move(battle).message)
                elif message[1] == "error":
                    # The vendor, not our wrapper, schedules the retry handshake.
                    self.retry_handshakes += 1

    monkeypatch.setattr(metamon.baselines, "get_baseline", lambda name: Baseline)
    player = _counted_baseline("unused")()
    if collect:
        player.collector = SimpleNamespace(
            events=[], record=lambda battle, order: None,
        )
    original = {
        "rqid": 9, "active": [{"maybeTrapped": True}],
        "choice": "/choose switch 2",
    }
    revised = {
        "rqid": 9, "active": [{"trapped": True}],
        "choice": "/choose move 1",
    }
    error = f"[{error_kind} choice] Can't switch: The active Pokémon is trapped"
    asyncio.run(player._handle_battle_message([
        [">" + tag],
        ["", "request", json.dumps(original)],
        ["", "error", error],
        ["", "request", json.dumps(revised)],
    ]))
    assert player.last_server_error is None
    assert player.retry_handshakes == 1
    assert player.sent == ["/choose switch 2", "/choose move 1"]
    assert player.protocol_retries[tag] == [{
        "order": "/choose switch 2",
        "trapped": False, "maybeTrapped": True,
        "rqid": 9, "turn": 4,
        "active_species": "corviknight", "active_ability": "pressure",
        "opponent_species": "magnezone", "opponent_ability": None,
        "error": "|error|" + error,
    }]
    # Duplicate or unrelated errors cannot reuse an accepted retry's evidence.
    asyncio.run(player._handle_battle_message([
        [">" + tag], ["", "error", error],
    ]))
    assert player.last_server_error == "|error|" + error
    assert len(player.protocol_retries[tag]) == 1


@pytest.mark.parametrize(
    "maybe_trapped,trapped,order,error",
    [
        (False, False, "/choose switch 2", "The active Pokémon is trapped"),
        (True, True, "/choose switch 2", "The active Pokémon is trapped"),
        (True, False, "/choose move 1", "The active Pokémon is trapped"),
        (True, False, "/choose switch 2", "You can't switch to a fainted Pokémon"),
    ],
)
def test_trap_retry_rejects_illegal_or_unrelated_choices(
    maybe_trapped, trapped, order, error
):
    from pokeai.env.runner import _is_expected_trapped_retry

    payload = "[Invalid choice] Can't switch: " + error
    assert not _is_expected_trapped_retry(payload, {
        "maybeTrapped": maybe_trapped, "trapped": trapped, "order": order,
    })
    assert not _is_expected_trapped_retry(payload, None)


def test_summary_counts_protocol_retries_without_discarding_failed_attempts():
    evidence = {
        "order": "/choose switch 2", "trapped": False, "maybeTrapped": True,
        "error": "|error|[Invalid choice] Can't switch: The active Pokémon is trapped",
    }
    records = [{
        "battle_id": f"battle-{index}", "matchup_index": index,
        "winner": "player" if index == 0 else None,
        "error": None if index == 0 else "RuntimeError: unrelated error",
        "decisions_player": 2,
        "protocol_retries": {
            "player": [evidence] if index == 0 else [],
            "opponent": [evidence] if index == 1 else [],
        },
    } for index in range(2)]
    summary = summarize(records, 1, {})
    assert summary["completed"] == 1
    assert summary["error_count"] == 1
    assert summary["protocol_retry_counts"] == {"player": 1, "opponent": 1}
    for role, index in (("player", 0), ("opponent", 1)):
        assert summary["protocol_retry_evidence"][role] == [{
            **evidence, "battle_id": f"battle-{index}", "matchup_index": index,
        }]


def test_runner_persists_retry_evidence_per_role(scheduled_runner, monkeypatch):
    from pokeai.env import runner

    run, _, _ = scheduled_runner
    base = runner._counted_baseline("fixture")
    evidence = {
        "order": "/choose switch 2", "trapped": False, "maybeTrapped": True,
        "active_species": "corviknight", "active_ability": "pressure",
        "opponent_species": "magnezone", "opponent_ability": None,
        "error": "|error|[Invalid choice] Can't switch: The active Pokémon is trapped",
    }

    class Player(base):
        async def battle_against(self, opponent, n_battles):
            await super().battle_against(opponent, n_battles)
            tag = next(reversed(self.battles))
            self.protocol_retries = {tag: [dict(evidence)]}
            opponent.protocol_retries = {tag: [dict(evidence), dict(evidence)]}

    monkeypatch.setattr(runner, "_counted_baseline", lambda name: Player)
    summary, records = run(concurrent=3)
    assert summary["completed"] == 12
    assert summary["error_count"] == 0
    assert summary["protocol_retry_counts"] == {"player": 12, "opponent": 24}
    assert len(summary["protocol_retry_evidence"]["player"]) == 12
    for record in records:
        assert record["protocol_retries"] == {
            "player": [evidence], "opponent": [evidence, evidence],
        }


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
