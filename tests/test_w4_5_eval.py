"""Fast synthetic checks for frozen W4.5 design."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
spec = importlib.util.spec_from_file_location("w4_5", "scripts/evaluate_w4_5.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_verdict_includes_zero():
    assert m.verdict([0, .1]) == "inconclusive"
    assert m.verdict([-.1, 0]) == "inconclusive"
    assert m.verdict([-.1, .1]) == "inconclusive"
    assert m.verdict([.01, .1]) == "positive"


def test_cluster_bootstrap_keeps_directions_together():
    def rows(winners):
        return [{"matchup_index": i, "winner": w, "error": None,
                     "player_team": "xy"[i % 2], "opponent_team": "yx"[i % 2],
                     "player_team_sha256": "xy"[i % 2],
                     "opponent_team_sha256": "yx"[i % 2],
                     "player_challenges": i % 2 == 0} for i, w in enumerate(winners)]
    a = rows(["player", "opponent"]*3)
    b = rows(["opponent", "player"]*3)
    x = m.cluster_differences(a, b)
    assert np.array_equal(x, np.zeros(3))
    assert m.bootstrap(x)["bootstrap_95"] == [0, 0]
    b[1]["player_team_sha256"] = "bad"
    with pytest.raises(ValueError):
        m.cluster_differences(a, b)


def test_swap_schedule(tmp_path):
    from pokeai.env.runner import RunConfig, _build_schedule
    class Teams:
        def __init__(self):
            self.team_files = []
        @staticmethod
        def parse_showdown_team(text):
            return text
        @staticmethod
        def join_team(text):
            return text
    teams = Teams()
    for i in range(4):
        p = tmp_path / str(i)
        p.write_text(str(i))
        teams.team_files.append(str(p))
    cfg = RunConfig(battles=20, schedule_seed=9, swap_team_pairs=True)
    schedule = _build_schedule(teams, cfg)
    assert schedule == _build_schedule(teams, cfg)
    for i in range(0, 20, 2):
        assert schedule[i] == schedule[i+1][::-1]
    with pytest.raises(ValueError):
        RunConfig(battles=3, schedule_seed=9, swap_team_pairs=True)


def test_baseline_revival_uses_current_request_slots(monkeypatch):
    from types import SimpleNamespace

    import metamon.baselines

    from pokeai.env import runner

    class Baseline:
        def choose_move(self, battle):
            raise AssertionError("must not use cached Pokemon objects for revival")

    monkeypatch.setattr(metamon.baselines, "get_baseline", lambda _: Baseline)
    monkeypatch.setattr(runner, "_protocol_choice_evidence", lambda *args: {})
    player = runner._counted_baseline("synthetic")()
    battle = SimpleNamespace(
        battle_tag="revival", reviving=True,
        _last_request={"side": {"pokemon": [
            {"condition": "100/100"}, {"condition": "0 fnt"},
            {"condition": "50/100"},
        ]}},
    )
    assert player.choose_move(battle).message == "/choose switch 2"
    # A previously fainted slot has been revived. Never reuse stale status.
    battle._last_request["side"]["pokemon"][1]["condition"] = "50/100"
    battle._last_request["side"]["pokemon"][2]["condition"] = "0 fnt"
    assert player.choose_move(battle).message == "/choose switch 3"
