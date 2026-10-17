"""BS0: belief-world reconstruction is oracle-free and matches the true root."""

import ast
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

import pokeai  # noqa: F401  (metamon.baselines import order)
from pokeai.cf.reconstruct import Tracker, to_id

SRC = Path("src/pokeai/cf")
RECON_PATH = ["reconstruct.py", "recon_world.py", "recon_match.py"]
FORBIDDEN_ATTRS = {"public_log", "public_view", "p2_full", "toJSON", "apply_world", "roster",
                   "to_json"}


@pytest.mark.parametrize("name", RECON_PATH)
def test_reconstruction_path_is_oracle_free(name):
    tree = ast.parse((SRC / name).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "recon_oracle" not in (node.module or ""), name
            assert all(a.name != "recon_oracle" for a in node.names), name
        elif isinstance(node, ast.Import):
            assert all("recon_oracle" not in a.name for a in node.names), name
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_ATTRS, f"{name}: .{node.attr}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in FORBIDDEN_ATTRS or node.value == "roster", name


def test_oracle_module_not_imported_by_library():
    for path in SRC.glob("*.py"):
        if path.name == "recon_oracle.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(a.name for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            assert all("recon_oracle" not in n for n in names), path


def _feed(lines):
    tracker = Tracker()
    tracker.feed(lines)
    return tracker


HEAD = [
    "|poke|p1|Gholdengo|", "|poke|p2|Dragonite, M|",
    "|switch|p1a: Gholdengo|Gholdengo|300/300", "|switch|p2a: Dragonite|Dragonite, M|100/100",
    "|turn|1",
]


def test_tracker_hp_boost_status_and_turn():
    t = _feed([
        *HEAD, "|move|p1a: Gholdengo|Nasty Plot|p1a: Gholdengo",
        "|-boost|p1a: Gholdengo|spa|2", "|-damage|p2a: Dragonite|60/100", "|-status|p2a: Dragonite|par",
        "|upkeep", "|turn|2",
    ])
    s = t.s
    assert s.turn == 2
    mon = s.active("p2")
    assert (mon.hp, mon.status) == (60, "par")
    assert s.active("p1").boosts["spa"] == 2


def test_tracker_duration_clock():
    t = _feed([*HEAD, "|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Dragonite",
               "|upkeep", "|turn|2", "|upkeep", "|turn|3"])
    rec = t.s.weather
    assert rec["id"] == "sunnyday"
    # set during turn 1 before the residual: two residual phases have run by turn 3
    assert t.ticks_since(rec["turn"], rec["post"]) == 2


def test_tracker_never_reads_unrevealed_hidden_fields():
    t = _feed(HEAD)
    mon = t.s.active("p2")
    assert mon.pp_used == {} and mon.stint_moves == []


needs_assets = pytest.mark.skipif(
    shutil.which("node") is None or not Path("models/pokedecision-v1-value").is_dir(),
    reason="needs node and models/pokedecision-v1-value",
)


@pytest.mark.sim
@needs_assets
def test_reconstructed_world_matches_true_root():
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.offline import OfflineMatch, new_tag
    from pokeai.cf.recon_oracle import compare_views, diff, true_root
    from pokeai.cf.recon_world import (
        LearnerInfo,
        Unsupported,
        learner_probe,
        reconstruct_world,
    )
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.rollouts import pick_root
    from pokeai.cf.sim import Sim

    pool = load_pool("data/teams_w5/collect/gen9ou")
    names = sorted(pool)
    with Sim() as sim:
        rng = random.Random(7)
        n1, n2 = rng.choice(names), rng.choice(names)
        seeds = {"sim": 71, "p1": 72, "p2": 73}
        teams = {"p1": pool[n1][0], "p2": pool[n2][0]}
        picked = pick_root(sim, teams, "Grunt", seeds, rng, greedy_prefix=True)
        assert picked is not None
        cp = picked["root"].checkpoint
        players = picked["players"]
        match = OfflineMatch.restore(sim, players, cp, new_tag())
        try:
            oracle = true_root(sim, match)
            pending = [x for ev in match.pending if ev["t"] == "update" for x in ev["p1"]]
            history = list(players["p1"].history[match.tag])
            info = LearnerInfo(
                lines=history + pending, request=json.loads(json.dumps(oracle["request"])),
                own_team=teams["p1"], pending_p1_lines=pending)
        finally:
            match.close()
        sampler = BeliefSampler(sim)
        try:
            recon = reconstruct_world(sim, sampler, info, world_seed=5, battle_seed=6)
        except Unsupported as e:  # unsupported roots are legitimate; S1 measures them
            pytest.skip(str(e))
        assert diff(oracle["request"], recon.p1_request) == []
        assert compare_views(oracle["view"], sim.public_view(recon.tag)) == []
        model = learner_probe(sim, players, cp.players["p1"], cp.tag, info, recon)
        assert model["mask"] == picked["mask_pass1"]
        assert np.isfinite(model["value"]) if model["value"] is not None else True
        assert to_id("Iron Valiant") == "ironvaliant"
