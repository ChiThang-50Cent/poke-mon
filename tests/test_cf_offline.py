"""CF0: Node sim sidecar, offline harness framing/privacy, restore and branching."""

import re
import shutil
from pathlib import Path

import pytest
import torch

import pokeai  # noqa: F401  (metamon.baselines import order)
from pokeai.cf.branching import candidate_actions, rollout, root_consistency
from pokeai.cf.offline import (
    SIDES,
    OfflineMatch,
    PlayerSpec,
    make_player,
    new_tag,
    normalise_frame,
    normalise_line,
)
from pokeai.cf.restore_check import (
    C_POLICY,
    PREVIEW,
    CheckpointRule,
    load_pool,
    negative_control,
    play_with_checkpoints,
    verify_checkpoints,
)
from pokeai.cf.sim import Sim, gen5_seed

pytestmark = pytest.mark.sim

TEAMS = "data/teams_w5/collect/gen9ou"
needs_assets = pytest.mark.skipif(
    shutil.which("node") is None or not Path(C_POLICY.split(":")[1]).is_dir(),
    reason="needs node and models/pokedecision-v1-value",
)


@pytest.fixture(scope="module")
def sim():
    with Sim() as s:
        yield s


@pytest.fixture(scope="module")
def teams():
    pool = load_pool(TEAMS)
    names = sorted(pool)
    return {"p1": pool[names[0]][0], "p2": pool[names[1]][0]}


def norm_events(events):
    out = []
    for ev in events:
        ev = dict(ev)
        for key in ("omni", "p1", "p2"):
            if key in ev:
                ev[key] = [normalise_line(x) for x in ev[key]]
        if ev["t"] == "end":
            ev["data"] = {k: v for k, v in ev["data"].items() if k != "seed"}
        out.append(ev)
    return out


def start(sim, teams, tag, seed=7):
    return sim.create(
        tag, gen5_seed(seed), {"name": "A", "team": teams["p1"]},
        {"name": "B", "team": teams["p2"]},
    )


def play_default(sim, tag, turns):
    events = []
    for _ in range(turns):
        for side in SIDES:
            events += sim.choose(tag, side, "default")["events"]
    return events


def test_driver_snapshot_restore_roundtrip(sim, teams):
    tag = new_tag()
    start(sim, teams, tag)
    play_default(sim, tag, 1)  # team preview
    key = sim.snapshot(tag)
    before = sim.state(tag)
    original = norm_events(play_default(sim, tag, 3))
    after = sim.state(tag)
    runs = []
    for _ in range(2):  # restoring twice: the stored snapshot is never mutated
        copy_tag = new_tag()
        sim.restore(key, copy_tag)
        assert sim.state(copy_tag) == before
        runs.append((norm_events(play_default(sim, copy_tag, 3)), sim.state(copy_tag)))
    assert runs[0] == runs[1]
    assert runs[0][0] == original
    assert runs[0][1] == after


def test_set_prng_does_not_write_public_log_and_changes_outcome(sim, teams):
    tag = new_tag()
    start(sim, teams, tag)
    play_default(sim, tag, 1)
    key = sim.snapshot(tag)
    outcomes, seeded = set(), set()
    for seed in range(6):
        copy_tag = new_tag()
        sim.restore(key, copy_tag)
        sim.set_prng(copy_tag, gen5_seed(seed))
        seeded.add(sim.get_prng(copy_tag))
        events = play_default(sim, copy_tag, 2)
        lines = [x for ev in events if ev["t"] == "update" for x in ev["omni"]]
        assert not any("RNG was reset" in x for x in lines)
        outcomes.add(sim.get_prng(copy_tag))
    assert len(seeded) == 6  # distinct seeds map to distinct canonical PRNG states
    assert len(outcomes) > 1


HP = re.compile(r"\|(?:switch|drag|detailschange|-damage|-heal)\|(p[12])[ab]?: [^|]*\|[^|]*?(\d+)/(\d+)")


def test_channel_split_privacy(sim, teams):
    tag = new_tag()
    start(sim, teams, tag)
    events = play_default(sim, tag, 8)
    seen = {"p1": set(), "p2": set()}
    for ev in events:
        if ev["t"] == "sideupdate":
            assert f'"id":"{ev["side"]}"' in ev["data"] or "error" in ev["data"]
        if ev["t"] != "update":
            continue
        for side in SIDES:
            other = "p2" if side == "p1" else "p1"
            for line in ev[side]:
                assert not line.startswith("|split|")
                match = HP.search(line)
                if not match:
                    continue
                owner, _, maximum = match.groups()
                if owner == other:  # the opponent's HP is public: percent only
                    assert maximum == "100", line
                else:
                    seen[side].add(maximum)
        omni_hp = {HP.search(x).group(3) for x in ev["omni"] if HP.search(x)}
        assert omni_hp - {"100"} or not omni_hp
    # a player does see its own exact HP somewhere (the secret half of the split)
    assert any(m != "100" for side in SIDES for m in seen[side])


def test_normalisation_is_limited_to_tag_and_timestamp():
    frame = ">battle-gen9ou-12\n|t:|1792181244\n|move|p1a: X|Surf|p2a: Y"
    assert normalise_frame(frame, "battle-gen9ou-12") == (
        "><TAG>\n|t:|\n|move|p1a: X|Surf|p2a: Y"
    )
    assert normalise_line("|-damage|p2a: Y|41/100") == "|-damage|p2a: Y|41/100"


@pytest.fixture(scope="module")
def battle(sim):
    """One C(sampled) vs GymLeader battle with pre-registered checkpoints."""
    pool = load_pool(TEAMS)
    names = sorted(pool)
    spec = {
        "p1": PlayerSpec(C_POLICY, sample=True, preview=PREVIEW),
        "p2": PlayerSpec("GymLeader", sample=True),
    }
    players = {s: make_player(spec[s], s) for s in SIDES}
    rule = CheckpointRule(seed=5, cap_remaining=[8], battle_index=1)
    match, taken = play_with_checkpoints(
        sim, players, new_tag(), {"p1": pool[names[3]][0], "p2": pool[names[4]][0]},
        sim_seed=11, side_seeds={"p1": 21, "p2": 22}, rule=rule,
    )
    return players, match, taken


@needs_assets
def test_battle_completes_without_errors(battle):
    players, match, taken = battle
    assert match.winner() in SIDES
    assert not [e for r in match.rounds for e in r.errors]
    assert players["p1"].illegal_actions == 0
    assert len(taken) >= 3


@needs_assets
def test_restore_replay_identical_and_negative_control(sim, battle):
    players, match, taken = battle
    rows = verify_checkpoints(sim, players, match, taken)
    assert rows
    for row in rows:
        assert all(not m for m in row["mismatch_vs_original"]), row
        assert not row["mismatch_restore1_vs_restore2"]
        assert all(row["restore_immediate_state_equal"])
        assert row["script_mismatches"] == [0, 0]
    assert negative_control(sim, players, match, taken[-1], "prng")
    assert negative_control(sim, players, match, taken[-1], "hidden")


@needs_assets
def test_branch_determinism_and_divergence_after_root(sim, battle):
    players, _, taken = battle
    for root in taken:
        info = candidate_actions(sim, players, root, "p1")
        if info is not None:
            break
    else:
        pytest.skip("no viable root")
    seeds = {"sim": 101, "opponent": 102, "learner": 103}
    first, second, other = (
        rollout(sim, players, root, "p1", cand, seeds)
        for cand in (info["candidates"][0], info["candidates"][0], info["candidates"][1])
    )
    assert first.signature == second.signature
    assert (first.winner, first.final_prng) == (second.winner, second.final_prng)
    assert not first.errors and first.illegal_actions == 0
    consistency = root_consistency(first.signature, other.signature, "p1")
    assert consistency["root_inputs_identical"]
    assert consistency["opponent_root_decision_identical"]
    assert consistency["learner_root_observation_identical"]
    assert consistency["learner_root_choice_differs"]


@needs_assets
def test_opponent_sample_pairing_c_vs_c(sim):
    pool = load_pool(TEAMS)
    names = sorted(pool)
    spec = PlayerSpec(C_POLICY, sample=True, preview=PREVIEW)
    players = {s: make_player(spec, s) for s in SIDES}
    rule = CheckpointRule(seed=9, cap_remaining=[6], battle_index=0)
    _match, taken = play_with_checkpoints(
        sim, players, new_tag(), {"p1": pool[names[5]][0], "p2": pool[names[6]][0]},
        sim_seed=3, side_seeds={"p1": 4, "p2": 5}, rule=rule,
    )
    root = next(
        t for t in taken if candidate_actions(sim, players, t, "p1") is not None
    )
    info = candidate_actions(sim, players, root, "p1")
    seeds = {"sim": 1, "opponent": 2, "learner": 3}
    results = [
        rollout(sim, players, root, "p1", c, seeds, max_rounds=600)
        for c in info["candidates"]
    ]
    opponent_roots = {
        repr(r.root_decisions["p2"]) for r in results  # message + obs + hidden hashes
    }
    assert len(opponent_roots) == 1
    assert len(results) >= 2


def test_seeded_sampler_streams_are_reproducible():
    torch.manual_seed(0)
    probabilities = torch.full((13,), 1 / 13)
    draws = []
    for _ in range(2):
        generator = torch.Generator()
        generator.manual_seed(77)
        draws.append(
            [int(torch.multinomial(probabilities, 1, generator=generator)) for _ in range(20)]
        )
    assert draws[0] == draws[1]
    assert len(set(draws[0])) > 1


def test_offline_match_class_is_server_shaped():
    assert OfflineMatch.restore and new_tag().startswith("battle-gen9ou-")
    assert new_tag().split("-")[2].isdigit()
