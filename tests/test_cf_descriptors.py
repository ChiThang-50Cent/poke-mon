"""CF1.5c-A: ActionDescriptorEncoder is public-only, world-invariant, and one code path."""

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

import pokeai  # noqa: F401  (metamon.baselines import order)
from pokeai.cf import cf1
from pokeai.cf.belief import BeliefSampler
from pokeai.cf.branching import candidate_actions
from pokeai.cf.descriptors import DIM, FEATURE_NAMES, describe_battle
from pokeai.cf.offline import SIDES, OfflineMatch, PlayerSpec, make_player, new_tag
from pokeai.cf.restore_check import (
    C_POLICY,
    PREVIEW,
    CheckpointRule,
    load_pool,
    play_with_checkpoints,
)
from pokeai.cf.sim import Sim

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


def _decision(sim, players, root, world=None):
    """Descriptor/mask of the learner's root decision plus the live battle object."""
    probe = OfflineMatch.restore(sim, players, root.checkpoint, new_tag(), record_probs=True)
    players["p1"].describe = True
    try:
        if world is not None:
            probe.apply_world(world)
        battle = next(iter(players["p1"].battles.values()))
        probe.step()
        model = next(d["model"] for d in probe.rounds[0].decisions["p1"] if d["model"])
        mask = np.array(model["mask"], dtype=bool)
        # same battle object, same request: the direct entry point must agree
        direct = describe_battle(battle, mask)
        return model["descriptor"], direct, mask, battle
    finally:
        players["p1"].describe = False
        probe.close()


@pytest.fixture(scope="module")
def root_ctx(sim):
    pool = load_pool(TEAMS)
    return cf1.build_root(sim, BeliefSampler(sim), pool, 20261010, 1000, n_worlds=2)


@needs_assets
def test_descriptor_is_bit_identical_across_sampled_worlds(sim, root_ctx):
    players, root = root_ctx.picked["players"], root_ctx.picked["root"]
    base = root_ctx.info["descriptor"]
    assert base.x.shape == (13, DIM) and np.isfinite(base.x).all()
    assert len(FEATURE_NAMES) == DIM
    assert base.x[np.array(root_ctx.info["mask"], dtype=bool)].any()
    for world in root_ctx.worlds:
        got, _, mask, _ = _decision(sim, players, root, world)
        assert mask.tolist() == [bool(m) for m in root_ctx.info["mask"]]
        assert np.array_equal(got.x, base.x)
        assert got.ids == base.ids and got.reasons == base.reasons


@needs_assets
def test_offline_hook_equals_direct_live_entry_point(sim, root_ctx):
    players, root = root_ctx.picked["players"], root_ctx.picked["root"]
    hooked, direct, _, _ = _decision(sim, players, root)
    assert np.array_equal(hooked.x, direct.x)
    assert hooked.ids == direct.ids and hooked.reasons == direct.reasons
    # candidate_actions returns that very descriptor for the root
    assert np.array_equal(root_ctx.info["descriptor"].x, hooked.x)
    # unmasked slots stay all-zero
    illegal = [i for i, m in enumerate(root_ctx.info["mask"]) if not m]
    assert not hooked.x[illegal].any()


@needs_assets
def test_descriptor_ignores_hidden_opponent_fields(sim, root_ctx):
    players, root = root_ctx.picked["players"], root_ctx.picked["root"]
    hooked, _, mask, battle = _decision(sim, players, root)
    mutated = copy.deepcopy(battle)
    opp = mutated.opponent_active_pokemon
    assert opp is not None
    revealed = opp.is_terastallized
    # scramble everything the opponent has not revealed
    opp._item = "choicescarf" if opp.item != "choicescarf" else "leftovers"
    opp._ability = "protean"
    opp._possible_abilities = ["protean"]
    for mon in mutated.opponent_team.values():
        if not mon.is_terastallized:
            mon._terastallized_type = None if mon.tera_type is not None else mon.type_1
        mon._stats = {k: 1 for k in mon.stats}
    assert opp.is_terastallized == revealed
    again = describe_battle(mutated, mask)
    assert np.array_equal(again.x, hooked.x)
    assert again.ids == hooked.ids and again.reasons == hooked.reasons


@needs_assets
def test_descriptors_cover_all_slot_types_without_missing_features(sim):
    pool = load_pool(TEAMS)
    names = sorted(pool)
    players = {s: make_player(PlayerSpec(C_POLICY, sample=True, preview=PREVIEW), s) for s in SIDES}
    rule = CheckpointRule(seed=5, cap_remaining=[8], battle_index=1)
    _match, taken = play_with_checkpoints(
        sim, players, new_tag(), {"p1": pool[names[3]][0], "p2": pool[names[4]][0]},
        sim_seed=11, side_seeds={"p1": 21, "p2": 22}, rule=rule,
    )
    seen = set()
    for root in taken:
        info = candidate_actions(sim, players, root, "p1")
        if info is None:
            continue
        d = info["descriptor"]
        assert np.isfinite(d.x).all()
        assert not d.reasons, d.reasons
        seen |= {("move", "switch", "tera")[0 if s < 4 else 1 if s < 9 else 2]
                 for s, m in enumerate(info["mask"]) if m}
    assert {"move", "switch"} <= seen
