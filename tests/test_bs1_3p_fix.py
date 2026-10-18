"""BS1.3-P: reconstructed-root bootstrap (no start-of-battle side effects), world invariants and
rollout stall/horizon handling.

Incident: a sampled opponent lead holding Eject Pack was consumed by the learner lead's Intimidate
during the bootstrap `team 123456`, which left `midTurn=true`, no `residual` for the first rollout
turn and, when a mon fainted before moving, no forced-switch request: the harness spun for 600
rounds on a single turn (`RuntimeError: battle exceeded max_rounds`).
"""

from __future__ import annotations

import contextlib
import copy
import json
import pickle
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bs1_1_live import StubPool, decision_result
from test_bs1_2b_fixes import LIVE_MASK, chunk, ok_rollout

from pokeai.agents.live_search import LiveSearch, SearchSettings
from pokeai.cf import recon_search as rs
from pokeai.cf.offline import (
    STALL_K,
    OfflineMatch,
    RolloutAborted,
    RolloutHorizonExceeded,
    RolloutStalled,
    RoundRecord,
)
from pokeai.cf.recon_world import WorldRejectedState, world_invariant_violations

needs_assets = pytest.mark.skipif(
    shutil.which("node") is None or not Path("models/pokedecision-v1-value").is_dir(),
    reason="needs node and models/pokedecision-v1-value",
)


# ------------------------------------------------------------------ engine-level roots
@pytest.fixture(scope="module")
def world():
    """(sim, spec, request) of a real collect root, reconstructed through the public path."""
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.offline import OfflineMatch as Match
    from pokeai.cf.offline import new_tag
    from pokeai.cf.recon_oracle import true_root
    from pokeai.cf.recon_world import LearnerInfo, reconstruct_world
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.rollouts import pick_root
    from pokeai.cf.sim import Sim

    pool = load_pool("data/teams_w5/collect/gen9ou")
    names = sorted(pool)
    sim = Sim()
    try:
        rng = random.Random(7)
        n1, n2 = rng.choice(names), rng.choice(names)
        seeds = {"sim": 71, "p1": 72, "p2": 73}
        teams = {"p1": pool[n1][0], "p2": pool[n2][0]}
        picked = pick_root(sim, teams, "Grunt", seeds, rng, greedy_prefix=True)
        assert picked is not None
        match = Match.restore(sim, picked["players"], picked["root"].checkpoint, new_tag())
        try:
            oracle = true_root(sim, match)
            pending = [x for ev in match.pending if ev["t"] == "update" for x in ev["p1"]]
            history = list(picked["players"]["p1"].history[match.tag])
            info = LearnerInfo(
                lines=history + pending, request=json.loads(json.dumps(oracle["request"])),
                own_team=teams["p1"], pending_p1_lines=pending)
        finally:
            match.close()
        recon = reconstruct_world(sim, BeliefSampler(sim), info, world_seed=5, battle_seed=6)
        assert "forceSwitch" not in info.request and recon.spec["request_state"] == "move"
        yield sim, recon.spec, info.request
    finally:
        sim.close()


def edit_lead(spec, side, *, item=None, ability=None, moves=None, species=None, pct=None,
              hp=None, status=None):
    """Replace the active (slot 0) mon's set, in the engine team and in the patched state."""
    spec = copy.deepcopy(spec)
    if side == "p2":
        entry, mon = spec["p2"]["team"][0], spec["mons"]["p2"][0]
        if species:
            entry["species"] = entry["name"] = species
        for key, value in (("item", item), ("ability", ability), ("moves", moves)):
            if value is not None:
                entry[key] = value
        if ability is not None:
            mon["ability"] = ability
        if species:
            entry.update(nature="Brave", evs={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                         ivs={"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 0})
        if pct is not None:
            mon["pct"] = pct
    else:
        name = spec["p1"]["order"][0]
        parts = spec["p1"]["team"].split("]")
        for i, packed in enumerate(parts):
            fields = packed.split("|")
            if (fields[1] or fields[0]).lower().replace("-", "").startswith(
                    name.lower().replace("-", "")):
                for idx, value in ((2, item), (3, ability), (4, ",".join(moves) if moves else None)):
                    if value is not None:
                        fields[idx] = value
                parts[i] = "|".join(fields)
        spec["p1"]["team"] = "]".join(parts)
        mon = spec["mons"]["p1"][0]
        if ability is not None:
            mon["ability"] = ability
        if item is not None:
            mon["item_exact"] = item
        if hp is not None:
            mon["hp_exact"] = hp
    if status is not None:
        spec["mons"][side][0]["status"] = status
        spec["mons"][side][0]["status_state"] = {}
    return spec


def build(sim, spec, tag):
    with contextlib.suppress(Exception):  # first use of the tag: nothing to destroy
        sim.destroy(tag)
    sim.reconstruct(tag, spec)
    for si, side in enumerate(("p1", "p2")):  # the edits reached the engine's lead sets
        lead = sim.state(tag)["sides"][si]["pokemon"][0]
        want = spec["mons"][side][0].get("ability")
        assert want is None or lead["baseAbility"] == "".join(ch for ch in want.lower() if ch.isalnum())
    return sim.invariants(tag)


def commit(sim, tag, p1, p2=None):
    """Both sides choose; the omniscient protocol lines and the per-side requests that follow."""
    lines: list[str] = []
    requests: dict[str, dict] = {}
    for side, choice in (("p1", p1), ("p2", p2)):
        if choice is None:
            continue
        for ev in sim.choose(tag, side, choice)["events"]:
            if ev["t"] == "update":
                lines.extend(ev["omni"])
            elif ev["t"] == "sideupdate" and ev["data"].startswith("|request|"):
                requests[ev["side"]] = json.loads(ev["data"][len("|request|"):])
    return lines, requests


def replacement(sim, tag):
    """`switch N` for p1's first healthy benched mon."""
    mons = sim.state(tag)["sides"][0]["pokemon"]
    return next(f"switch {i + 1}" for i, m in enumerate(mons) if i and m["hp"] > 0)


INTIMIDATE_VS_EJECT_PACK = {"p1": {"ability": "intimidate", "item": "leftovers"},
                            "p2": {"item": "Eject Pack"}}


def eject_pack_world(spec, **p2_extra):
    spec = edit_lead(spec, "p1", **INTIMIDATE_VS_EJECT_PACK["p1"])
    return edit_lead(spec, "p2", **{**INTIMIDATE_VS_EJECT_PACK["p2"], **p2_extra})


@pytest.mark.sim
@needs_assets
def test_eject_pack_lead_is_not_consumed_by_the_bootstrap(world):
    sim, spec, request = world
    inv = build(sim, eject_pack_world(spec), "b13p-ep")
    state = sim.state("b13p-ep")
    lead = state["sides"][1]["pokemon"][0]
    assert lead["item"] == "ejectpack" and lead["itemState"].get("id") != ""
    assert lead["lastItem"] == "" and not lead["switchFlag"]
    assert world_invariant_violations(inv, request, spec) == []
    assert inv["midTurn"] is False and inv["queue"] == 0 and inv["requestState"] == "move"
    assert [s["request"] for s in inv["sides"]] == [["active"], ["active"]]


def state_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in state_diff(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in state_diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [(path, a, b)]


@pytest.mark.sim
@needs_assets
def test_eject_pack_to_leftovers_gives_the_same_root(world):
    sim, spec, _ = world
    build(sim, eject_pack_world(spec), "b13p-a")
    build(sim, eject_pack_world(spec, item="Leftovers"), "b13p-b")
    a, b = sim.state("b13p-a"), sim.state("b13p-b")
    for key in ("log", "inputLog", "messageLog", "sentLogPos", "sentEnd"):
        a.pop(key), b.pop(key)
    for state in (a, b):
        lead = state["sides"][1]["pokemon"][0]
        lead.pop("item"), lead.pop("itemState"), lead["set"].pop("item")
    assert state_diff(a, b) == []  # only the held item differs; no entry effect left anything


@pytest.mark.sim
@needs_assets
def test_first_turn_residual_runs_after_an_eject_pack_bootstrap(world):
    """BEFORE the fix the root kept midTurn=true: no residual (no poison damage, no upkeep) on
    the first rollout turn. AFTER: the full turn sequence."""
    sim, spec, _ = world
    spec = eject_pack_world(spec, moves=["Water Gun", "Taunt", "Focus Blast", "Light Screen"],
                            species="Shuckle", ability="Gluttony", pct=100)
    spec = edit_lead(spec, "p2", status="psn")
    spec = edit_lead(spec, "p1", moves=["Splash", "Splash", "Splash", "Splash"])
    build(sim, spec, "b13p-res")
    lines, requests = commit(sim, "b13p-res", "move 1", "move 2")
    assert any("[from] psn" in ln for ln in lines), lines
    assert "|upkeep" in lines and lines.index("|upkeep") > max(
        i for i, ln in enumerate(lines) if "[from] psn" in ln) - 3
    assert any(ln.startswith("|turn|") for ln in lines)
    assert {s: "active" in r for s, r in requests.items()} == {"p1": True, "p2": True}


@pytest.mark.sim
@needs_assets
def test_learner_ko_before_moving_gets_its_forced_switch(world):
    """The incident: the learner lead is KO'd before it moves in the first rollout turn."""
    sim, spec, _ = world
    spec = eject_pack_world(
        spec, species="Deoxys-Speed", ability="Pressure",
        moves=["Water Gun", "Taunt", "Focus Blast", "Light Screen"])
    spec["p2"]["team"][0].update(nature="Timid", evs={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 252},
                                 ivs={"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31})
    spec = edit_lead(spec, "p1", hp=1, moves=["Tackle", "Scratch", "Pound", "Cut"])
    build(sim, spec, "b13p-ko1")
    lines, requests = commit(sim, "b13p-ko1", "move 1", "move 1")
    assert "|faint|p1a" in " ".join(lines) or any(ln.startswith("|faint|p1a") for ln in lines)
    assert "forceSwitch" in requests["p1"] and "wait" in requests["p2"]
    assert not any(ln.startswith("|turn|") for ln in lines)  # the replacement comes first
    assert "|upkeep" in lines
    lines, requests = commit(sim, "b13p-ko1", replacement(sim, "b13p-ko1"))
    assert any(ln.startswith("|turn|") for ln in lines)
    assert "active" in requests["p1"] and "active" in requests["p2"]


@pytest.mark.sim
@needs_assets
def test_opponent_ko_before_moving_gets_its_forced_switch(world):
    sim, spec, _ = world
    spec = eject_pack_world(
        spec, species="Shuckle", ability="Gluttony", pct=1,
        moves=["Infestation", "Rollout", "Rest", "Toxic"])
    spec = edit_lead(spec, "p1", moves=["Water Gun", "Bubble", "Surf", "Scald"])
    build(sim, spec, "b13p-ko2")
    lines, requests = commit(sim, "b13p-ko2", "move 1", "move 1")
    assert any(ln.startswith("|faint|p2a") for ln in lines), lines
    assert "forceSwitch" in requests["p2"] and "wait" in requests["p1"]
    assert not any(ln.startswith("|turn|") for ln in lines) and "|upkeep" in lines


@pytest.mark.sim
@needs_assets
def test_valid_forced_switch_root_has_the_engine_boundary_state(world):
    """Forced-switch root: the residual already ran (midTurn true, empty queue); the replacement
    must be followed by the turn change only, never by a second residual."""
    sim, spec, _ = world
    spec = copy.deepcopy(spec)
    spec["request_state"], spec["switch_sides"] = "switch", ["p1"]
    spec["mons"]["p1"][0].update(hp_exact=0, fainted=True)
    spec = edit_lead(spec, "p2", item="Leftovers", status="psn")
    inv = build(sim, spec, "b13p-fs")
    forced_request = {"forceSwitch": [True]}
    assert world_invariant_violations(inv, forced_request, spec) == []
    assert inv["midTurn"] is True and inv["queue"] == 0 and inv["requestState"] == "switch"
    assert world_invariant_violations(inv, {"active": [{}]}, spec)  # a move request: rejected
    lines, requests = commit(sim, "b13p-fs", replacement(sim, "b13p-fs"))
    assert any(ln.startswith("|switch|p1a") for ln in lines)
    assert any(ln.startswith("|turn|") for ln in lines)
    assert "|upkeep" not in lines and not any("[from] psn" in ln for ln in lines)
    assert "active" in requests["p1"] and "active" in requests["p2"]


@pytest.mark.sim
@needs_assets
def test_leads_with_entry_effects_start_clean(world):
    """Weather/terrain setters, seeds, Booster Energy, Air Balloon and White Herb leads: nothing
    is consumed or left active by the bootstrap."""
    sim, spec, request = world
    cases = [
        {"ability": "Drought", "item": "Leftovers"},
        {"ability": "Grassy Surge", "item": "Grassy Seed"},
        {"ability": "Protosynthesis", "item": "Booster Energy"},
        {"ability": "Pressure", "item": "Air Balloon"},
        {"ability": "Pressure", "item": "White Herb"},
        {"ability": "Pressure", "item": "Adrenaline Orb"},
    ]
    for case in cases:
        edited = edit_lead(edit_lead(spec, "p1", ability="intimidate"), "p2", **case)
        inv = build(sim, edited, "b13p-fx")
        lead = sim.state("b13p-fx")["sides"][1]["pokemon"][0]
        assert world_invariant_violations(inv, request, edited) == [], case
        assert lead["item"] == case["item"].lower().replace(" ", ""), case
        assert not lead["boosts"]["def"] and not lead["boosts"]["atk"], case


# ------------------------------------------------------------------ Layer 2 (pure)
def invariants(**over):
    side = {"request": ["active"], "switchFlags": 0, "activeFainted": False,
            "faintedThisTurn": False, "pokemonLeft": 6, "aliveCount": 6, "choiceError": "",
            "active": [{"isActive": True}]}
    inv = {"ended": False, "midTurn": False, "queue": 0, "faintQueue": 0, "requestState": "move",
           "sides": [{"id": "p1", **side}, {"id": "p2", **side}]}
    inv.update(over)
    return inv


def test_normal_root_invariants():
    spec = {"switch_sides": []}
    assert world_invariant_violations(invariants(), {"active": [{}]}, spec) == []
    stale = world_invariant_violations(invariants(midTurn=True), {"active": [{}]}, spec)
    assert stale and "midTurn" in stale[0]  # the Eject Pack bootstrap state
    assert world_invariant_violations(invariants(queue=1), {"active": [{}]}, spec)
    assert world_invariant_violations(invariants(requestState="switch"), {"active": [{}]}, spec)
    bad = invariants()
    bad["sides"][1]["switchFlags"] = 1
    assert world_invariant_violations(bad, {"active": [{}]}, spec)


def test_forced_switch_root_invariants():
    spec = {"switch_sides": ["p1"]}
    inv = invariants(midTurn=True, requestState="switch")
    inv["sides"][0].update(request=["forceSwitch"], switchFlags=1, activeFainted=True,
                           faintedThisTurn=True, pokemonLeft=3, aliveCount=3,
                           active=[{"isActive": False}])
    inv["sides"][1]["request"] = ["wait"]
    forced = {"forceSwitch": [True]}
    assert world_invariant_violations(inv, forced, spec) == []
    assert world_invariant_violations(inv, {"active": [{}]}, spec)
    inv["midTurn"] = False  # the pre-fix forced root: a second residual would run
    assert world_invariant_violations(inv, forced, spec)


def test_world_rejected_state_is_counted_and_applies_to_every_candidate(monkeypatch):
    calls = []

    def reconstruct(*_a, world_seed, battle_seed):
        if world_seed % 100 == 1:
            raise WorldRejectedState("midTurn=True (expected False)")
        return SimpleNamespace(timing={"sample_s": 0.0, "reconstruct_s": 0.0}, world=world_seed % 100)

    monkeypatch.setattr("pokeai.cf.recon_world.reconstruct_world", reconstruct)
    monkeypatch.setattr(
        "pokeai.cf.recon_match.build_world_root",
        lambda sim, player, snap, tag, info, recon: SimpleNamespace(sim_key=recon.world))
    monkeypatch.setattr(rs, "world_root_mask", lambda sim, players, cp: LIVE_MASK)
    monkeypatch.setattr("pokeai.cf.branching.rollout", ok_rollout(calls))
    payload = rs.Payload(root_id=0, opp_model="grunt", seed=0, info=None, learner_snapshot={},
                         old_tag="t", mask=LIVE_MASK, candidates=[1, 2, 3], n_worlds=3, n_seeds=2)
    calls.clear()
    now = time.time()
    st = rs._decide_chunk(SimpleNamespace(drop_snapshot=lambda key: None), None, {"Grunt": None},
                          pickle.dumps(payload), [(w, k) for k in range(2) for w in range(3)],
                          now + 60, now + 90, lambda new, snap: None)
    assert st["world_rejected_state"] == 1 and "world_rejected_state" in st["unsupported"]
    assert st["state_rejects"][0]["world"] == 1 and "midTurn" in st["state_rejects"][0]["violations"]
    assert {w for w, _ in calls} == {0, 2}  # nothing rolled out from the rejected world


# ------------------------------------------------------------------ Layer 3 (synthetic harness)
def stub_match(script, max_round=10**6):
    """An OfflineMatch whose delivery rounds follow `script(round_index) -> dict`."""
    m = object.__new__(OfflineMatch)
    m.tag, m.rounds, m.round_index, m.ended, m.end_data, m.pending = "t", [], 0, False, None, []
    m.wait = {s: {"rqid": 1, "is_wait": False, "choice": ""} for s in ("p1", "p2")}
    state = {}

    def info(_tag):
        return {"turn": state["turn"], "requestState": state["rs"], "ended": m.ended,
                "requests": [{"rqid": state["rq"], "active": [{}]}, None]}

    m.sim = SimpleNamespace(info=info)

    def step():
        s = script(m.round_index)
        state.update(turn=s["turn"], rs=s.get("rs", "move"), rq=s.get("rq", 1))
        m.pending = [{"t": "sideupdate", "side": "p1", "data": d} for d in s.get("pending", [])]
        m.ended = s.get("ended", False)
        record = RoundRecord(
            index=m.round_index, delivered={"p1": [], "p2": []}, omni=[s.get("omni", [])],
            decisions={"p1": [], "p2": []}, submitted=s.get("submitted", []),
            errors=s.get("errors", []), events_out=[])
        m.rounds.append(record)
        m.round_index += 1
        return record

    m.step = step
    return m


def test_deadlock_is_a_stall_after_k_unchanged_rounds():
    """The incident: p1 owes a switch, nobody can answer: nothing pending, nothing submitted."""
    m = stub_match(lambda i: {"turn": 10, "rs": "switch", "rq": 6, "omni": ["|move|p2a: Deoxys"]})
    with pytest.raises(RolloutStalled) as exc:
        m.play(max_rounds=600)
    assert m.round_index == STALL_K  # proven after K rounds, not after 600
    rec = exc.value.record
    assert rec["turn"] == 10 and rec["request_state"] == "switch" and rec["unchanged_rounds"] == STALL_K
    assert rec["last_protocol_lines"] and rec["requests"][0] == (6, ("active",))
    assert isinstance(exc.value, RolloutAborted) and not isinstance(exc.value, RolloutHorizonExceeded)


def test_invalid_choice_loop_is_a_stall():
    loop = {"turn": 10, "rs": "move", "pending": ["|error|[Invalid choice] Can't move"],
            "submitted": [("p1", "/choose move 1")], "errors": ["|error|[Invalid choice]"]}
    m = stub_match(lambda i: loop)
    with pytest.raises(RolloutStalled) as exc:
        m.play(max_rounds=600)
    assert exc.value.record["last_errors"] and exc.value.record["pending"] == [("sideupdate", "p1")]


def test_k_is_locked_by_legitimate_repeats():
    """Up to K-1 identical rounds (a retried order) followed by progress is NOT a stall."""
    def script(i):
        turn = 1 + max(0, i - (STALL_K - 2))  # first K-1 rounds identical, then the turn advances
        return {"turn": min(turn, 8), "ended": i >= 12, "omni": [f"|r{i}"]}

    m = stub_match(script)
    m.play(max_rounds=600)
    assert m.ended and m.round_index > STALL_K


def test_genuine_long_rollout_is_horizon_censored_not_stalled():
    def script(i):
        return {"turn": 1 + i, "ended": i >= 699, "omni": [f"|move|{i}"]}

    m = stub_match(script)
    with pytest.raises(RolloutHorizonExceeded, match="battle exceeded max_rounds") as exc:
        m.play(max_rounds=600)
    assert m.round_index == 600 and exc.value.record["turn"] == 600
    assert not isinstance(exc.value, RolloutStalled)
    stub_match(script).play(max_rounds=1000)  # the same game finishes with a longer horizon


def test_stalled_rollout_voids_the_whole_decision(monkeypatch):
    calls = []

    def rollout(sim, players, root, side, slot, seeds, keep_signature=False):
        if root.checkpoint.sim_key == 1 and slot == 2:
            raise RolloutStalled("stalled", {"round_index": 4, "turn": 10, "request_state": "switch"})
        return ok_rollout(calls)(sim, players, root, side, slot, seeds)

    pairs = [(w, k) for k in range(2) for w in range(3)]
    st, _emitted = chunk(monkeypatch, world_masks={}, rollout_fn=rollout, pairs=pairs)
    assert [a["kind"] for a in st["rollout_aborts"]] == ["stalled"]
    abort = st["rollout_aborts"][0]
    assert (abort["world"], abort["seed"], abort["candidate"], abort["turn"]) == (1, 0, 2, 10)
    choice = rs.abort_choice([1, 2, 3], st["rollout_aborts"])
    assert choice == {"n_pairs": 0, "action": 1, "override": False,
                      "reason": "fallback:rollout_stalled", "stats": []}
    horizon = [{"kind": "horizon"}]
    assert rs.abort_choice([1, 2, 3], horizon)["reason"] == "fallback:rollout_horizon"
    assert rs.abort_choice([1, 2, 3], []) is None


def test_other_exceptions_still_abort_the_chunk(monkeypatch):
    def rollout(*_a, **_k):
        raise KeyError("brand-new")

    with pytest.raises(KeyError):
        chunk(monkeypatch, world_masks={}, rollout_fn=rollout, pairs=[(0, 0)])


@pytest.mark.parametrize("kind", ["stalled", "horizon"])
def test_aborted_decision_falls_back_to_c_with_forensics(tmp_path, kind):
    result = decision_result(n_pairs=0)
    result["choice"].update(reason=f"fallback:rollout_{kind}", action=1, override=False)
    result["rollout_aborts"] = [{"kind": kind, "world": 3, "seed": 1, "opp": "Grunt",
                                 "candidate": 0, "round_index": 600, "turn": 10,
                                 "request_state": "switch", "unchanged_rounds": 4, "last_errors": []}]
    layer = LiveSearch(SearchSettings(fail_dir=str(tmp_path)), pool=StubPool(result=result))
    import test_bs1_1_live as t

    t_layer = t.LiveSearch
    t.LiveSearch = lambda *_a, **_k: layer
    try:
        _, action, row, _ = t.run_layer(layer._pool)
    finally:
        t.LiveSearch = t_layer
    assert action == 1 and row["category"] == f"search_fallback_rollout_{kind}"
    assert layer.abort_reason is None  # not a new class: the run continues
    assert row["rollout_aborts"][0]["kind"] == kind and row["rollout_aborts"][0]["turn"] == 10
    kept = list((tmp_path / "forensics").glob("*.payload.pkl"))
    assert len(kept) == 1
    record = json.loads(next((tmp_path / "forensics").glob("*.json")).read_text())
    assert record["rollout_aborts"][0]["world"] == 3
