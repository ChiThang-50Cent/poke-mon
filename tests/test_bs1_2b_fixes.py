"""BS1.2b: world mask validation, paired pair dropping on parser failures, exception allowlist
and fail-fast abort."""

from __future__ import annotations

import asyncio
import pickle
import time
from types import SimpleNamespace

import pytest
from metamon.backend.replay_parser.exceptions import CalledForeignConsecutive
from test_bs1_1_live import StubPool, run_layer

from pokeai.agents.live_search import LiveSearch, SearchSettings, classify_error
from pokeai.cf import recon_search as rs
from pokeai.env.runner import _gather_lanes

LIVE_MASK = [0, 1, 1, 1, 0, 0, 0, 0, 0]


def chunk(monkeypatch, *, world_masks, rollout_fn, pairs):
    """Run `_decide_chunk` on stubbed world construction / rollouts."""
    monkeypatch.setattr(
        "pokeai.cf.recon_world.reconstruct_world",
        lambda *_a, world_seed, battle_seed: SimpleNamespace(
            timing={"sample_s": 0.0, "reconstruct_s": 0.0}, world=world_seed % 100))
    monkeypatch.setattr(
        "pokeai.cf.recon_match.build_world_root",
        lambda sim, player, snap, tag, info, recon: SimpleNamespace(sim_key=recon.world))
    monkeypatch.setattr(
        rs, "world_root_mask", lambda sim, players, cp: world_masks.get(cp.sim_key, LIVE_MASK))
    monkeypatch.setattr("pokeai.cf.branching.rollout", rollout_fn)
    payload = rs.Payload(
        root_id=0, opp_model="grunt", seed=0, info=None, learner_snapshot={}, old_tag="t",
        mask=LIVE_MASK, candidates=[1, 2, 3], n_worlds=3, n_seeds=2)
    emitted = []
    sim = SimpleNamespace(drop_snapshot=lambda key: None)
    now = time.time()
    st = rs._decide_chunk(
        sim, None, {"Grunt": None}, pickle.dumps(payload), pairs, now + 60, now + 90,
        lambda new, snap: emitted.extend(new))
    return st, emitted


def ok_rollout(calls):
    def rollout(sim, players, world, side, slot, seeds, keep_signature=False):
        calls.append((world.checkpoint.sim_key, slot))
        return SimpleNamespace(winner="p1", errors=[], illegal_actions=0, rounds=1, turns=1)
    return rollout


def test_world_with_a_live_legal_candidate_illegal_is_rejected_for_all_candidates(monkeypatch):
    calls = []
    masks = {1: [0, 1, 0, 1, 0, 0, 0, 0, 0]}  # world 1: slot 2 illegal, live-legal
    pairs = [(w, k) for k in range(2) for w in range(3)]
    st, emitted = chunk(monkeypatch, world_masks=masks, rollout_fn=ok_rollout(calls), pairs=pairs)
    assert st["world_rejected_mask"] == 1 and "world_rejected_mask" in st["unsupported"]
    assert st["mask_rejects"][0]["bad"] == [2] and st["mask_rejects"][0]["world"] == 1
    assert {w for w, _ in calls} == {0, 2}  # nothing rolled out from the rejected world
    assert all(len(p["outcomes"]) == 3 for p in emitted) and len(emitted) == 4  # complete pairs


def test_parser_failure_drops_the_whole_pair_for_every_candidate(monkeypatch):
    calls = []

    def rollout(sim, players, world, side, slot, seeds, keep_signature=False):
        if world.checkpoint.sim_key == 0 and slot == 2:  # fails on the second candidate
            raise CalledForeignConsecutive(["move", "p1a: Pawmot", "Fire Spin"])
        return ok_rollout(calls)(sim, players, world, side, slot, seeds)

    pairs = [(0, 0), (1, 0), (0, 1)]
    st, emitted = chunk(monkeypatch, world_masks={}, rollout_fn=rollout, pairs=pairs)
    assert st["dropped_pairs"] == 2 and all("CalledForeignConsecutive" in d for d in st["dropped"])
    assert [(p["world"], p["seed"]) for p in emitted] == [(1, 0)]  # pair-complete data only
    assert len(emitted[0]["outcomes"]) == 3


def test_other_rollout_exceptions_are_not_swallowed(monkeypatch):
    def rollout(*_a, **_k):
        raise ValueError("unknown")

    with pytest.raises(ValueError):
        chunk(monkeypatch, world_masks={}, rollout_fn=rollout, pairs=[(0, 0)])


# ------------------------------------------------------------------ allowlist and fail-fast
def test_allowlist_classifies_known_classes_only():
    assert classify_error("RuntimeError: forced action 3 outside mask\n") == (
        "forced_action_outside_mask")
    assert classify_error(
        "metamon.backend.replay_parser.exceptions.CalledForeignConsecutive: Exception on "
        "Replay Forward-Fill: x") == "parser_forward_exception"
    assert classify_error("KeyError: 'x'") is None


def test_known_error_falls_back_without_aborting():
    pool = StubPool(raises=RuntimeError("forced action 2 outside mask"))
    layer, action, row, _ = run_layer(pool)
    assert action == 1 and row["category"] == "search_error"
    assert row["error_class"] == "forced_action_outside_mask" and layer.abort_reason is None


def test_new_class_returns_c_logs_evidence_and_flags_abort(tmp_path):
    pool = StubPool(raises=KeyError("brand-new"))
    settings = SearchSettings(fail_dir=str(tmp_path))
    layer = LiveSearch(settings, pool=pool)
    import test_bs1_1_live as t

    t_layer = t.LiveSearch
    t.LiveSearch = lambda *_a, **_k: layer  # run_layer builds its own layer; inject ours
    try:
        _, action, row, _ = t.run_layer(pool)
    finally:
        t.LiveSearch = t_layer
    assert action == 1 and row["category"] == "search_error_new_class"
    assert layer.abort_reason and "brand-new" in layer.abort_reason
    files = sorted(p.name for p in tmp_path.iterdir())
    assert any(f.endswith(".txt") for f in files) and any(f.endswith(".payload.pkl") for f in files)
    assert "Traceback" in next(tmp_path.glob("*.txt")).read_text()


def test_run_aborts_at_once_on_new_class_flag():
    live = SimpleNamespace(abort_reason=None)

    async def lane():
        live.abort_reason = "new search exception class: KeyError"
        await asyncio.sleep(30)

    async def go():
        t = time.time()
        with pytest.raises(RuntimeError, match="ABORT"):
            await _gather_lanes([lane()], live)
        assert time.time() - t < 2

    asyncio.run(go())


def test_run_without_flag_completes_normally():
    async def lane():
        await asyncio.sleep(0)

    asyncio.run(_gather_lanes([lane()], SimpleNamespace(abort_reason=None)))


def test_rejected_worlds_persist_a_reproducible_payload_and_the_fallback_category(tmp_path):
    from test_bs1_1_live import decision_result

    result = decision_result(n_pairs=0, unsupported=["world_rejected_mask"])
    result["world_rejected_mask"] = 2
    result["mask_rejects"] = [{"world": 1, "bad": [3], "live_mask": LIVE_MASK, "world_mask": None}]
    layer = LiveSearch(SearchSettings(fail_dir=str(tmp_path)), pool=StubPool(result=result))
    import test_bs1_1_live as t

    t_layer = t.LiveSearch
    t.LiveSearch = lambda *_a, **_k: layer
    try:
        _, action, row, _ = t.run_layer(layer._pool)
    finally:
        t.LiveSearch = t_layer
    assert action == 1 and row["category"] == "search_fallback_unsupported"
    assert row["world_rejected_mask"] == 2 and row["mask_rejects"][0]["bad"] == [3]
    kept = list((tmp_path / "forensics").glob("*.payload.pkl"))
    assert len(kept) == 1 and pickle.loads(kept[0].read_bytes()).candidates == row["candidates"]


# ------------------------------------------------------------------ reconstruction gap (b)
def test_choice_lock_starts_after_an_item_is_received_mid_stint():
    """BS1.2b root cause of the live-mask mismatches: a Choice Scarf received by Trick must lock
    the FIRST move executed after the acquisition (Brave Bird), not the first move of the stint
    (Roost, 0 PP => the world's learner was reduced to Struggle)."""
    from pokeai.cf.reconstruct import build_public

    lines = [
        "|player|p1|A|1|", "|player|p2|B|2|", "|teamsize|p1|6", "|teamsize|p2|6",
        "|poke|p1|Corviknight, F|", "|poke|p2|Gholdengo|",
        "|switch|p1a: Corviknight|Corviknight, F|400/400",
        "|switch|p2a: Gholdengo|Gholdengo|100/100", "|turn|1",
        "|move|p1a: Corviknight|Roost|p1a: Corviknight", "|turn|2",
        "|move|p2a: Gholdengo|Trick|p1a: Corviknight",
        "|-activate|p2a: Gholdengo|move: Trick|[of] p1a: Corviknight",
        "|-item|p1a: Corviknight|Choice Scarf|[from] move: Trick",
        "|move|p1a: Corviknight|Brave Bird|p2a: Gholdengo", "|turn|3",
    ]
    _, tracker = build_public(lines)
    mon = tracker.s.mons["p1"][0]
    assert mon.stint_moves == ["bravebird"] and mon.pp_used == {"roost": 1, "bravebird": 1}
    # a non-acquisition `-item` (Frisk reveal) leaves the stint untouched
    _, tracker = build_public(lines[:-3] + [
        "|-item|p1a: Corviknight|Leftovers|[from] ability: Frisk|[of] p2a: Gholdengo|[identify]",
        "|move|p1a: Corviknight|Brave Bird|p2a: Gholdengo", "|turn|3"])
    assert tracker.s.mons["p1"][0].stint_moves == ["roost", "bravebird"]
