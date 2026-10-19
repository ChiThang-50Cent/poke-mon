"""BS1.1: live search integration in NeuralPlayer (no-op when off, one C forward per decision,
frame snapshots, retry restoration with search on) and the non-blocking worker lifecycle."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_bs1_deadline import fake_worker, make_payload
from test_neural_player import encoded, policy

from pokeai.agents.live_search import LiveSearch, SearchSettings, hidden_digest
from pokeai.cf.recon_search import SearchPool


def fake_battle(tag="a"):
    return SimpleNamespace(
        battle_tag=tag, reviving=False, trapped=False, force_switch=False, turn=3,
        available_moves=[SimpleNamespace(id="tackle")],
        _last_request={"active": [{}], "rqid": 1, "side": {"pokemon": []}},
        active_pokemon=SimpleNamespace(species="corviknight", ability="pressure"),
        opponent_active_pokemon=SimpleNamespace(species="magnezone", ability=None))


@pytest.fixture
def patched(monkeypatch):
    from metamon.interface import UniversalAction, UniversalState

    monkeypatch.setattr(UniversalState, "from_Battle", lambda _: object())
    monkeypatch.setattr(
        UniversalAction, "definitely_valid_actions",
        lambda *_: {UniversalAction(0), UniversalAction(1), UniversalAction(2)})
    monkeypatch.setattr(
        UniversalAction, "to_BattleOrder",
        lambda action, _: SimpleNamespace(message=f"/choose {action.action_idx}"))

    async def accept(*_a, **_k):
        return None

    monkeypatch.setattr(
        "metamon.env.metamon_player.MetamonPlayer._handle_battle_message", accept)
    return monkeypatch


def searching_player(search):
    player = policy()
    player.encoder = SimpleNamespace(encode=lambda _: encoded(), unknown_id=0)
    player.search = search
    player._frame_snap, player._frame_start, player._frame_decided = {}, {}, {}
    player._t_request, player.search_rows, player._search_open = {}, {}, {}
    player._last_probs = None
    player._battles = {"a": object(), "b": object()}
    player.forward_count = 0
    return player


class StubSearch:
    def __init__(self, override=None):
        self.override, self.calls, self.player = override, [], None

    def run(self, player, battle, tag, c_action, mask, probs, t_request, snapshot,
            pending_start, decision_no):
        before = hidden_digest(player.hidden.get(tag))
        self.calls.append({
            "tag": tag, "c_action": c_action, "probs": probs, "snapshot": snapshot,
            "pending_start": pending_start, "decision_no": decision_no, "hidden": before})
        action = c_action if self.override is None else self.override
        return action, {"category": "stub", "action": action, "c_action": c_action}


def test_search_off_is_a_strict_noop(patched):
    player = policy()
    player.encoder = SimpleNamespace(encode=lambda _: encoded(), unknown_id=0)
    patched.setattr("pokeai.cf.offline.snapshot_player", lambda *_: pytest.fail("snapshot"))
    assert player.search is None
    keys = set(vars(player))
    asyncio.run(player._handle_battle_message([["a"], ["", "turn", "2"]]))
    order = player.choose_move(fake_battle())
    assert order.message.startswith("/choose")
    # nothing but the pre-existing bookkeeping (+ the always-present forward counter)
    assert set(vars(player)) - keys <= {"forward_count", "last_descriptor"}
    assert not hasattr(player, "search_rows")


def test_search_layer_without_override_leaves_c_untouched(patched):
    base = policy()
    base.encoder = SimpleNamespace(encode=lambda _: encoded(), unknown_id=0)
    stub = StubSearch()
    live = searching_player(stub)
    live.model = base.model  # identical weights
    for tag in ("a",):
        a = base.choose_move(fake_battle(tag))
        b = live.choose_move(fake_battle(tag))
        assert a.message == b.message
    assert torch.equal(base.hidden["a"], live.hidden["a"])
    assert base.latencies_ms.__len__() == live.latencies_ms.__len__() == 1
    assert live.forward_count == 1 and live.search_rows["a"][0]["c_forwards"] == 1
    assert stub.calls[0]["hidden"] == hidden_digest(live.hidden["a"])  # search saw post-C state


def test_override_changes_action_but_not_memory_or_forward_count(patched):
    base = policy()
    base.encoder = SimpleNamespace(encode=lambda _: encoded(), unknown_id=0)
    c_order = base.choose_move(fake_battle())
    other = next(i for i in (0, 1, 2) if f"/choose {i}" != c_order.message)
    base.hidden.pop("a")
    stub = StubSearch(override=other)
    live = searching_player(stub)
    live.model = base.model
    base.choose_move(fake_battle())
    s_order = live.choose_move(fake_battle())
    assert s_order.message == f"/choose {other}" and c_order.message != s_order.message
    assert torch.equal(base.hidden["a"], live.hidden["a"])
    assert live.forward_count == 1  # exactly one C forward for the represented decision
    assert live._pending_choice["a"]["order"] == s_order.message  # evidence = the order sent


def test_collector_records_sent_action_when_search_overrides(patched):
    """W6.0 bug: the trajectory stored C's argmax instead of the overridden action sent."""
    from metamon.interface import UniversalState

    from pokeai.env.collector import TrajectoryCollector

    patched.setattr(UniversalState, "from_Battle", lambda _: SimpleNamespace(to_dict=dict))
    patched.setattr("pokeai.env.collector.order_to_action", lambda *_: (-1, "unsupported"))
    base = policy()
    base.encoder = SimpleNamespace(encode=lambda _: encoded(), unknown_id=0)
    c_action = int(base.choose_move(fake_battle()).message.removeprefix("/choose "))
    other = next(i for i in (0, 1, 2) if i != c_action)
    live = searching_player(StubSearch(override=other))
    live.model = base.model
    live.collector = TrajectoryCollector()
    order = live.choose_move(fake_battle())
    record = live.collector.behavior[0]
    assert order.message == f"/choose {other}" and live.collector.actions == [other]
    assert record["chosen_index"] == c_action and record["executed_index"] == other
    assert int(np.argmax(record["probabilities"])) == c_action  # C's distribution kept
    assert live.search_rows["a"][record["search_row"]]["action"] == other


def test_frame_snapshot_lifecycle_and_battle_isolation(patched):
    taken = []

    def fake_snapshot(player, tag):
        taken.append((tag, len(player.history.get(tag, []))))
        return {"snap": len(taken)}

    patched.setattr("pokeai.cf.offline.snapshot_player", fake_snapshot)
    live = searching_player(StubSearch())
    msg = asyncio.run(live._handle_battle_message([[">a"], ["", "move", "p1a: X", "Tackle"]]))
    assert msg is None
    # same undecided frame: second update batch and an error batch do not retake
    asyncio.run(live._handle_battle_message([[">a"], ["", "-damage", "p2a: Y", "50/100"]]))
    asyncio.run(live._handle_battle_message([[">a"], ["", "error", "[Invalid choice] x"]]))
    assert taken == [("a", 0)]
    assert live._frame_start["a"] == 0 and live.history["a"][0].startswith("|move")
    # request arrival clock is set by the request batch only
    assert "a" not in live._t_request
    asyncio.run(live._handle_battle_message([[">a"], ["", "request", "{}"]]))
    assert "a" in live._t_request
    live.choose_move(fake_battle("a"))
    assert live.search.calls[0]["snapshot"] == {"snap": 1}
    assert live.search.calls[0]["pending_start"] == 0
    # next frame retakes (decided), battle b is independent
    asyncio.run(live._handle_battle_message([[">a"], ["", "turn", "4"]]))
    asyncio.run(live._handle_battle_message([[">b"], ["", "turn", "1"]]))
    assert taken == [("a", 0), ("a", 3), ("b", 0)]
    assert live._frame_start == {"a": 3, "b": 0}


def test_retry_with_search_restores_memory_and_keeps_the_frame_snapshot(patched):
    patched.setattr("pokeai.cf.offline.snapshot_player", lambda p, t: {"s": 1})
    live = searching_player(StubSearch(override=1))
    asyncio.run(live._handle_battle_message([[">a"], ["", "turn", "2"]]))
    before = live.hidden.get("a")
    live.choose_move(fake_battle("a"))
    assert not torch.equal(live.hidden["a"], before) if before is not None else True
    snap = live._frame_snap["a"]
    live._model_choice["a"] = True
    live._pending_choice["a"] = {
        "order": "/choose switch 2", "trapped": False, "maybeTrapped": True,
        "active_species": "corviknight", "active_ability": "pressure",
        "opponent_species": "magnezone", "opponent_ability": None}
    error = "[Unavailable choice] Can't switch: The active Pokémon is trapped"
    asyncio.run(live._handle_battle_message([[">a"], ["", "error", error]]))
    assert live.illegal_actions == 0
    if before is None:
        assert "a" not in live.hidden
    else:
        assert torch.equal(live.hidden["a"], before)
    assert live._frame_snap["a"] is snap  # the retry decision reuses the same root
    live.choose_move(fake_battle("a"))  # re-decision: exactly one more forward
    assert live.forward_count == 2
    assert [r["c_forwards"] for r in live.search_rows["a"]] == [1, 1]
    assert live.search.calls[1]["snapshot"] is snap


# ------------------------------------------------------------------ LiveSearch layer
class StubPool:
    def __init__(self, usable=True, result=None, raises=None):
        self._usable, self.result, self.raises = usable, result, raises
        self.restarts, self.maintained, self.decisions = 0, 0, []

    def usable(self):
        return self._usable

    def maintain(self):
        self.maintained += 1

    def decision(self, payload, budget, t_request=None):
        self.decisions.append((payload, budget, t_request))
        if self.raises:
            raise self.raises
        return self.result


def decision_result(override_slot=None, n_pairs=8, unsupported=()):
    action = 1 if override_slot is None else override_slot
    return {
        "choice": {"n_pairs": n_pairs, "action": action, "override": override_slot is not None,
                   "reason": "override" if override_slot is not None else "fallback:snr",
                   "stats": []},
        "n_rollouts": 10, "worlds_built": 4, "unsupported": list(unsupported), "illegal": 0,
        "retries": 0, "rounds": 0, "turns": 0, "world_rejected_mask": 0,
        "mask_rejects": [], "dropped_pairs": 0, "dropped": [], "errors": [], "hard_deadline_hit": False, "workers_done": 4,
        "wall_s": 1.0, "timing": {}, "pairs": [{"opp": "Grunt"}]}


def run_layer(pool, t_request_ago=0.0, snapshot=None, mask=(0, 1, 1, 1, 0, 0, 0, 0, 0),
              role="p1", history=None):
    snapshot = {"snap": 1} if snapshot is None else snapshot
    layer = LiveSearch(SearchSettings(), pool=pool)
    history = history or ["|a", "|b", "|c", "|d"]
    player = SimpleNamespace(
        history={"t": history}, hidden={}, _team=SimpleNamespace(yield_team=lambda: "PACKED"))
    battle = SimpleNamespace(
        _last_request={"rqid": 7, "side": {"id": role, "pokemon": [{"ident": f"{role}: X"}]}},
        player_role=role)
    probs = [0.0, 0.5, 0.3, 0.2, 0, 0, 0, 0, 0]
    now = time.time()
    action, row = layer.run(player, battle, "t", 1, np.array(mask, bool), probs,
                            now - t_request_ago, snapshot, 2, 5)
    return layer, action, row, now


def test_search_budget_is_d_minus_elapsed_minus_margin():
    pool = StubPool(result=decision_result(override_slot=2))
    layer, action, row, _now = run_layer(pool, t_request_ago=0.8)
    payload, budget, t_req = pool.decisions[0]
    assert 3.85 < budget < 3.95 and row["category"] == "search_override" and action == 2
    assert payload.opp_model == "mixture" and not hasattr(payload, "opponent")
    assert payload.info.own_team == "PACKED" and payload.info.request["rqid"] == 7 and not payload.info.swapped
    assert payload.info.lines == ["|a", "|b", "|c", "|d"]
    assert payload.info.pending_p1_lines == ["|c", "|d"]
    assert payload.candidates[0] == 1 and len(payload.candidates) >= 2
    assert payload.learner_snapshot == {"snap": 1} and payload.old_tag == "t"
    assert abs(t_req - row["t_search_start"]) < 1e-6  # hard deadline = t_request + D - M
    assert layer.counts == {"search_override": 1}


def test_skip_when_budget_below_half_second_uses_c():
    pool = StubPool(result=decision_result())
    _, action, row, _ = run_layer(pool, t_request_ago=4.3)  # 5.0 - 4.3 - 0.3 = 0.4 < 0.5
    assert row["category"] == "search_skipped_budget" and action == 1 and not pool.decisions


def test_workers_not_ready_uses_c_immediately_without_waiting():
    pool = StubPool(usable=False, result=decision_result())
    t = time.time()
    _, action, row, _ = run_layer(pool)
    assert time.time() - t < 0.2 and action == 1
    assert row["category"] == "search_skipped_workers" and pool.maintained == 1
    assert not pool.decisions


def test_search_exception_keeps_c_action_and_is_categorised():
    pool = StubPool(raises=RuntimeError("boom"))  # unknown class (BS1.2b: new class)
    _, action, row, _ = run_layer(pool)
    assert action == 1 and row["category"] == "search_error_new_class" and "boom" in row["error"]


def test_candidate_legal_live_but_outside_reconstructed_mask_falls_back_to_c():
    # BS1.1b saw 2 decisions where a world's mask excluded a candidate that is legal live:
    # the pool re-raises the worker error; the live choice must stay C's legal action.
    cause = "forced action 2 outside mask"
    pool = StubPool(raises=RuntimeError(cause))
    mask = (0, 1, 1, 1, 0, 0, 0, 0, 0)
    layer, action, row, _ = run_layer(pool, mask=mask)
    assert action == 1 and mask[action] == 1
    assert row["category"] == "search_error" and row["error_cause"] == f"RuntimeError: {cause}"
    assert layer.counts == {"search_error": 1}


def test_missing_snapshot_and_unsupported_and_no_signal_categories():
    pool = StubPool(result=decision_result())
    assert run_layer(pool, snapshot=False)[2]["category"] in (
        "search_skipped_no_snapshot", "search_fallback_no_signal")
    _, _, row, _ = run_layer(StubPool(result=decision_result(n_pairs=0, unsupported=["pivot"])))
    assert row["category"] == "search_fallback_unsupported"
    _, _, row, _ = run_layer(StubPool(result=decision_result()))
    assert row["category"] == "search_fallback_no_signal"
    _, action, row, _ = run_layer(StubPool(result=decision_result()), mask=(0, 1, 0, 0))
    assert row["category"] == "search_fallback_no_signal" and action == 1  # single candidate


# ------------------------------------------------------------------ worker lifecycle
def test_pool_not_ready_then_ready_and_async_restart_does_not_block():
    pool = SearchPool(workers=2, worker_target=fake_worker, wait=False)
    try:
        assert not pool.usable()  # still starting: callers use C immediately
        pool.wait_ready()
        assert pool.usable()
        pool.decision(make_payload(slow=True), 0.6)  # straggler stays inside decision 1
        t = time.time()
        started = pool.maintain(stuck_after_s=0.0)
        assert time.time() - t < 0.3  # respawn is asynchronous
        assert started >= 1 and pool.restarts == started
        assert not pool.usable()  # the respawned worker is not ready yet
        pool.wait_ready()
        assert pool.usable()
        d = pool.decision(make_payload(), 1.0)
        assert d["wall_s"] <= 1.05 and {p["tag"] for p in d["pairs"]} == {pool.did}
    finally:
        pool.close()


def test_swap_roles_exchanges_seats_in_lines_and_requests():
    from pokeai.cf.offline import swap_roles

    assert swap_roles(">battle-gen9ou-1\n|switch|p1a: Gholdengo|Gholdengo, L100|100/100") == (
        ">battle-gen9ou-1\n|switch|p2a: Gholdengo|Gholdengo, L100|100/100")
    assert swap_roles("|-damage|p2a: X|50/100|[from] item: Life Orb|[of] p1a: Y") == (
        "|-damage|p1a: X|50/100|[from] item: Life Orb|[of] p2a: Y")
    assert swap_roles("|player|p1|w1abc|102|") == "|player|p2|w1abc|102|"
    assert swap_roles("|-sidestart|p2: Name|move: Stealth Rock") == (
        "|-sidestart|p1: Name|move: Stealth Rock")
    assert swap_roles('|request|{"side":{"name":"wp1","id":"p2","pokemon":[{"ident":"p2: A"}]}}'
                      ) == '|request|{"side":{"name":"wp1","id":"p1","pokemon":[{"ident":"p1: A"}]}}'
    line = "|move|p1a: A|Tackle|p2a: B"
    assert swap_roles(swap_roles(line)) == line


def test_p2_seated_learner_is_presented_to_the_search_in_the_canonical_seat():
    pool = StubPool(result=decision_result())
    history = ["|switch|p2a: A|A, L100|100/100", "|move|p1a: B|Tackle|p2a: A", "|turn|3"]
    run_layer(pool, role="p2", history=history)
    info = pool.decisions[0][0].info
    assert info.swapped
    assert info.lines == ["|switch|p1a: A|A, L100|100/100", "|move|p2a: B|Tackle|p1a: A",
                          "|turn|3"]
    assert info.pending_p1_lines == info.lines[2:]
    assert info.request["side"]["id"] == "p1"
    assert info.request["side"]["pokemon"][0]["ident"] == "p1: X"
