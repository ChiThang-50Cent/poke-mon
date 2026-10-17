"""BS1.0: hard deadline of the search pool (fake workers, no simulator) and opponent models."""

from __future__ import annotations

import dataclasses
import pickle
import time

import pytest

from pokeai.cf.recon_search import Payload, SearchPool, opponent_for_pair

B = 1.0
JITTER = 0.05


def fake_worker(conn):
    """Speaks the pool protocol. Pair (0, 0) takes 2.5 s when payload.extra['slow'] is set
    (or every pair when extra['all_slow']); all other pairs take 20 ms."""
    conn.send(("ready", 0, None))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        _, did, blob, pairs, _sched, _hard = msg
        payload = pickle.loads(blob)
        snap = {"timing": {k: 0.0 for k in ("load_s", "sample_s", "reconstruct_s", "pov_s",
                                            "rollout_s")},
                "n_rollouts": 0, "worlds_built": 0, "unsupported": [], "errors": [],
                "illegal": 0, "retries": 0, "t_end": 0.0}
        for w, k in pairs:
            slow = payload.extra.get("all_slow") or (
                payload.extra.get("slow") and (w, k) == (0, 0))
            time.sleep(2.5 if slow else 0.02)
            snap["n_rollouts"] += len(payload.candidates)
            conn.send(("prog", did, [{"world": w, "seed": k, "tag": did,
                                      "outcomes": [1.0] + [0.0] * (len(payload.candidates) - 1)}],
                       snap))
        conn.send(("done", did, snap))
        conn.send(("idle", did, None))


def make_payload(**extra) -> Payload:
    return Payload(root_id=1, opp_model="mixture", seed=7, info=None, learner_snapshot={},
                   old_tag="t", mask=[0, 1, 1, 0], candidates=[1, 2, 0 + 2], n_worlds=4,
                   n_seeds=4, extra=extra)


@pytest.fixture
def pool():
    p = SearchPool(workers=4, worker_target=fake_worker)
    yield p
    p.close()


def test_returns_by_deadline_and_ignores_late_results(pool):
    t = time.time()
    d = pool.decision(make_payload(slow=True), B)
    assert time.time() - t <= B + JITTER
    assert d["wall_s"] <= B + JITTER
    assert d["hard_deadline_hit"] and d["workers_done"] == 3
    assert (0, 0) not in {(p["world"], p["seed"]) for p in d["pairs"]}
    first_pairs = len(d["pairs"])
    assert first_pairs >= 4
    # The straggler is still inside decision 1; decision 2 must contain no decision-1 result.
    d2 = pool.decision(make_payload(), B)
    assert {p["tag"] for p in d2["pairs"]} == {2}
    assert d2["wall_s"] <= B + JITTER
    assert pool.settle(timeout=0.5) == 1  # straggler recycled, others idle
    d3 = pool.decision(make_payload(), B)
    assert not d3["hard_deadline_hit"] and d3["workers_done"] == 4
    assert {p["tag"] for p in d3["pairs"]} == {3} and len(d3["pairs"]) == 16


def test_fallback_is_legal_c_action_when_no_pair_completes(pool):
    d = pool.decision(make_payload(all_slow=True), 0.5)
    assert d["wall_s"] <= 0.5 + JITTER
    assert d["pairs"] == []
    assert d["choice"]["action"] == 1 and not d["choice"]["override"]
    assert d["choice"]["reason"] == "fallback:few_pairs"
    pool.settle(timeout=0.2)


def test_request_clock_includes_time_before_dispatch(pool):
    t0 = time.time() - 0.4  # the request arrived 0.4 s ago
    pool.decision(make_payload(), B, t_request=t0)
    assert time.time() - t0 <= B + JITTER


def test_mixture_draw_is_paired_seeded_and_fair():
    p = make_payload()
    draws = [opponent_for_pair(p, w, k) for w in range(8) for k in range(8)]
    assert draws == [opponent_for_pair(p, w, k) for w in range(8) for k in range(8)]
    assert 16 <= sum(d == "Grunt" for d in draws) <= 48
    p.candidates = [1]  # the draw does not depend on the candidates
    assert draws == [opponent_for_pair(p, w, k) for w in range(8) for k in range(8)]
    p.opp_model = "grunt"
    assert {opponent_for_pair(p, 0, 0), opponent_for_pair(p, 3, 3)} == {"Grunt"}
    p.opp_model = "gymleader"
    assert opponent_for_pair(p, 0, 0) == "GymLeader"


def test_payload_carries_no_real_opponent_identity():
    names = {f.name for f in dataclasses.fields(Payload)}
    assert "opponent" not in names and "opp_model" in names
