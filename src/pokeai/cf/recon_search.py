"""BS0.3/0.4: anytime belief search over reconstructed worlds (oracle-free).

A decision = learner information payload (picklable) -> 4 persistent worker processes, each with
its own simulator driver, belief sampler and learner/opponent players. Work items are
(world, seed) pairs enumerated seed-major (round-robin over worlds) and dealt to the workers
round-robin; one pair = every candidate rolled out from the same reconstructed world with the
same seeds (paired). A pair is complete only if all its candidates finished before the deadline.
"""

from __future__ import annotations

import contextlib
import gc
import multiprocessing as mp
import multiprocessing.connection as mp_connection
import pickle
import random
import time
import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from pokeai.cf.restore_check import TRAP_PREFIXES

N_WORKERS = 4
SCHED_MARGIN_S = 0.3  # stop scheduling new rollouts this long before the hard deadline
OPP_MODELS = ("grunt", "gymleader", "mixture")
HEURISTIC = {"grunt": "Grunt", "gymleader": "GymLeader"}
MIN_PAIRS = 4
SNR_MIN = 1.0


@dataclass
class Payload:
    """Everything the learner knows at the root (no oracle fields)."""

    root_id: int
    opp_model: str  # SEARCH opponent model: grunt | gymleader | mixture (never the real one)
    seed: int
    info: Any  # LearnerInfo
    learner_snapshot: dict[str, Any]
    old_tag: str
    mask: list[int]
    candidates: list[int]  # candidate slots, C greedy first
    n_worlds: int = 8
    n_seeds: int = 2
    extra: dict[str, Any] = field(default_factory=dict)


def outcome(winner: str | None) -> float:
    return 0.5 if winner is None or winner == "tie" else 1.0 if winner == "p1" else 0.0


def pair_order(n_worlds: int, n_seeds: int) -> list[tuple[int, int]]:
    """Seed-major round-robin: all worlds under seed 0, then seed 1, ..."""
    return [(w, k) for k in range(n_seeds) for w in range(n_worlds)]


def assignment(n_worlds: int, n_seeds: int, workers: int = N_WORKERS) -> list[list[tuple[int, int]]]:
    pairs = pair_order(n_worlds, n_seeds)
    return [pairs[j::workers] for j in range(workers)]


def decide(
    candidates: list[int], mask: list[int], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fixed decision rule. `pairs`: complete pairs, each {"outcomes": [per candidate]}.

    Highest paired mean delta vs a_C (candidates[0]); override only if delta > 0 and
    delta / SE > SNR_MIN with at least MIN_PAIRS complete pairs, else the legal C action.
    """
    a_c = candidates[0]
    assert mask[a_c], "fallback action must be a legal C action"
    n = len(pairs)
    out: dict[str, Any] = {
        "n_pairs": n, "action": a_c, "override": False, "reason": "fallback:few_pairs",
        "stats": [],
    }
    if n < MIN_PAIRS or len(candidates) < 2:
        if len(candidates) < 2:
            out["reason"] = "fallback:single_candidate"
        return out
    table = np.array([p["outcomes"] for p in pairs], dtype=float)  # [pairs, candidates]
    for c in range(1, len(candidates)):
        delta = table[:, c] - table[:, 0]
        out["stats"].append({
            "slot": candidates[c], "delta": float(delta.mean()),
            "se": float(delta.std(ddof=1) / np.sqrt(n)),
        })
    best = max(out["stats"], key=lambda s: s["delta"])
    snr_ok = best["se"] == 0 or best["delta"] / best["se"] > SNR_MIN
    if best["delta"] > 0 and snr_ok:
        out.update(action=best["slot"], override=True, reason="override")
    else:
        out["reason"] = "fallback:no_positive_delta" if best["delta"] <= 0 else "fallback:snr"
    return out


def opponent_for_pair(payload: Payload, world: int, k: int) -> str:
    """Heuristic opponent policy of one (world, seed) pair. Mixture: a seeded fair coin per pair,
    the same for every candidate of the pair. Reads only the search parameter."""
    if payload.opp_model == "mixture":
        coin = random.Random(f"{payload.seed}:{payload.root_id}:mix:{world}:{k}").random()
        return "Grunt" if coin < 0.5 else "GymLeader"
    return HEURISTIC[payload.opp_model]


# ------------------------------------------------------------------------------- workers
def _worker_main(conn: Any) -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.restore_check import load_pool  # noqa: F401  (import warm-up)
    from pokeai.cf.rollouts import make_players
    from pokeai.cf.sim import Sim

    sim = Sim()
    sampler = BeliefSampler(sim)
    players = {o: make_players(o, greedy_prefix=True) for o in ("GymLeader", "Grunt")}
    conn.send(("ready", 0, None))
    gc.collect()
    gc.disable()  # cyclic-GC pauses (0.3-1 s on the 3 MB payload) would blow the budget
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            sim.close()
            return
        _, did, blob, pairs, sched_deadline, hard_deadline = msg
        try:
            snap = _decide_chunk(
                sim, sampler, players, blob, pairs, sched_deadline, hard_deadline,
                lambda new, snap, did=did: conn.send(("prog", did, new, snap)))
            conn.send(("done", did, snap))
        except Exception:  # noqa: BLE001
            conn.send(("error", did, traceback.format_exc()[-2000:]))
        gc.collect()  # between decisions, outside the timed window
        conn.send(("idle", did, None))


def _decide_chunk(
    sim: Any, sampler: Any, players: dict[str, Any], blob: bytes,
    pairs: list[tuple[int, int]], sched_deadline: float, hard_deadline: float, emit: Any,
) -> dict[str, Any]:
    from pokeai.cf.branching import rollout
    from pokeai.cf.cf1 import seed_triple
    from pokeai.cf.recon_match import build_world_root
    from pokeai.cf.recon_world import Unsupported, reconstruct_world

    t_start = time.time()
    payload: Payload = pickle.loads(blob)
    load_s = time.time() - t_start
    cache: dict[tuple[int, str], Any] = {}
    timing = {"load_s": load_s, "sample_s": 0.0, "reconstruct_s": 0.0, "pov_s": 0.0,
              "rollout_s": 0.0}
    roll_times: list[float] = []
    st: dict[str, Any] = {
        "timing": timing, "n_rollouts": 0, "worlds_built": 0, "unsupported": [], "errors": [],
        "illegal": 0, "retries": 0, "t_end": 0.0}
    keys = []

    def can_start() -> bool:
        now = time.time()
        est = float(np.mean(roll_times) * 1.25) if roll_times else 0.0
        return now < sched_deadline and now + est < hard_deadline

    try:
        for w, k in pairs:
            if not can_start():
                break
            opp = opponent_for_pair(payload, w, k)
            if (w, opp) not in cache:
                try:
                    recon = reconstruct_world(
                        sim, sampler, payload.info,
                        world_seed=payload.seed + payload.root_id * 100 + w,
                        battle_seed=payload.seed + payload.root_id)
                    timing["sample_s"] += recon.timing["sample_s"]
                    timing["reconstruct_s"] += recon.timing["reconstruct_s"]
                    t = time.time()
                    cp = build_world_root(
                        sim, players[opp], payload.learner_snapshot, payload.old_tag,
                        payload.info, recon)
                    timing["pov_s"] += time.time() - t
                    keys.append(cp.sim_key)
                    cache[(w, opp)] = SimpleNamespace(checkpoint=cp)
                    st["worlds_built"] += 1
                except Unsupported as e:
                    cache[(w, opp)] = None
                    st["unsupported"].append(str(e))
                emit([], st)
            if cache[(w, opp)] is None:
                continue
            outs = []
            complete = True
            for slot in payload.candidates:
                if not can_start():
                    complete = False
                    break
                t = time.time()
                r = rollout(sim, players[opp], cache[(w, opp)], "p1", slot,
                            seed_triple(payload.root_id, k), keep_signature=False)
                roll_times.append(time.time() - t)
                timing["rollout_s"] += roll_times[-1]
                st["n_rollouts"] += 1
                outs.append(outcome(r.winner))
                st["illegal"] += r.illegal_actions
                for e in r.errors:
                    if e.startswith(("protocol-retry:", *TRAP_PREFIXES)):
                        st["retries"] += 1
                    else:
                        st["errors"].append(e)
            if not complete:
                break
            emit([{"world": w, "seed": k, "outcomes": outs, "opp": opp}], st)
    finally:
        for key in keys:
            sim.drop_snapshot(key)
    st["t_end"] = time.time()
    return st


class SearchPool:
    """Four persistent search workers (one simulator driver each) with a hard deadline.

    Every message carries the decision id; replies for any other id are discarded, so a late
    worker can never leak into a later decision. A worker still busy at the next request keeps
    its queued work; `settle()` (between decisions, untimed) waits for idle and recycles a
    worker that does not come back.
    """

    def __init__(
        self, workers: int = N_WORKERS, worker_target: Any = None
    ) -> None:
        self.ctx = mp.get_context("spawn")
        self.target = worker_target or _worker_main
        self.conns: list[Any] = []
        self.procs: list[Any] = []
        self.outstanding: list[int] = []
        self.did = 0
        for _ in range(workers):
            self._spawn()
        for conn in self.conns:
            assert conn.recv()[0] == "ready"

    def _spawn(self, index: int | None = None) -> None:
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=self.target, args=(child,), daemon=True)
        proc.start()
        if index is None:
            self.conns.append(parent)
            self.procs.append(proc)
            self.outstanding.append(0)
        else:
            self.conns[index], self.procs[index], self.outstanding[index] = parent, proc, 0

    def _poll(self, timeout: float, sink: Any = None) -> bool:
        """Receive what is available; route by decision id. True if anything was received."""
        got = False
        for conn in mp_connection.wait(self.conns, timeout=max(timeout, 0.0)):
            i = self.conns.index(conn)
            while conn.poll():
                kind, did, body = self._recv3(conn)
                got = True
                if kind == "idle":
                    self.outstanding[i] = max(self.outstanding[i] - 1, 0)
                elif sink is not None and did == self.did:
                    sink(i, kind, body)
        return got

    @staticmethod
    def _recv3(conn: Any) -> tuple[str, int, Any]:
        msg = conn.recv()
        if msg[0] == "prog":
            return "prog", msg[1], (msg[2], msg[3])
        return msg[0], msg[1], msg[2]

    def settle(self, timeout: float = 10.0) -> int:
        """Untimed, between decisions: wait until every worker is idle; recycle stragglers.
        Returns the number of recycled workers."""
        end = time.time() + timeout
        while any(self.outstanding) and time.time() < end:
            self._poll(end - time.time())
        recycled = 0
        for i, n in enumerate(self.outstanding):
            if n:
                self.procs[i].kill()
                self.procs[i].join(timeout=5)
                self._spawn(i)
                assert self.conns[i].recv()[0] == "ready"
                recycled += 1
        return recycled

    def decision(
        self, payload: Payload, budget_s: float, t_request: float | None = None,
        sched_margin_s: float = SCHED_MARGIN_S,
    ) -> dict[str, Any]:
        """One decision under a hard deadline. The clock starts at `t_request` (the arrival of
        the decision request; default: now) and covers pickling, reconstruction, world sampling,
        rollouts and finalisation. At t_request + budget the decision is returned with the
        complete pairs received so far, without waiting for in-flight workers."""
        t0 = time.time() if t_request is None else t_request
        hard, sched = t0 + budget_s, t0 + budget_s - sched_margin_s
        self._poll(0.0)  # absorb idle/stale messages that are already waiting
        self.did += 1
        did = self.did
        blob = pickle.dumps(payload)
        chunks = assignment(payload.n_worlds, payload.n_seeds, len(self.conns))
        for i, (conn, pairs) in enumerate(zip(self.conns, chunks, strict=True)):
            conn.send(("decide", did, blob, pairs, sched, hard))
            self.outstanding[i] += 1
        t_sent = time.time()
        n = len(self.conns)
        pairs_by: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        snaps: list[dict[str, Any] | None] = [None] * n
        done = [False] * n
        error: list[str] = []

        def sink(i: int, kind: str, body: Any) -> None:
            if kind == "prog":
                new, snap = body
                pairs_by[i].extend(new)
                snaps[i] = snap
            elif kind == "done":
                snaps[i] = body
                done[i] = True
            elif kind == "error":
                error.append(body)
                done[i] = True

        while not all(done) and time.time() < hard:
            self._poll(hard - time.time(), sink)
        if error:
            raise RuntimeError(error[0])
        pairs = [p for chunk in pairs_by for p in chunk]
        choice = decide(payload.candidates, payload.mask, pairs)
        wall = time.time() - t0
        have = [s for s in snaps if s is not None]
        keys = ("load_s", "sample_s", "reconstruct_s", "pov_s", "rollout_s")
        return {
            "wall_s": wall, "dispatch_s": t_sent - t0, "choice": choice,
            "n_rollouts": sum(s["n_rollouts"] for s in have),
            "worlds_built": sum(s["worlds_built"] for s in have),
            "timing": {k: [s["timing"][k] for s in have] for k in keys},
            "unsupported": [u for s in have for u in s["unsupported"]],
            "errors": [e for s in have for e in s["errors"]],
            "illegal": sum(s["illegal"] for s in have),
            "retries": sum(s["retries"] for s in have),
            "pairs": pairs, "workers_done": sum(done), "workers": n,
            "hard_deadline_hit": not all(done),
        }

    def close(self) -> None:
        for conn in self.conns:
            with contextlib.suppress(Exception):
                conn.send(("stop",))
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
