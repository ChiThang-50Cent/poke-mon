"""BS0.3/0.4: anytime belief search over reconstructed worlds (oracle-free).

A decision = learner information payload (picklable) -> 4 persistent worker processes, each with
its own simulator driver, belief sampler and learner/opponent players. Work items are
(world, seed) pairs enumerated seed-major (round-robin over worlds) and dealt to the workers
round-robin; one pair = every candidate rolled out from the same reconstructed world with the
same seeds (paired). A pair is complete only if all its candidates finished before the deadline.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import pickle
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_WORKERS = 4
MIN_PAIRS = 4
SNR_MIN = 1.0


@dataclass
class Payload:
    """Everything the learner knows at the root (no oracle fields)."""

    root_id: int
    opponent: str
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
    conn.send(("ready", None))
    gc.collect()
    gc.disable()  # cyclic-GC pauses (0.3-1 s on the 3 MB payload) would blow the budget
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            sim.close()
            return
        try:
            conn.send(("ok", _decide_chunk(sim, sampler, players, *msg[1:])))
        except Exception:  # noqa: BLE001
            conn.send(("error", traceback.format_exc()[-2000:]))
        gc.collect()  # between decisions, outside the timed window (the pool waits for "idle")
        conn.send(("idle", None))


def _decide_chunk(
    sim: Any, sampler: Any, players: dict[str, Any], blob: bytes,
    pairs: list[tuple[int, int]], deadline: float,
) -> dict[str, Any]:
    from types import SimpleNamespace

    from pokeai.cf.branching import rollout
    from pokeai.cf.cf1 import seed_triple
    from pokeai.cf.recon_match import build_world_root
    from pokeai.cf.recon_world import Unsupported, reconstruct_world

    t_start = time.time()
    payload: Payload = pickle.loads(blob)
    load_s = time.time() - t_start
    pl = players[payload.opponent]
    cache: dict[int, Any] = {}
    done: list[dict[str, Any]] = []
    timing = {"load_s": load_s, "sample_s": 0.0, "reconstruct_s": 0.0, "pov_s": 0.0,
              "rollout_s": 0.0}
    roll_times: list[float] = []
    n_rollouts = 0
    unsupported: list[str] = []
    errors: list[str] = []
    illegal = 0
    retries = 0
    keys = []
    try:
        for w, k in pairs:
            if time.time() >= deadline:
                break
            if w not in cache:
                try:
                    recon = reconstruct_world(
                        sim, sampler, payload.info,
                        world_seed=payload.seed + payload.root_id * 100 + w,
                        battle_seed=payload.seed + payload.root_id)
                    timing["sample_s"] += recon.timing["sample_s"]
                    timing["reconstruct_s"] += recon.timing["reconstruct_s"]
                    t = time.time()
                    cp = build_world_root(
                        sim, pl, payload.learner_snapshot, payload.old_tag, payload.info, recon)
                    timing["pov_s"] += time.time() - t
                    keys.append(cp.sim_key)
                    cache[w] = SimpleNamespace(checkpoint=cp)
                except Unsupported as e:
                    cache[w] = None
                    unsupported.append(str(e))
            if cache[w] is None:
                continue
            outs = []
            complete = True
            for slot in payload.candidates:
                est = float(np.mean(roll_times) * 1.25) if roll_times else 0.0
                if time.time() + est >= deadline:
                    complete = False
                    break
                t = time.time()
                r = rollout(sim, pl, cache[w], "p1", slot, seed_triple(payload.root_id, k),
                            keep_signature=False)
                roll_times.append(time.time() - t)
                timing["rollout_s"] += roll_times[-1]
                n_rollouts += 1
                outs.append(outcome(r.winner))
                illegal += r.illegal_actions
                for e in r.errors:
                    if e.startswith("protocol-retry:"):
                        retries += 1
                    else:
                        errors.append(e)
            if not complete:
                break
            done.append({"world": w, "seed": k, "outcomes": outs})
    finally:
        for key in keys:
            sim.drop_snapshot(key)
    return {"pairs": done, "timing": timing, "n_rollouts": n_rollouts,
            "worlds_built": sum(v is not None for v in cache.values()),
            "unsupported": unsupported, "errors": errors, "illegal": illegal,
            "retries": retries, "t_end": time.time()}


class SearchPool:
    """Four persistent search workers (one simulator driver each)."""

    def __init__(self, workers: int = N_WORKERS) -> None:
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child,), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        for conn in self.conns:
            assert conn.recv()[0] == "ready"

    def decision(self, payload: Payload, budget_s: float) -> dict[str, Any]:
        """One timed decision (wall clock starts here)."""
        t0 = time.time()
        deadline = t0 + budget_s
        blob = pickle.dumps(payload)
        chunks = assignment(payload.n_worlds, payload.n_seeds, len(self.conns))
        for conn, pairs in zip(self.conns, chunks, strict=True):
            conn.send(("decide", blob, pairs, deadline))
        t_sent = time.time()
        replies = []
        for conn in self.conns:
            status, body = conn.recv()
            if status != "ok":
                raise RuntimeError(body)
            replies.append(body)
        pairs = [p for r in replies for p in r["pairs"]]
        choice = decide(payload.candidates, payload.mask, pairs)
        wall = time.time() - t0
        for conn in self.conns:  # workers collect garbage between decisions, untimed
            assert conn.recv()[0] == "idle"
        return {
            "wall_s": wall, "dispatch_s": t_sent - t0, "choice": choice,
            "n_rollouts": sum(r["n_rollouts"] for r in replies),
            "worlds_built": sum(r["worlds_built"] for r in replies),
            "timing": {k: [r["timing"][k] for r in replies] for k in replies[0]["timing"]},
            "unsupported": [u for r in replies for u in r["unsupported"]],
            "errors": [e for r in replies for e in r["errors"]],
            "illegal": sum(r["illegal"] for r in replies),
            "retries": sum(r["retries"] for r in replies),
            "pairs": pairs,
        }

    def close(self) -> None:
        for conn in self.conns:
            conn.send(("stop",))
        for proc in self.procs:
            proc.join(timeout=10)
