"""BS0.3 (profile) and BS0.4 (benefit): anytime belief search from reconstructed worlds.

Phase A (untimed, 4 owner processes): draw fresh roots; build the learner-information payload;
for S4 also precompute the ORACLE table (true-state rollouts of every candidate on 16 fresh
seeds; evaluation only, never visible to the search).
Phase B (timed, 4 search workers): one decision at a time per budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import pickle
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bs0_1_reconstruct as s1

S3_SEED = 20261300
S4_SEED = 20261400
N_EVAL = 16


def make_payload(args: tuple[int, str, int, bool, int]) -> dict[str, Any]:
    root_id, opponent, seed, oracle, n_seeds = args
    from pokeai.cf.branching import candidate_actions, rollout
    from pokeai.cf.cf1 import choose_candidates, seed_triple
    from pokeai.cf.recon_search import Payload
    from pokeai.cf.rollouts import phase_bucket

    sim = s1._S["sim"]
    out: dict[str, Any] = {"root": root_id, "opponent": opponent}
    try:
        picked = s1.draw_root(sim, root_id, seed, opponent)
        info, _, _ = s1.learner_info(sim, picked)
        players, root = picked["players"], picked["root"]
        cinfo = candidate_actions(sim, players, root, "p1")
        if cinfo is None or cinfo["mask"] != picked["mask_pass1"]:
            raise RuntimeError("root decision not reproducible")
        cands = [c["slot"] for c in choose_candidates(cinfo, random.Random(seed * 7 + root_id))]
        payload = Payload(
            root_id=root_id, opponent=opponent, seed=seed, info=info,
            learner_snapshot=root.checkpoint.players["p1"], old_tag=root.checkpoint.tag,
            mask=cinfo["mask"], candidates=cands, n_worlds=8, n_seeds=n_seeds)
        out.update(
            blob=pickle.dumps(payload), candidates=cands, mask=cinfo["mask"],
            turn=picked["turn"], phase=phase_bucket(picked["turn"]),
            pair="|".join(picked["names"]), forced_switch=picked["forced_switch"])
        if oracle:  # true-state evaluation (oracle, evaluation only)
            table = []
            for slot in cands:
                row = []
                for k in range(N_EVAL):
                    r = rollout(sim, players, root, "p1", slot,
                                seed_triple(root_id + 10_000, 100 + k), keep_signature=False)
                    row.append(0.5 if r.winner in (None, "tie") else 1.0 if r.winner == "p1" else 0.0)
                table.append(row)
            out["oracle"] = table
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
        out["trace"] = traceback.format_exc()[-1500:]
    return out


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def run_decisions(pool: Any, payloads: list[dict[str, Any]], budget: float) -> list[dict[str, Any]]:
    from pokeai.cf.recon_search import Payload

    rows = []
    for item in payloads:
        payload: Payload = pickle.loads(item["blob"])
        d = pool.decision(payload, budget)
        choice = d["choice"]
        rows.append({
            "root": item["root"], "opponent": item["opponent"], "phase": item["phase"],
            "turn": item["turn"], "n_cands": len(item["candidates"]),
            "candidates": item["candidates"], "budget": budget,
            "wall_s": d["wall_s"], "dispatch_s": d["dispatch_s"],
            "n_pairs": choice["n_pairs"], "n_rollouts": d["n_rollouts"],
            "action": choice["action"], "override": choice["override"],
            "reason": choice["reason"], "stats": choice["stats"],
            "legal": bool(item["mask"][choice["action"]]),
            "a_c": item["candidates"][0],
            "worlds_built": d["worlds_built"],
            "timing": d["timing"], "unsupported": d["unsupported"], "errors": d["errors"],
            "illegal": d["illegal"], "retries": d["retries"],
            "pairs": d["pairs"],
        })
        print(item["root"], budget, rows[-1]["n_pairs"], rows[-1]["reason"],
              round(rows[-1]["wall_s"], 2), flush=True)
        del payload
        gc.collect()  # the main process runs with the cyclic GC off during timed decisions
    return rows


def gen_payloads(roots: int, seed: int, oracle: bool, n_seeds: int, workers: int) -> list[dict[str, Any]]:
    jobs = [(i, ["GymLeader", "Grunt"][i % 2], seed, oracle, n_seeds) for i in range(roots)]
    out = []
    with mp.get_context("spawn").Pool(workers, initializer=s1.init) as owners:
        for r in owners.imap_unordered(make_payload, jobs):
            out.append(r)
            print("payload", r["root"], r.get("phase"), r.get("error", ""), flush=True)
    out.sort(key=lambda r: r["root"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[3, 4], required=True)
    ap.add_argument("--roots", type=int, default=100)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 2.0, 5.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--independent-only", action="store_true",
                    help="stage 4: add the secondary independent belief-world evaluation to "
                    "existing decisions_*.json")
    args = ap.parse_args()
    seed = args.seed or (S3_SEED if args.stage == 3 else S4_SEED)
    out = Path(args.out or f"runs/w5-bs0-{args.stage}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cache = out / "payloads.pkl"
    if cache.exists():
        payloads = pickle.loads(cache.read_bytes())
    else:
        payloads = gen_payloads(args.roots, seed, args.stage == 4, args.seeds, 4)
        cache.write_bytes(pickle.dumps(payloads))
    payload_wall = time.perf_counter() - started
    good = [p for p in payloads if "error" not in p]
    print("payload errors:", len(payloads) - len(good), flush=True)

    from pokeai.cf.recon_search import SearchPool

    pool = SearchPool()
    try:

        warm = pickle.loads(good[0]["blob"])
        pool.decision(warm, 4.0)  # warm-up: imports, model, JIT
        gc.collect()
        gc.disable()
        results: dict[str, list[dict[str, Any]]] = {}
        if args.independent_only:
            for budget in args.budgets:
                results[str(budget)] = json.loads((out / f"decisions_{budget}.json").read_text())
            evaluate_independent(pool, good, results)
            for budget, rows in results.items():
                (out / f"decisions_{budget}.json").write_text(json.dumps(rows))
            return
        for budget in args.budgets:
            rows = run_decisions(pool, good, budget)
            results[str(budget)] = rows
            (out / f"decisions_{budget}.json").write_text(json.dumps(rows))
        if args.stage == 4:
            evaluate_independent(pool, good, results)
            for budget, rows in results.items():
                (out / f"decisions_{budget}.json").write_text(json.dumps(rows))
    finally:
        pool.close()
    (out / "meta.json").write_text(json.dumps({
        "payload_errors": [{k: p[k] for k in ("root", "error")} for p in payloads if "error" in p],
        "payload_wall_s": payload_wall, "wall_s": time.perf_counter() - started,
        "seed": seed, "n_seeds": args.seeds,
    }))
    (out / "payload_meta.json").write_text(json.dumps([
        {k: v for k, v in p.items() if k not in ("blob",)} for p in payloads]))


def evaluate_independent(pool: Any, good: list[dict[str, Any]], results: dict[str, Any]) -> None:
    """Secondary: independent belief-world rollouts (fresh worlds and seeds) of the override
    roots, a_search vs a_C paired."""
    from pokeai.cf.recon_search import Payload

    by_root = {p["root"]: p for p in good}
    for rows in results.values():
        for row in rows:
            if not row["override"]:
                continue
            payload: Payload = pickle.loads(by_root[row["root"]]["blob"])
            payload.candidates = [row["a_c"], row["action"]]
            payload.seed = payload.seed + 777_000
            payload.n_worlds, payload.n_seeds = 8, 4
            d = pool.decision(payload, 120.0)
            table = np.array([p["outcomes"] for p in d["pairs"]], dtype=float)
            row["independent_delta"] = (
                float((table[:, 1] - table[:, 0]).mean()) if len(table) else None)
            row["independent_pairs"] = len(table)


if __name__ == "__main__":
    main()
