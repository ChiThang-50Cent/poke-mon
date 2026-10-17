"""BS1.0: hard-deadline latency (stage `deadline`) and the opponent-model matrix (stage `matrix`).

The search sees only the learner's information and a SEARCH opponent model (grunt / gymleader /
mixture). The real opponent identity is used here, in evaluation code, only to pick the true
policy for the oracle rollouts and to label rows; it never enters the Payload.
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

sys.path.insert(0, str(Path(__file__).parent))
import bs0_1_reconstruct as s1

DEADLINE_SEED = 20261700
MATRIX_SEED = 20261800
BUDGET = 5.0
N_EVAL = 16
REAL = ["Grunt", "GymLeader", "C"]
MODELS = {"Grunt": ["grunt", "gymleader", "mixture"],
          "GymLeader": ["gymleader", "grunt", "mixture"], "C": ["mixture"]}
EXTRA_USED = ["runs/w5-bs0-*/roots.jsonl", "runs/w5-bs0-*/payload_meta.json",
              "runs/w5-bs1-0-*/payload_meta.json"]


def init() -> None:
    s1.init()
    for pattern in EXTRA_USED:
        for f in Path(".").glob(pattern):
            text = f.read_text()
            rows = json.loads(text) if f.suffix == ".json" else [
                json.loads(x) for x in text.splitlines()]
            s1._S["used"].update(r["pair"] for r in rows if "pair" in r)


def make_payload(args: tuple[int, str, int, bool, int]) -> dict[str, Any]:
    root_id, real, seed, oracle, n_seeds = args
    from pokeai.cf.branching import candidate_actions, rollout
    from pokeai.cf.cf1 import choose_candidates, seed_triple
    from pokeai.cf.recon_search import Payload
    from pokeai.cf.rollouts import phase_bucket

    sim = s1._S["sim"]
    out: dict[str, Any] = {"root": root_id, "real": real}
    try:
        picked = s1.draw_root(sim, root_id, seed, real)
        info, _, _ = s1.learner_info(sim, picked)
        players, root = picked["players"], picked["root"]
        cinfo = candidate_actions(sim, players, root, "p1")
        if cinfo is None or cinfo["mask"] != picked["mask_pass1"]:
            raise RuntimeError("root decision not reproducible")
        cands = [c["slot"] for c in choose_candidates(cinfo, random.Random(seed * 7 + root_id))]
        payload = Payload(
            root_id=root_id, opp_model="mixture", seed=seed, info=info,
            learner_snapshot=root.checkpoint.players["p1"], old_tag=root.checkpoint.tag,
            mask=cinfo["mask"], candidates=cands, n_worlds=8, n_seeds=n_seeds)
        out.update(
            blob=pickle.dumps(payload), candidates=cands, mask=cinfo["mask"],
            turn=picked["turn"], phase=phase_bucket(picked["turn"]),
            pair="|".join(picked["names"]), forced_switch=picked["forced_switch"])
        if oracle:  # true-state evaluation with the REAL opponent policy (evaluation only)
            table = []
            for slot in cands:
                row = []
                for k in range(N_EVAL):
                    r = rollout(sim, players, root, "p1", slot,
                                seed_triple(root_id + 10_000, 100 + k), keep_signature=False)
                    row.append(0.5 if r.winner in (None, "tie") else 1.0 if r.winner == "p1" else 0.0)
                table.append(row)
            out["oracle"] = table
            out["oracle_illegal_free"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
        out["trace"] = traceback.format_exc()[-1500:]
    return out


def gen_payloads(jobs: list[tuple[int, str, int, bool, int]], workers: int = 4) -> list[dict[str, Any]]:
    out = []
    with mp.get_context("spawn").Pool(workers, initializer=init) as owners:
        for r in owners.imap_unordered(make_payload, jobs):
            out.append(r)
            print("payload", r["root"], r.get("real"), r.get("phase"), r.get("error", ""),
                  flush=True)
    out.sort(key=lambda r: r["root"])
    return out


def run(stage: str, roots: int, out: Path) -> None:
    from pokeai.cf.recon_search import Payload, SearchPool

    out.mkdir(parents=True, exist_ok=True)
    seed = DEADLINE_SEED if stage == "deadline" else MATRIX_SEED
    cache = out / "payloads.pkl"
    started = time.perf_counter()
    if cache.exists():
        payloads = pickle.loads(cache.read_bytes())
    else:
        if stage == "deadline":
            jobs = [(i, ["GymLeader", "Grunt"][i % 2], seed, False, 8) for i in range(roots)]
        else:
            jobs = [(i, REAL[i % 3], seed, True, 8) for i in range(roots)]
        payloads = gen_payloads(jobs)
        cache.write_bytes(pickle.dumps(payloads))
        (out / "payload_meta.json").write_text(json.dumps(
            [{k: v for k, v in p.items() if k != "blob"} for p in payloads]))
    good = [p for p in payloads if "error" not in p]
    print("payload errors:", len(payloads) - len(good), "wall", time.perf_counter() - started,
          flush=True)
    rows_file = out / "decisions.jsonl"
    done = set()
    if rows_file.exists():
        for line in rows_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["root"], r["model"]))
    pool = SearchPool()
    try:
        warm = pickle.loads(good[0]["blob"])
        pool.decision(warm, 4.0)
        pool.settle()
        gc.collect()
        gc.disable()
        with rows_file.open("a") as fh:
            for item in good:
                models = ["mixture"] if stage == "deadline" else MODELS[item["real"]]
                for model in models:
                    if (item["root"], model) in done:
                        continue
                    payload: Payload = pickle.loads(item["blob"])
                    payload.opp_model = model
                    t_req = time.time()
                    d = pool.decision(payload, BUDGET, t_request=t_req)
                    outer = time.time() - t_req
                    recycled = pool.settle()
                    ch = d["choice"]
                    row = {
                        "root": item["root"], "real": item["real"], "model": model,
                        "phase": item["phase"], "turn": item["turn"], "pair": item["pair"],
                        "n_cands": len(item["candidates"]), "candidates": item["candidates"],
                        "wall_s": d["wall_s"], "outer_wall_s": outer,
                        "n_pairs": ch["n_pairs"], "n_rollouts": d["n_rollouts"],
                        "action": ch["action"], "override": ch["override"],
                        "reason": ch["reason"], "stats": ch["stats"],
                        "legal": bool(item["mask"][ch["action"]]), "a_c": item["candidates"][0],
                        "unsupported": d["unsupported"], "errors": d["errors"],
                        "illegal": d["illegal"], "retries": d["retries"],
                        "hard_deadline_hit": d["hard_deadline_hit"],
                        "workers_done": d["workers_done"], "recycled": recycled,
                        "worlds_built": d["worlds_built"], "timing": d["timing"],
                        "opps": sorted({p["opp"] for p in d["pairs"]}),
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(item["root"], item["real"], model, row["n_pairs"], row["reason"],
                          round(outer, 3), flush=True)
                    del payload
                    gc.collect()
    finally:
        pool.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["deadline", "matrix"], required=True)
    ap.add_argument("--roots", type=int, required=True)
    args = ap.parse_args()
    run(args.stage, args.roots, Path(f"runs/w5-bs1-0-{args.stage}"))


if __name__ == "__main__":
    main()
