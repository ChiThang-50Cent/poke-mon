"""CF1 runner: belief-aware counterfactual rollouts over fresh-battle roots.

Collect pool only. Writes roots.jsonl (one record per root) and rollouts.jsonl (one row
per rollout) under --out-dir. Run through the resource wrapper (<=4 workers).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

POOL_DIR = "data/teams_w5/collect/gen9ou"
_STATE: dict = {}


def _init() -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.sim import Sim

    sim = Sim()
    _STATE.update(sim=sim, sampler=BeliefSampler(sim), pool=load_pool(POOL_DIR))


def _work(job: dict) -> dict:
    from pokeai.cf.belief import WorldStats
    from pokeai.cf.cf1 import process_root

    _STATE["sampler"].stats = WorldStats()
    start = time.perf_counter()
    try:
        return process_root(
            _STATE["sim"], _STATE["sampler"], _STATE["pool"], job["seed"], job["root"],
            job["variants"], job["qa"], job["repeat_frac"],
        ) | {"sampler_stats": _stats()}
    except Exception:  # noqa: BLE001 - recorded per root, counted in the report
        return {"root": job["root"], "error": traceback.format_exc(),
                "wall_total_s": time.perf_counter() - start}


def _stats() -> dict:
    s = _STATE["sampler"].stats
    return {
        "worlds": s.worlds, "mon_attempts": s.mon_attempts,
        "mon_rejections": s.mon_rejections, "team_attempts": s.team_attempts,
        "team_rejections": s.team_rejections, "reasons": dict(s.rejection_reasons),
    }


def rollout_rows(record: dict) -> list[dict]:
    rows = []
    for variant, data in record.get("variants", {}).items():
        for ci, cand in enumerate(record["candidates"]):
            for k, outcome in enumerate(data["outcomes"][ci]):
                rows.append({
                    "root": record["root"], "variant": variant, "slot": cand["slot"],
                    "reason": cand["reason"], "k": k, "world": k // 2, "seed_index": k % 2,
                    "outcome": outcome, "wall_s": data["wall_s"][ci][k],
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--control-roots", type=int, default=0,
                    help="first N roots also get oracle + T=1 control variants")
    ap.add_argument("--qa-roots", type=int, default=0)
    ap.add_argument("--repeat-frac", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root-ids", type=int, nargs="*", default=None,
                    help="re-run only these roots, appending to the output files")
    args = ap.parse_args()
    assert args.workers <= 4
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "root": r, "seed": args.seed, "qa": r < args.qa_roots,
            "repeat_frac": args.repeat_frac,
            "variants": ["belief_greedy", "oracle_greedy", "belief_t1"]
            if r < args.control_roots else ["belief_greedy"],
        }
        for r in (args.root_ids if args.root_ids is not None else range(args.roots))
    ]
    mode = "a" if args.root_ids is not None else "w"
    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init) as pool, \
            open(out / "roots.jsonl", mode) as roots_f, \
            open(out / "rollouts.jsonl", mode) as rolls_f:
        for done, record in enumerate(pool.imap_unordered(_work, jobs), 1):
            roots_f.write(json.dumps(record) + "\n")
            roots_f.flush()
            rolls_f.writelines(json.dumps(row) + "\n" for row in rollout_rows(record))
            rolls_f.flush()
            print(f"[{done}/{len(jobs)}] root {record['root']} "
                  f"{'ERROR' if 'error' in record else 'ok'} "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    meta_path = out / "run_meta.json"
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta_path.write_text(json.dumps({
        "args": previous.get("args", vars(args)),
        "wall_s": previous.get("wall_s", 0.0) + time.perf_counter() - started,
        "reruns": previous.get("reruns", [])
        + ([vars(args)] if args.root_ids is not None else []),
    }))


if __name__ == "__main__":
    main()
