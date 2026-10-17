"""CF1.4: scale belief-aware counterfactual labels (+ C's frozen root features).

Groups (pre-registered): train_sampled 900, train_greedy 300, val_sampled 300,
val_greedy 100 new roots, plus deterministic regeneration of the 400 CF1.2 pilot roots
(features only; their labels stay in the CF1.2 run). Validation battles draw both teams
from a reserved subset of collect-pool teams that no pilot or new-train battle uses.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

POOL_DIR = "data/teams_w5/collect/gen9ou"
PILOT_DIR = "runs/w5-cf1-2-20261017-094033"
PILOT_SEED = 20260602
NEW_SEED = 20261010
N_VAL_TEAMS = 250
TEAM_SEED = 20261009
GROUPS = {
    "train_sampled": {"split": "train", "greedy": False, "n": 900, "offset": 1000},
    "train_greedy": {"split": "train", "greedy": True, "n": 300, "offset": 2000},
    "val_sampled": {"split": "val", "greedy": False, "n": 300, "offset": 3000},
    "val_greedy": {"split": "val", "greedy": True, "n": 100, "offset": 4000},
}
_S: dict = {}


def team_split(pilot_dir: str, all_teams: list[str]) -> dict:
    pilot = [json.loads(x) for x in (Path(pilot_dir) / "roots.jsonl").read_text().splitlines()]
    used = sorted({t for r in pilot if "error" not in r for t in (r["p1_team"], r["p2_team"])})
    free = sorted(set(all_teams) - set(used))
    order = np.random.default_rng(TEAM_SEED).permutation(len(free))
    val = sorted(free[i] for i in order[:N_VAL_TEAMS])
    train = sorted(set(all_teams) - set(val))
    return {"pilot_teams": used, "val_teams": val, "train_teams": train,
            "pool_size": len(all_teams), "seed": TEAM_SEED}


def _init(teams_path: str) -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.sim import Sim

    pool = load_pool(POOL_DIR)
    teams = json.loads(Path(teams_path).read_text())
    sim = Sim()
    _S.update(
        sim=sim, sampler=BeliefSampler(sim), pool=pool,
        train={k: pool[k] for k in teams["train_teams"]},
        val={k: pool[k] for k in teams["val_teams"]},
    )


def _work(job: dict) -> dict:
    from pokeai.cf.belief import WorldStats
    from pokeai.cf.cf1 import process_root

    _S["sampler"].stats = WorldStats()
    start = time.perf_counter()
    try:
        if job["group"] == "pilot":
            rec = process_root(
                _S["sim"], _S["sampler"], _S["pool"], PILOT_SEED, job["root"], [],
                False, 0.0, greedy_prefix=False, n_worlds=0,
            )
        else:
            spec = GROUPS[job["group"]]
            rec = process_root(
                _S["sim"], _S["sampler"], _S[spec["split"]], NEW_SEED, job["root"],
                ["belief_greedy"], True, 0.1, greedy_prefix=spec["greedy"],
            )
        s = _S["sampler"].stats
        rec["sampler_stats"] = {
            "worlds": s.worlds, "mon_attempts": s.mon_attempts,
            "mon_rejections": s.mon_rejections, "team_attempts": s.team_attempts,
            "team_rejections": s.team_rejections,
        }
    except Exception:  # noqa: BLE001 - recorded per root and counted in the report
        rec = {"root": job["root"], "error": traceback.format_exc(),
               "wall_total_s": time.perf_counter() - start}
    rec["group"] = job["group"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--groups", nargs="*",
                    default=["pilot", *GROUPS])
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these 'group:root' ids (append mode; may be replacement ids)")
    args = ap.parse_args()
    assert args.workers <= 4
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    teams_path = out / "teams.json"
    if not teams_path.exists():
        from pokeai.cf.restore_check import load_pool

        teams_path.write_text(json.dumps(team_split(PILOT_DIR, sorted(load_pool(POOL_DIR)))))
    jobs = []
    for g in args.groups:
        if g == "pilot":
            jobs += [{"group": g, "root": r} for r in range(400)]
        else:
            spec = GROUPS[g]
            jobs += [{"group": g, "root": spec["offset"] + i} for i in range(spec["n"])]
    if args.only is not None:  # explicit ids (re-runs and replacement roots)
        jobs = [{"group": x.split(":")[0], "root": int(x.split(":")[1])} for x in args.only]
    mode = "a" if args.only is not None else "w"
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_init, initargs=(str(teams_path),)
    ) as pool, open(out / "roots.jsonl", mode) as f:
        for done, rec in enumerate(pool.imap(_work, jobs, chunksize=1), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{done}/{len(jobs)}] {rec['group']}:{rec['root']} "
                  f"{'ERROR' if 'error' in rec else 'ok'} "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    meta_path = out / "run_meta.json"
    prev = json.loads(meta_path.read_text()) if meta_path.exists() else {"wall_s": 0.0}
    meta_path.write_text(json.dumps({
        "wall_s": prev["wall_s"] + time.perf_counter() - started,
        "new_seed": NEW_SEED, "pilot_seed": PILOT_SEED, "team_seed": TEAM_SEED,
        "groups": GROUPS,
    }))


if __name__ == "__main__":
    main()
