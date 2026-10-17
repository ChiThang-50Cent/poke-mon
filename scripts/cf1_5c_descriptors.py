"""CF1.5c-A: backfill ActionDescriptors for the CF1.4 TRAIN roots by deterministic regeneration.

No rollouts. Each root is regenerated exactly as `cf1_4_collect.py` did; the regenerated
mask / policy probs / candidates / turn are checked against the record, the descriptor is
taken from the same root decision (the learner's `NeuralPlayer` path), and for the first
`--qa-roots` roots the descriptor is recomputed after applying each of 2 sampled worlds and
required to be bit-identical.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
POOL_DIR = "data/teams_w5/collect/gen9ou"
_S: dict = {}


def _init(teams_path: str) -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.sim import Sim

    pool = load_pool(POOL_DIR)
    teams = json.loads(Path(teams_path).read_text())
    sim = Sim()
    _S.update(sim=sim, sampler=BeliefSampler(sim), pool=pool,
              train={k: pool[k] for k in teams["train_teams"]})


def _world_descriptor(ctx, world) -> object:
    from pokeai.cf.offline import OfflineMatch, new_tag

    players = ctx.picked["players"]
    probe = OfflineMatch.restore(
        _S["sim"], players, ctx.picked["root"].checkpoint, new_tag(), record_probs=True
    )
    players["p1"].describe = True
    try:
        probe.apply_world(world)
        probe.step()
        model = next(d["model"] for d in probe.rounds[0].decisions["p1"] if d["model"])
    finally:
        players["p1"].describe = False
        probe.close()
    return model["descriptor"], model["mask"], model["probs"]


def _work(job: dict) -> dict:
    import cf1_4_collect as C

    from pokeai.cf import cf1

    spec = C.GROUPS.get(job["group"])
    seed = C.PILOT_SEED if job["group"] == "pilot" else C.NEW_SEED
    pool = _S["pool"] if job["group"] == "pilot" else _S["train"]
    greedy = bool(spec and spec["greedy"])
    out = {"root": job["root"], "group": job["group"]}
    try:
        ctx = cf1.build_root(
            _S["sim"], _S["sampler"], pool, seed, job["root"],
            n_worlds=2 if job["qa"] else 0, greedy_prefix=greedy,
        )
        rec = job["rec"]
        info = ctx.info
        out["match"] = bool(
            info["mask"] == rec["mask"] and info["probs"] == rec["policy_probs"]
            and ctx.candidates == rec["candidates"] and ctx.picked["turn"] == rec["turn"]
            and "|".join(ctx.names) == rec["pair"]
        )
        d = info["descriptor"]
        out["x"] = d.x
        out["ids"] = d.ids
        out["reasons"] = {str(k): v for k, v in d.reasons.items()}
        out["mask"] = info["mask"]
        if job["qa"]:
            same = []
            for world in ctx.worlds:
                dw, mask_w, _probs = _world_descriptor(ctx, world)
                same.append(bool(
                    np.array_equal(dw.x, d.x) and dw.ids == d.ids and dw.reasons == d.reasons
                    and mask_w == info["mask"]
                ))
            out["world_identical"] = same
        _S["sim"].drop_snapshot(ctx.root.checkpoint.sim_key)
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = traceback.format_exc() + repr(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/w5-cf1-4-20261017-104148")
    ap.add_argument("--out-dir", default="runs/w5-cf1-5c")
    ap.add_argument("--qa-roots", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, "src")
    run = Path(args.run_dir)
    recs: dict[tuple[str, int], dict] = {}
    for line in (run / "roots.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "error" not in r and r["group"] in ("pilot", "train_sampled", "train_greedy"):
            recs[(r["group"], r["root"])] = r
    keys = sorted(recs)
    if args.limit:
        keys = keys[: args.limit]
    jobs = [{"group": g, "root": r, "rec": recs[(g, r)], "qa": i < args.qa_roots}
            for i, (g, r) in enumerate(keys)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(
        4, initializer=_init, initargs=(str(run / "teams.json"),)
    ) as pool:
        results = []
        for n, res in enumerate(pool.imap(_work, jobs, chunksize=1), 1):
            results.append(res)
            if n % 100 == 0:
                print(f"[{n}/{len(jobs)}]", flush=True)
    from pokeai.cf.descriptors import DIM, FEATURE_NAMES, coverage

    ok = [r for r in results if "error" not in r]
    errors = [{"root": r["root"], "group": r["group"], "error": r["error"][-600:]}
              for r in results if "error" in r]
    x = np.stack([r["x"] for r in ok]).astype(np.float32)
    np.savez_compressed(
        out / "descriptors.npz", x=x, root=np.array([r["root"] for r in ok]),
        group=np.array([r["group"] for r in ok]),
    )
    (out / "ids.json").write_text(json.dumps(
        {f"{r['group']}:{r['root']}": r["ids"] for r in ok}))

    class _D:
        def __init__(self, r):
            self.reasons = {int(k): v for k, v in r["reasons"].items()}

    cov = coverage([(r["mask"], _D(r)) for r in ok])
    qa = [r for r in ok if "world_identical" in r]
    summary = {
        "roots": len(jobs), "ok": len(ok), "errors": errors, "dim": DIM,
        "feature_names": FEATURE_NAMES,
        "record_match": sum(r["match"] for r in ok),
        "world_qa_roots": len(qa),
        "world_qa_comparisons": sum(len(r["world_identical"]) for r in qa),
        "world_qa_identical": sum(sum(r["world_identical"]) for r in qa),
        "coverage": cov,
        "nan_inf": int((~np.isfinite(x)).sum()),
    }
    Path("reports/w5_cf1_5c_descriptors.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("feature_names", "errors")}))
    print("errors", len(errors))


if __name__ == "__main__":
    main()
