"""Regenerate CF1 roots + sampled worlds with the CURRENT code and compare with a run.

Compares root mask/probs/candidates/turn with the recorded record, and the sha of the 8
sampled worlds under the final sampler vs the previous (pre-final-fix) sampler logic.
No rollouts are executed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import sys
from pathlib import Path

POOL_DIR = "data/teams_w5/collect/gen9ou"
_S: dict = {}
OLD = '''    ability = k.ability if known_legal else _draw(rng, ability_dist)'''
NEW = '''    ability = (
        k.ability if k.ability and not k.freeze_ability else _draw(rng, ability_dist)
    )'''


def _init() -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf import belief
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.sim import Sim

    sim = Sim()
    src = inspect.getsource(belief.sample_mon).replace(OLD, NEW)
    ns = dict(vars(belief))
    exec(src, ns)  # noqa: S102 - legacy variant of the one changed statement
    _S.update(sim=sim, pool=load_pool(POOL_DIR), belief=belief, legacy=ns["sample_mon"],
              final=belief.sample_mon)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _work(job: dict) -> dict:
    from pokeai.cf import cf1

    belief = _S["belief"]
    out = {"root": job["root"]}
    try:
        shas = {}
        for name in ("final", "legacy"):
            belief.sample_mon = _S[name]
            cf1.sample_mon = _S[name]
            sampler = belief.BeliefSampler(_S["sim"])
            ctx = cf1.build_root(_S["sim"], sampler, _S["pool"], job["seed"], job["root"])
            shas[name] = _sha(ctx.worlds)
            if name == "final":
                out["final"] = {
                    "mask": ctx.info["mask"], "probs": ctx.info["probs"],
                    "candidates": ctx.candidates, "turn": ctx.picked["turn"],
                    "pair": "|".join(ctx.names),
                }
            _S["sim"].drop_snapshot(ctx.root.checkpoint.sim_key)
        out["world_sha"] = shas
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    belief.sample_mon = _S["final"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, "src")
    from pokeai.cf import cf1_metrics as M

    run = Path(args.run_dir)
    meta = json.loads((run / "run_meta.json").read_text())["args"]
    rec = {r["root"]: r for r in M.load_run(args.run_dir)}
    jobs = [{"root": r, "seed": meta["seed"]} for r in sorted(rec)]
    with mp.get_context("spawn").Pool(4, initializer=_init) as pool:
        results = list(pool.imap_unordered(_work, jobs))
    diffs, ident = [], 0
    for res in sorted(results, key=lambda x: x["root"]):
        r = rec[res["root"]]
        if "error" in res:
            diffs.append({"root": res["root"], "why": res["error"]})
            continue
        f = res["final"]
        same_root = (
            f["mask"] == r["mask"] and f["probs"] == r["policy_probs"]
            and f["candidates"] == r["candidates"] and f["turn"] == r["turn"]
            and f["pair"] == r["pair"]
        )
        same_worlds = res["world_sha"]["final"] == res["world_sha"]["legacy"]
        if same_root and same_worlds:
            ident += 1
        else:
            diffs.append({"root": res["root"], "same_root": same_root,
                          "final_equals_legacy_worlds": same_worlds})
    summary = {"run_dir": args.run_dir, "roots": len(jobs), "identical": ident,
               "different": len(diffs), "differing": diffs}
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("roots", "identical", "different")}))


if __name__ == "__main__":
    main()
