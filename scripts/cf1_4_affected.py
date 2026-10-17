"""Which roots does the corrected knowledge parser change? (old = fa7daaa parser)."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

_S: dict = {}


def _init(old_path: str, teams_path: str) -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.sim import Sim

    spec = importlib.util.spec_from_file_location("belief_old", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["belief_old"] = old
    spec.loader.exec_module(old)
    pool = load_pool("data/teams_w5/collect/gen9ou")
    teams = json.loads(Path(teams_path).read_text())
    _S.update(old=old, sim=Sim(), pool=pool,
              train={k: pool[k] for k in teams["train_teams"]},
              val={k: pool[k] for k in teams["val_teams"]})


def _work(job: dict) -> dict:
    sys.path.insert(0, "scripts")
    import cf1_4_collect as C

    from pokeai.cf import belief, cf1
    from pokeai.cf.offline import OfflineMatch, new_tag

    sim = _S["sim"]
    try:
        if job["group"] == "pilot":
            pool, seed, greedy = _S["pool"], C.PILOT_SEED, False
        else:
            spec = C.GROUPS[job["group"]]
            pool, seed, greedy = _S[spec["split"]], C.NEW_SEED, spec["greedy"]
        ctx = cf1.build_root(sim, None, pool, seed, job["root"], n_worlds=0,
                             greedy_prefix=greedy)
        m = OfflineMatch.restore(sim, ctx.players, ctx.root.checkpoint, new_tag())
        lines, roster = sim.public_log(m.tag), sim.roster(m.tag)
        m.close()
        old_k = _S["old"].parse_knowledge(lines, roster)
        new_k = belief.parse_knowledge(lines, roster)
        _S["old"].annotate_categories(old_k, sim)
        belief.annotate_categories(new_k, sim)
        d_old = {k: dataclasses.asdict(v) for k, v in old_k.items()}
        d_new = {k: dataclasses.asdict(v) for k, v in new_k.items()}
        changed = [k for k in d_old if d_old[k] != d_new[k]]
        return {**job, "changed": changed}
    except Exception as exc:  # noqa: BLE001
        return {**job, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--old", default="/tmp/t/belief_old.py")
    ap.add_argument("--jobs", nargs="+", required=True, help="group:root ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [{"group": x.split(":")[0], "root": int(x.split(":")[1])} for x in args.jobs]
    teams = str(Path(args.run_dir) / "teams.json")
    with mp.get_context("spawn").Pool(4, initializer=_init, initargs=(args.old, teams)) as p:
        res = list(p.imap_unordered(_work, jobs))
    Path(args.out).write_text(json.dumps(res))
    print(sum(1 for r in res if r.get("changed")), "changed of", len(res),
          "errors", sum(1 for r in res if "error" in r))


if __name__ == "__main__":
    main()
