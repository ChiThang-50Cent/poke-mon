"""BS0.1 / gate S1: reconstruct belief worlds from the learner's information only and compare
them with the true root (oracle comparison lives in `pokeai.cf.recon_oracle`)."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

POOL_DIR = "data/teams_w5/collect/gen9ou"
BS0_SEED = 20261100
USED_GLOBS = "runs/w5-cf1-*/roots.jsonl"
_S: dict[str, Any] = {}


def used_pairs() -> set[str]:
    out: set[str] = set()
    for f in Path(".").glob(USED_GLOBS):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "pair" in r:
                out.add(r["pair"])
    return out


def init() -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.restore_check import load_pool
    from pokeai.cf.sim import Sim

    sim = Sim()
    _S.update(sim=sim, sampler=BeliefSampler(sim), pool=load_pool(POOL_DIR),
              used=used_pairs())


def draw_root(sim: Any, root_id: int, seed: int, opponent: str) -> dict[str, Any]:
    from pokeai.cf.rollouts import pick_root

    pool, used = _S["pool"], _S["used"]
    names = sorted(pool)
    for attempt in range(30):
        idx = root_id * 30 + attempt
        rng = random.Random(seed * 100 + idx)
        n1, n2 = rng.choice(names), rng.choice(names)
        if f"{n1}|{n2}" in used:
            continue
        seeds = {"sim": seed + idx, "p1": seed + 1000 + idx, "p2": seed + 2000 + idx}
        teams = {"p1": pool[n1][0], "p2": pool[n2][0]}
        picked = pick_root(sim, teams, opponent, seeds, rng, greedy_prefix=True)
        if picked is not None:
            picked.update(teams=teams, names=(n1, n2), seeds=seeds)
            return picked
    raise RuntimeError("no root")


def learner_info(sim: Any, picked: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Extract exactly the learner's information from its own python state + pending frame."""
    from pokeai.cf.offline import OfflineMatch, new_tag
    from pokeai.cf.recon_oracle import true_root
    from pokeai.cf.recon_world import LearnerInfo

    cp = picked["root"].checkpoint
    players = picked["players"]
    match = OfflineMatch.restore(sim, players, cp, new_tag())
    try:
        oracle = true_root(sim, match)
        pending_lines = [x for ev in match.pending if ev["t"] == "update" for x in ev["p1"]]
        history = list(players["p1"].history.get(match.tag, []))
        request = json.loads(json.dumps(oracle["request"]))
        info = LearnerInfo(
            lines=history + pending_lines, request=request, own_team=picked["teams"]["p1"],
            pending_p1_lines=pending_lines,
        )
        return info, oracle, match.tag
    finally:
        match.close()


def one_root(args: tuple[int, str, int, int]) -> dict[str, Any]:
    root_id, opponent, n_worlds, seed = args
    from pokeai.cf.recon_oracle import compare_views, diff
    from pokeai.cf.recon_world import Unsupported, learner_probe, reconstruct_world
    from pokeai.cf.reconstruct import legal_abilities
    from pokeai.cf.rollouts import phase_bucket

    sim = _S["sim"]
    out: dict[str, Any] = {"root": root_id, "opponent": opponent}
    try:
        picked = draw_root(sim, root_id, seed, opponent)
        info, oracle, _ = learner_info(sim, picked)
        out.update(turn=picked["turn"], phase=phase_bucket(picked["turn"]),
                   forced_switch=picked["forced_switch"], names=list(picked["names"]),
                   pair="|".join(picked["names"]))
        worlds = []
        for w in range(n_worlds):
            rec: dict[str, Any] = {}
            tag = None
            try:
                recon = reconstruct_world(
                    sim, _S["sampler"], info, world_seed=seed + root_id * 100 + w,
                    battle_seed=seed + root_id)
                tag = recon.tag
                rec["supported"] = True
                rec["request_diff"] = diff(oracle["request"], recon.p1_request)
                view = sim.public_view(tag)
                exempt = {
                    m["name"] for m in view["p2"]["pokemon"]
                    if not m["is_active"] and not m["fainted"]
                    and "Regenerator" in legal_abilities(m["base_species"])
                }
                rec["view_diff"] = compare_views(oracle["view"], view, hp_exempt=exempt)
                model = learner_probe(sim, picked["players"], picked["root"].checkpoint.players["p1"],
                                      picked["root"].checkpoint.tag, info, recon)
                rec["mask"] = model["mask"] if model else None
                rec["mask_equal"] = bool(model) and model["mask"] == picked["mask_pass1"]
                rec["timing"] = recon.timing
            except Unsupported as e:
                rec.update(supported=False, reason=str(e))
            except Exception as e:  # noqa: BLE001
                rec.update(supported=False, reason="ERROR:" + repr(e)[:200],
                           trace=traceback.format_exc()[-1500:])
            finally:
                if tag:
                    with contextlib.suppress(Exception):
                        sim.destroy(tag)
            worlds.append(rec)
        out["worlds"] = worlds
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
        out["trace"] = traceback.format_exc()[-1500:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=int, default=100)
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/w5-bs0-1")
    ap.add_argument("--seed", type=int, default=BS0_SEED)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, ["GymLeader", "Grunt"][i % 2], args.worlds, args.seed) for i in range(args.roots)]
    started = time.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(args.workers, initializer=init) as pool:
        for r in pool.imap_unordered(one_root, jobs):
            results.append(r)
            print(r["root"], r.get("phase"), [w.get("supported") for w in r.get("worlds", [])],
                  r.get("error", ""), flush=True)
    results.sort(key=lambda r: r["root"])
    (out / "roots.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    summarize(results, out, time.perf_counter() - started)


def summarize(results: list[dict[str, Any]], out: Path, wall: float) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    worlds = [(r, w) for r in ok for w in r["worlds"]]
    sup = [(r, w) for r, w in worlds if w["supported"]]
    reasons = Counter(w["reason"].split(" (")[0] for r, w in worlds if not w["supported"])
    # a root is fully supported iff all its sampled worlds are supported
    root_sup = [r for r in ok if all(w["supported"] for w in r["worlds"])]
    bad_req = [(r["root"], w["request_diff"]) for r, w in sup if w["request_diff"]]
    bad_view = [(r["root"], w["view_diff"]) for r, w in sup if w["view_diff"]]
    bad_mask = [r["root"] for r, w in sup if not w["mask_equal"]]
    summary = {
        "roots": len(results), "root_errors": len(results) - len(ok),
        "worlds": len(worlds), "supported_worlds": len(sup),
        "supported_world_fraction": len(sup) / max(len(worlds), 1),
        "fully_supported_roots": len(root_sup),
        "supported_root_fraction": len(root_sup) / max(len(ok), 1),
        "unsupported_reasons": dict(reasons.most_common()),
        "request_mismatch_worlds": len(bad_req), "view_mismatch_worlds": len(bad_view),
        "mask_mismatch_worlds": len(bad_mask),
        "by_phase": dict(Counter(r["phase"] for r in ok)),
        "by_opponent": dict(Counter(r["opponent"] for r in ok)),
        "forced_switch_roots": sum(r["forced_switch"] for r in ok),
        "request_diff_paths": dict(Counter(
            p.split(":")[0] for _, d in bad_req for p in d).most_common(30)),
        "view_diff_paths": dict(Counter(
            p.split(":")[0] for _, d in bad_view for p in d).most_common(30)),
        "wall_s": wall,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
