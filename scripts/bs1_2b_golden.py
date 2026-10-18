"""BS1.2b golden-trace parity: fixed-pair search outcomes of cached roots, old tree vs new tree.

  prepare  : pick roots (dev + collect payload caches of BS0 / BS1.0; no heldout) -> roots.pkl
  run      : (PYTHONPATH = the tree under test) fixed pairs, no deadline, one shard
  compare  : old vs new per-root decision and per-pair outcomes; touched roots listed
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
import traceback
from pathlib import Path

SOURCES = ["runs/w5-bs0-3", "runs/w5-bs0-4", "runs/w5-bs1-0-matrix", "runs/w5-bs1-0-deadline"]
OUT = Path("runs/w5-bs1-2b-golden")
N_WORLDS, N_SEEDS = 8, 1  # fixed pairs: worlds 0..7 under seed 0


def prepare(per_source: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roots, seen = [], set()
    for src in SOURCES:
        items = pickle.loads((Path(src) / "payloads.pkl").read_bytes())
        items = sorted((x for x in items if "blob" in x), key=lambda x: x["root"])
        taken = 0
        for item in items:
            key = (src, item["root"])
            if key in seen or taken >= per_source:
                continue
            seen.add(key)
            roots.append({"src": src, "root": item["root"], "opponent": item.get("opponent"),
                          "blob": item["blob"], "turn": item.get("turn")})
            taken += 1
    (OUT / "roots.pkl").write_bytes(pickle.dumps(roots))
    print("roots", len(roots))


def run(shard: int, shards: int, out: str) -> None:
    import torch

    torch.set_num_threads(1)
    from pokeai.cf import recon_search as rs
    from pokeai.cf.belief import BeliefSampler
    from pokeai.cf.rollouts import make_players
    from pokeai.cf.sim import Sim

    print("tree:", rs.__file__, flush=True)
    roots = pickle.loads((OUT / "roots.pkl").read_bytes())
    mine = roots[shard::shards]
    sim = Sim()
    sampler = BeliefSampler(sim)
    players = {o: make_players(o, greedy_prefix=True) for o in ("GymLeader", "Grunt")}
    results = []
    for item in mine:
        payload = pickle.loads(item["blob"])
        if not hasattr(payload, "opp_model"):
            payload.opp_model = item["opponent"].lower()
        acq = re.compile(r"^\|-item\|.*\[from\] (move|ability): ?(Trick|Switcheroo|Bestow|Thief|"
                         r"Covet|Pickpocket|Magician|Symbiosis)\b")
        row = {"item_acquired_lines": sum(1 for x in payload.info.lines if acq.match(x)),
               "src": item["src"], "root": item["root"], "turn": item["turn"],
               "candidates": payload.candidates, "opp_model": payload.opp_model}
        emitted: list[dict] = []
        far = time.time() + 10**6
        t = time.time()
        try:
            st = rs._decide_chunk(
                sim, sampler, players, pickle.dumps(payload), rs.pair_order(N_WORLDS, N_SEEDS),
                far, far, lambda new, snap, sink=emitted: sink.extend(new))
            row["unsupported"] = sorted(st["unsupported"])
            row["errors"] = st["errors"]
            row["retries"] = st["retries"]
            row["illegal"] = st["illegal"]
            for key in ("world_rejected_mask", "dropped_pairs"):
                row[key] = st.get(key, 0)
            row["mask_rejects"] = st.get("mask_rejects", [])
            row["dropped"] = st.get("dropped", [])
        except Exception:  # noqa: BLE001 - the old tree raises on mask mismatches
            row["exception"] = traceback.format_exc().strip().splitlines()[-1][:200]
        row["pairs"] = [{"world": p["world"], "seed": p["seed"], "opp": p["opp"],
                         "outcomes": p["outcomes"]} for p in emitted]
        try:
            row["decision"] = rs.decide(payload.candidates, payload.mask, emitted)
        except AssertionError:
            row["decision"] = None
        row["wall_s"] = time.time() - t
        results.append(row)
        print(item["src"], item["root"], len(emitted), row.get("exception", ""), flush=True)
    Path(out).write_text(json.dumps(results))


def compare(old_glob: str, new_glob: str) -> dict:
    def load(pattern: str) -> dict:
        rows = {}
        for f in sorted(Path().glob(pattern)):
            for r in json.loads(f.read_text()):
                rows[(r["src"], r["root"])] = r
        return rows

    old, new = load(old_glob), load(new_glob)
    assert set(old) == set(new), "root sets differ"
    touched, identical, changed_decision = [], 0, []
    for key in sorted(old):
        o, n = old[key], new[key]
        reasons = []
        if o.get("exception"):
            reasons.append(f"old raised: {o['exception']}")
        if n.get("world_rejected_mask"):
            reasons.append(f"worlds rejected by mask: {n['world_rejected_mask']}")
        if n.get("item_acquired_lines"):
            reasons.append(f"item received mid-stint (choice-lock fix): {n['item_acquired_lines']}")
        if n.get("dropped_pairs"):
            reasons.append(f"pairs dropped (parser): {n['dropped_pairs']}")
        same_pairs = o["pairs"] == n["pairs"]
        same_dec = o["decision"] == n["decision"]
        if reasons:
            touched.append({
                "root": list(key), "reasons": reasons,
                "old_action": (o["decision"] or {}).get("action"),
                "new_action": (n["decision"] or {}).get("action"),
                "old_pairs": len(o["pairs"]), "new_pairs": len(n["pairs"]),
                "mask_rejects": [{k: m[k] for k in ("world", "bad", "live_mask", "world_mask")}
                                 for m in n.get("mask_rejects", [])]})
            if (o["decision"] or {}).get("action") != (n["decision"] or {}).get("action"):
                changed_decision.append(list(key))
        elif same_pairs and same_dec:
            identical += 1
        else:
            touched.append({"root": list(key), "reasons": ["UNEXPLAINED DIFFERENCE"],
                            "old_pairs": len(o["pairs"]), "new_pairs": len(n["pairs"])})
            changed_decision.append(list(key))
    unexplained = [t for t in touched if "UNEXPLAINED DIFFERENCE" in t["reasons"]]
    out = {
        "roots": len(old), "identical_untouched": identical, "touched": touched,
        "unexplained": len(unexplained), "decision_changed": changed_decision,
        "decision_change_rate": len(changed_decision) / len(old),
        "PASS": not unexplained and len(changed_decision) / len(old) <= 0.01,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "run", "compare"])
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--old", default="runs/w5-bs1-2b-golden/old_s*.json")
    ap.add_argument("--new", default="runs/w5-bs1-2b-golden/new_s*.json")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.per_source)
    elif args.stage == "run":
        run(args.shard, args.shards, args.out)
    else:
        res = compare(args.old, args.new)
        Path(args.out or OUT / "compare.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in res if k != "touched"}), len(res["touched"]))


if __name__ == "__main__":
    sys.exit(main())
