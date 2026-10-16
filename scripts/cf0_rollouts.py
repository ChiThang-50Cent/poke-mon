"""CF0.3: counterfactual rollout pilot -> reports/w5_cf0_3_rollouts.json (+ runs/ jsonl)."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import pokeai  # noqa: F401  (import order: metamon.baselines before GenData)
from pokeai.cf.branching import candidate_actions, rollout
from pokeai.cf.restore_check import (
    TRAP_PREFIXES,
    load_pool,
    signature_mechanics,
    write_json,
)
from pokeai.cf.rollouts import (
    choose_candidates,
    outcome,
    phase_bucket,
    pick_root,
)
from pokeai.cf.sim import Sim

OPPONENTS = ["C", "GymLeader", "Grunt"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", default="data/teams_w5/collect/gen9ou")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--battles", type=int, default=90)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--out", default="reports/w5_cf0_3_rollouts.json")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = Path("runs") / f"w5-cf0-3-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pool = load_pool(args.teams)
    names = sorted(pool)
    started = time.perf_counter()
    roots, skipped = [], 0
    total_rollouts = 0
    unexpected: list[str] = []
    illegal = 0
    mech: Counter = Counter()
    walls: list[float] = []
    log = (run_dir / "rollouts.jsonl").open("w")
    with Sim() as sim:
        index = 0
        while len(roots) < args.battles:
            rng = random.Random(args.seed * 100 + index)
            opponent = OPPONENTS[len(roots) % 3]
            name1, name2 = rng.choice(names), rng.choice(names)
            seeds = {
                "sim": args.seed + index, "p1": args.seed + 1000 + index,
                "p2": args.seed + 2000 + index,
            }
            index += 1
            picked = pick_root(
                sim, {"p1": pool[name1][0], "p2": pool[name2][0]}, opponent, seeds, rng
            )
            if picked is None:
                skipped += 1
                continue
            root, players = picked["root"], picked["players"]
            info = candidate_actions(sim, players, root, "p1")
            if info is None or info["mask"] != picked["mask_pass1"]:
                raise RuntimeError("replayed root differs from pass 1 (determinism bug)")
            cands = choose_candidates(info, rng)
            root_id = len(roots)
            unexpected += [e for e in picked["pass1_errors"] if not e.startswith(TRAP_PREFIXES)]
            q = np.zeros((len(cands), args.k))
            cand_rows = []
            for ci, cand in enumerate(cands):
                wall, outs = 0.0, []
                for k in range(args.k):
                    s = 90_000_000 + root_id * 1000 + k * 3
                    result = rollout(
                        sim, players, root, "p1", cand,
                        {"sim": s, "opponent": s + 1, "learner": s + 2},
                    )
                    q[ci, k] = outcome(result.winner)
                    outs.append(q[ci, k])
                    wall += result.wall_s
                    walls.append(result.wall_s)
                    total_rollouts += 1
                    bad = [e for e in result.errors if not e.startswith(TRAP_PREFIXES)]
                    unexpected += bad
                    illegal += result.illegal_actions
                    if k == 0:
                        mech.update(signature_mechanics(result.signature))
                    log.write(json.dumps({
                        "root": root_id, "candidate": cand, "k": k,
                        "winner": result.winner, "rounds": result.rounds,
                        "wall_s": round(result.wall_s, 4), "errors": len(bad),
                        "illegal": result.illegal_actions,
                    }) + "\n")
                mean = float(q[ci].mean())
                se = float(q[ci].std(ddof=1) / math.sqrt(args.k))
                cand_rows.append({
                    "slot": cand, "prob": info["probs"][cand], "q_hat": mean, "se": se,
                    "outcomes_by_k": outs, "ties": int(sum(o == 0.5 for o in outs)),
                    "wall_s": round(wall, 3), "is_tera": cand >= 9,
                })
            log.flush()
            sim.drop_snapshot(root.checkpoint.sim_key)
            roots.append({
                "root": root_id, "opponent": opponent, "p1_team": name1, "p2_team": name2,
                "turn": picked["turn"], "phase": phase_bucket(picked["turn"]),
                "forced_switch": picked["forced_switch"],
                "eligible_decisions": picked["eligible_decisions"],
                "policy_probs": info["probs"], "mask": info["mask"],
                "probe_value": info["value"], "candidates": cand_rows,
            })
            print(f"root {root_id} ({opponent}) turn {picked['turn']} m={len(cands)}", flush=True)
    log.close()
    ordered = sorted(walls)
    report = {
        "phase": "CF0.3", "command": "scripts/cf0_rollouts.py", "seed": args.seed,
        "design": {
            "battles": args.battles, "k": args.k, "opponents": OPPONENTS,
            "opponent_assignment": "root i uses OPPONENTS[i % 3] -> 30 each",
            "learner": "C sampled T=1.0, pretrained preview argmax",
            "root_rule": "uniform random (seeded) learner model decision with >=2 legal slots; two-pass replay",
            "candidates": "top-2 by prob + one uniform other legal + best legal Tera",
            "pairing": "rollout k shares sim PRNG, opponent and learner streams across candidates",
            "target": "win=1 loss=0 tie=0.5",
            "caveat": "labels use true simulator state (real opponent team); POV-only policy inputs; no determinization",
        },
        "battles_skipped_no_eligible_decision": skipped,
        "rollouts": total_rollouts, "unexpected_error_lines": unexpected,
        "illegal_actions": illegal, "mechanics_first_rollouts": dict(mech),
        "wall": {
            "total_s": round(time.perf_counter() - started, 1),
            "rollout_mean_s": sum(walls) / len(walls),
            "rollout_p90_s": ordered[int(0.9 * (len(ordered) - 1))],
        },
        "run_dir": str(run_dir), "roots": roots,
    }
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in ("rollouts", "illegal_actions", "wall")}))
    if unexpected or illegal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
