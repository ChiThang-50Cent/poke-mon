"""CF1.2: per-root label report from a cf1_run.py directory (400 roots)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pokeai.cf import cf1_metrics as M

SPLIT_SEED = 20260603


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="reports/w5_cf1_2_labels.json")
    ap.add_argument("--reproducibility", nargs="*", default=[],
                    help="JSON files from cf1_repro_check.py to embed")
    args = ap.parse_args()
    records = M.load_run(args.run_dir)
    meta = json.loads((Path(args.run_dir) / "run_meta.json").read_text())
    good = [r for r in records if "error" not in r]
    split = M.assign_split([r["pair"] for r in good], SPLIT_SEED)
    roots, det = [], {"pairs": 0, "identical": 0}
    unexpected, illegal, walls = [], 0, {}
    rejections = {"mon_attempts": 0, "mon_rejections": 0, "team_attempts": 0,
                  "team_rejections": 0}
    for r in good:
        entry = {
            "root": r["root"], "opponent": r["opponent"], "pair": r["pair"],
            "split": split[r["pair"]], "turn": r["turn"], "phase": r["phase"],
            "forced_switch": r["forced_switch"], "a_c": r["a_c"],
            "candidates": [
                {k: c[k] for k in ("slot", "reason", "inclusion_prob", "prob")}
                for c in r["candidates"]
            ],
            "variants": {}, "wall_total_s": r["wall_total_s"],
        }
        for v, d in r["variants"].items():
            q = np.array(d["outcomes"], float)
            prec = M.delta_precision(q)
            entry["variants"][v] = {
                "q_hat": q.mean(axis=1).tolist(),
                "delta_q_hat": (q.mean(axis=1) - q[0].mean()).tolist(),
                "delta_q_se": [0.0] + [p["se"] for p in prec],
                "q_hat_halves": [q[:, :8].mean(axis=1).tolist(),
                                 q[:, 8:].mean(axis=1).tolist()],
                "rollout_wall_mean_s": float(np.mean(d["wall_s"])),
            }
            det["pairs"] += d["determinism"]["pairs"]
            det["identical"] += d["determinism"]["identical"]
            unexpected += d["unexpected_errors"]
            illegal += d["illegal"]
            walls.setdefault(v, []).extend(x for row in d["wall_s"] for x in row)
        unexpected += r["pass1_unexpected_errors"]
        for k in rejections:
            rejections[k] += r["sampler_stats"][k]
        roots.append(entry)
    counts = {s: sum(1 for e in roots if e["split"] == s) for s in ("train", "val")}
    pair_counts = {s: sum(1 for p in split.values() if p == s) for s in ("train", "val")}
    raw = [json.loads(x) for x in (Path(args.run_dir) / "roots.jsonl").read_text().splitlines()]
    reproducibility = {
        "raw_records": len(raw),
        "superseded_error_records": sum(
            1 for r in raw if "error" in r and any(
                "error" not in o and o["root"] == r["root"] for o in raw)
        ),
        "report_records_per_root_unique": len({e["root"] for e in roots}) == len(roots),
        "report_roots_with_error": sum(1 for r in good if "error" in r),
        "checks": [json.loads(Path(f).read_text()) for f in args.reproducibility],
    }
    out = {
        "reproducibility": reproducibility,
        "roots": len(roots), "root_failures": [
            {"root": r["root"], "error": r["error"][-500:]} for r in records if "error" in r
        ],
        "split": {"seed": SPLIT_SEED, "unit": "team pair (p1_team|p2_team)",
                  "roots": counts, "pairs": pair_counts},
        "determinism": det, "unexpected_errors": len(unexpected),
        "unexpected_examples": unexpected[:5], "illegal_actions": illegal,
        "sampler_rejections": rejections,
        "timing": {v: {"rollouts": len(w), "mean_s": float(np.mean(w)),
                       "p95_s": float(np.percentile(w, 95))} for v, w in walls.items()},
        "wall_s_run": meta["wall_s"], "run_dir": args.run_dir,
        "labels": roots,
    }
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ("roots", "split", "determinism",
                                          "unexpected_errors", "illegal_actions")}))


if __name__ == "__main__":
    main()
