"""CF1.4: build dataset.npz + reports/w5_cf1_4_collection.json from a collection run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pokeai.cf import cf1_metrics as M

PILOT_RUN = "runs/w5-cf1-2-20261017-094033"
CHECKS = ("legal", "public_view_equal", "learner_state_equal", "prng_unchanged",
          "root_inputs_identical", "turn_leak_free")


def load(run_dir: str) -> dict[tuple[str, int], dict]:
    last: dict[tuple[str, int], dict] = {}
    for line in (Path(run_dir) / "roots.jsonl").read_text().splitlines():
        r = json.loads(line)
        last[(r["group"], r["root"])] = r
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="reports/w5_cf1_4_collection.json")
    args = ap.parse_args()
    run = Path(args.run_dir)
    recs = load(args.run_dir)
    meta = json.loads((run / "run_meta.json").read_text())
    teams = json.loads((run / "teams.json").read_text())
    pilot_labels = {r["root"]: r for r in M.load_run(PILOT_RUN)}
    failures = [{"group": g, "root": r, "error": v["error"][-300:]}
                for (g, r), v in recs.items() if "error" in v]
    rows: list[dict] = []
    pilot_mismatch = []
    for (g, r), v in sorted(recs.items()):
        if "error" in v:
            continue
        if g == "pilot":
            lab = pilot_labels[r]
            if (v["mask"] != lab["mask"] or v["candidates"] != lab["candidates"]
                    or v["policy_probs"] != lab["policy_probs"]):
                pilot_mismatch.append(r)
                continue
            rows.append({**v, "source": "pilot", "split": "train",
                         "outcomes": lab["variants"]["belief_greedy"]["outcomes"],
                         "variant_rec": lab["variants"]["belief_greedy"], "qa": None})
        else:
            split = "val" if g.startswith("val") else "train"
            source = "greedy" if g.endswith("greedy") else "sampled"
            rows.append({**v, "source": source, "split": split,
                         "outcomes": v["variants"]["belief_greedy"]["outcomes"],
                         "variant_rec": v["variants"]["belief_greedy"]})
    n = len(rows)
    h = np.array([r["features"] for r in rows], np.float32)
    mask = np.array([r["mask"] for r in rows], np.int8)
    logp = np.where(mask > 0, np.log(np.clip(np.array([r["policy_probs"] for r in rows]),
                                              1e-30, None)), -1e4).astype(np.float32)
    slots = -np.ones((n, 4), np.int16)
    y = np.full((n, 4, 16), np.nan, np.float32)
    for i, r in enumerate(rows):
        for c, cand in enumerate(r["candidates"]):
            slots[i, c] = cand["slot"]
            y[i, c] = r["outcomes"][c]
    arrays = {
        "h": h, "logp": logp, "mask": mask, "slots": slots, "y": y,
        "root": np.array([r["root"] for r in rows]),
        "group": np.array([r["group"] for r in rows]),
        "source": np.array([r["source"] for r in rows]),
        "split": np.array([r["split"] for r in rows]),
        "opponent": np.array([r["opponent"] for r in rows]),
        "phase": np.array([r["phase"] for r in rows]),
        "forced_switch": np.array([r["forced_switch"] for r in rows]),
        "pair": np.array([r["pair"] for r in rows]),
        "p1_team": np.array([r["p1_team"] for r in rows]),
        "p2_team": np.array([r["p2_team"] for r in rows]),
        "reasons": np.array([[c["reason"] for c in r["candidates"]]
                             + [""] * (4 - len(r["candidates"])) for r in rows]),
    }
    np.savez_compressed(run / "dataset.npz", **arrays)

    def count(key: str, **where) -> dict:
        sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
        return dict(Counter(str(r[key]) for r in sel))

    by = {
        f"{s}/{src}": {
            "roots": sum(1 for r in rows if r["split"] == s and r["source"] == src),
            "opponent": count("opponent", split=s, source=src),
            "phase": count("phase", split=s, source=src),
            "forced_switch": count("forced_switch", split=s, source=src),
            "candidates": dict(Counter(
                str(len(r["candidates"])) for r in rows
                if r["split"] == s and r["source"] == src)),
        }
        for s in ("train", "val") for src in ("pilot", "sampled", "greedy")
        if any(r["split"] == s and r["source"] == src for r in rows)
    }
    new = [r for r in rows if r["source"] != "pilot"]
    # belief + determinism QA over the new roots
    worlds = [w for r in new for w in r["qa"]["worlds"]]
    qa = {c: {"ok": sum(bool(w[c]) for w in worlds), "total": len(worlds)} for c in CHECKS}
    qa["facts_violations"] = sum(len(w["facts_violations"]) for w in worlds)
    qa["world_step_errors"] = sum(len(w["errors"]) for w in worlds)
    qa["roots_with_violations"] = sum(1 for r in new if r["qa"]["violations"])
    qa["negative_control_detected"] = sum(
        1 for r in new if r["qa"].get("negative_control", {}).get("violations_detected", 0) > 0)
    qa["negative_control_total"] = sum(1 for r in new if "negative_control" in r["qa"])
    qa["true_team_legal"] = sum(r["qa"]["true_team_legal"] for r in new)
    det, unexpected, illegal, walls = {"pairs": 0, "identical": 0}, [], 0, []
    sampler = Counter()
    for r in new:
        d = r["variant_rec"]
        det["pairs"] += d["determinism"]["pairs"]
        det["identical"] += d["determinism"]["identical"]
        unexpected += d["unexpected_errors"] + r["pass1_unexpected_errors"]
        illegal += d["illegal"]
        walls += [x for row in d["wall_s"] for x in row]
        for k in ("worlds", "mon_attempts", "mon_rejections", "team_attempts",
                  "team_rejections"):
            sampler[k] += r["sampler_stats"][k]
    train_teams = {t for r in rows if r["split"] == "train" for t in (r["p1_team"], r["p2_team"])}
    val_teams = {t for r in rows if r["split"] == "val" for t in (r["p1_team"], r["p2_team"])}
    train_pairs = {r["pair"] for r in rows if r["split"] == "train"}
    val_pairs = {r["pair"] for r in rows if r["split"] == "val"}
    descriptive = {}
    for src in ("sampled", "greedy"):
        sel = [r for r in new if r["source"] == src]
        qs = [np.array(r["outcomes"], float) for r in sel]
        meta_rows = [M.root_meta(r, r["split"]) for r in sel]
        descriptive[src] = {
            "roots": len(sel),
            "c_greedy_best_fraction": float(np.mean([q.mean(axis=1).argmax() == 0 for q in qs])),
            "G4_style_select": M.g4_select(qs, meta_rows, 2000, 0)
            | {"by_group": None},
        }
    report = {
        "run_dir": args.run_dir, "seeds": {
            "new_roots": meta["new_seed"], "pilot_regeneration": meta["pilot_seed"],
            "team_partition": meta["team_seed"]},
        "groups": meta["groups"], "roots_total": n,
        "roots_by_split_source": by, "root_failures": failures,
        "pilot_regeneration": {
            "regenerated": sum(1 for r in rows if r["source"] == "pilot"),
            "mismatch_with_cf1_2_record": pilot_mismatch,
        },
        "team_split": {
            "val_reserved_teams": len(teams["val_teams"]),
            "pilot_teams": len(teams["pilot_teams"]),
            "train_teams_used": len(train_teams), "val_teams_used": len(val_teams),
            "team_overlap_train_val": len(train_teams & val_teams),
            "pair_overlap_train_val": len(train_pairs & val_pairs),
        },
        "belief_qa_new_roots": qa,
        "determinism": det, "unexpected_errors": len(unexpected),
        "unexpected_examples": unexpected[:5], "illegal_actions": illegal,
        "sampler_rejections": dict(sampler),
        "timing": {
            "rollouts": len(walls), "rollout_mean_s": float(np.mean(walls)),
            "rollout_p95_s": float(np.percentile(walls, 95)),
            "root_wall_mean_s": float(np.mean([r["wall_total_s"] for r in new])),
            "collection_wall_s": meta["wall_s"],
        },
        "descriptive_by_source": descriptive,
        "dataset": {"path": str(run / "dataset.npz"), "feature_dim": int(h.shape[1]),
                    "all_finite": bool(np.isfinite(h).all() and np.isfinite(logp).all())},
    }
    for d in descriptive.values():
        d["G4_style_select"].pop("by_group", None)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in (
        "roots_total", "determinism", "unexpected_errors", "illegal_actions")}))


if __name__ == "__main__":
    main()
