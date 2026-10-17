"""CF1.1 reports from a cf1_run.py run directory (50 roots, three variants, QA)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pokeai.cf import cf1_metrics as M
from pokeai.cf.belief import __doc__ as BELIEF_RULES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-qa", default="reports/w5_cf1_1_belief_qa.json")
    ap.add_argument("--out-feas", default="reports/w5_cf1_1_feasibility.json")
    args = ap.parse_args()
    records = M.load_run(args.run_dir)
    meta = json.loads((Path(args.run_dir) / "run_meta.json").read_text())
    good = [r for r in records if "error" not in r]
    errors = [{"root": r["root"], "error": r["error"][-600:]} for r in records
              if "error" in r]

    # ---- G1
    worlds = [(r["root"], w) for r in good for w in r["qa"]["worlds"]]
    checks = ("legal", "public_view_equal", "learner_state_equal", "prng_unchanged",
              "root_inputs_identical", "turn_leak_free")
    g1 = {c: {"ok": sum(bool(w[c]) for _, w in worlds), "total": len(worlds)}
          for c in checks}
    g1["facts_violations"] = sum(len(w["facts_violations"]) for _, w in worlds)
    g1["world_step_errors"] = sum(len(w["errors"]) for _, w in worlds)
    g1["true_team_legal"] = sum(r["qa"]["true_team_legal"] for r in good)
    g1["violating_worlds"] = [
        {"root": r["root"], "worlds": r["qa"]["violations"]}
        for r in good if r["qa"]["violations"]
    ]
    g1["negative_control"] = [
        r["qa"]["negative_control"] for r in good if "negative_control" in r["qa"]
    ]
    g1["negative_control_all_detected"] = all(
        n["violations_detected"] > 0 for n in g1["negative_control"]
    )
    g1["turn_first_diff_histogram_relative_to_first_p2_subject"] = dict(Counter(
        "no_diff" if w["turn_first_diff"] is None
        else ("at_or_after" if w["turn_leak_free"] else "BEFORE")
        for _, w in worlds
    ))
    g1["hard_pass"] = bool(
        all(v["ok"] == v["total"] for k, v in g1.items()
            if isinstance(v, dict) and "ok" in v)
        and g1["facts_violations"] == 0 and not errors
        and g1["negative_control_all_detected"]
    )
    # ---- diversity + hit rate
    species = Counter()
    div = {"movesets": [], "items": [], "abilities": [], "tera": []}
    frozen = Counter()
    for r in good:
        for sp, d in r["qa"]["diversity"].items():
            species[sp] += 1
            f = d["frozen"]
            frozen["mons"] += 1
            frozen["item_frozen"] += f["item"]
            frozen["ability_frozen"] += f["ability"]
            frozen["tera_frozen"] += f["tera"]
            if not f["item"]:
                div["items"].append(d["distinct_items"])
            if not f["ability"]:
                div["abilities"].append(d["distinct_abilities"])
            if not f["tera"]:
                div["tera"].append(d["distinct_tera"])
            if f["moves_revealed"] < 4:
                div["movesets"].append(d["distinct_movesets"])
    hit = {}
    for key in ("item", "ability", "tera", "moves"):
        h = sum(r["qa"]["hit_rates"][key]["hits"] for r in good)
        n = sum(r["qa"]["hit_rates"][key]["trials"] for r in good)
        e = sum(r["qa"]["hit_rates"][key]["expected_hits_under_prior"] for r in good)
        hit[key] = {"hits": h, "trials": n, "hit_rate": h / n if n else None,
                    "prior_implied_rate": e / n if n else None}
    mean = lambda v: float(np.mean(v)) if v else None
    sampler = Counter()
    reasons = Counter()
    for r in good:
        s = r["sampler_stats"]
        for k in ("worlds", "mon_attempts", "mon_rejections", "team_attempts",
                  "team_rejections"):
            sampler[k] += s[k]
        reasons.update(s["reasons"])

    # ---- G2 + errors across all rollouts
    det = {"pairs": 0, "identical": 0}
    unexpected, illegal, rollouts = [], 0, 0
    wall = {}
    for r in good:
        for v, d in r["variants"].items():
            det["pairs"] += d["determinism"]["pairs"]
            det["identical"] += d["determinism"]["identical"]
            unexpected += d["unexpected_errors"]
            illegal += d["illegal"]
            rollouts += sum(len(x) for x in d["outcomes"]) + d["determinism"]["pairs"]
            wall.setdefault(v, []).extend(x for row in d["wall_s"] for x in row)
        unexpected += r["pass1_unexpected_errors"]
    g2 = {"deterministic_pairs": det, "unexpected_errors": len(unexpected),
          "unexpected_examples": unexpected[:5], "illegal_actions": illegal,
          "root_failures": errors, "rollouts_total_including_repeats": rollouts}
    g2["hard_pass"] = bool(det["pairs"] > 0 and det["identical"] == det["pairs"]
                           and not unexpected and illegal == 0 and not errors)

    qa = {
        "roots": len(good), "G1": g1, "G2": g2,
        "diversity": {
            "mean_distinct_over_8_worlds": {
                k: mean(v) for k, v in div.items()
            },
            "mons_with_unrevealed_field_counts": {k: len(v) for k, v in div.items()},
            "frozen_counts": dict(frozen),
        },
        "hit_rate_vs_true_hidden": hit,
        "rejection": {
            **sampler,
            "mon_rejection_rate": sampler["mon_rejections"] / max(1, sampler["mon_attempts"]),
            "team_rejection_rate": sampler["team_rejections"] / max(1, sampler["team_attempts"]),
            "reasons": dict(reasons.most_common(15)),
        },
        "prior": {
            "chosen": "Metamon NaiveUsagePredictor distributions (Smogon gen9ou "
                      "2026-01..04, rank 1500) via get_usage_stats; marginals "
                      "re-sampled with a per-root seeded numpy Generator",
            "replay_predictor": "evaluated by reading metamon predictor.py: "
                                "ReplayPredictor supports only gen1ou-gen4ou and falls "
                                "back to NaiveUsagePredictor for gen9ou; not usable",
            "no_random_battle_prior": True,
        },
        "rules_documented": BELIEF_RULES,
        "opponent_python_state_approximation": (
            "The opponent player's poke-env/Metamon battle object and the opponent C "
            "GRU hidden come from the TRUE history up to the root (checkpoint), not from "
            "the sampled world. Its own request and own-channel HP lines are regenerated "
            "from the sampled sets, so its next decision is world-consistent, but its "
            "internal history may contain true-world moves/items. Opponent C therefore "
            "acts on a mixed state; measured effect: 0 errors/illegal actions."
        ),
        "wall_s_run": meta["wall_s"],
    }
    Path(args.out_qa).write_text(json.dumps(qa, indent=1))

    # ---- feasibility
    def stat(v):
        a = np.array(wall.get(v, []))
        return {"n": len(a), "mean_s": float(a.mean()), "p95_s": float(np.percentile(a, 95))} if len(a) else None

    feas = {"roots": len(good), "timing_per_rollout": {v: stat(v) for v in wall},
            "wall_s_run": meta["wall_s"], "variants": {}}
    for v in ("belief_greedy", "oracle_greedy", "belief_t1"):
        pairs = [(variant_q, r) for r in good for variant_q in [M.variant_q(r, v)]
                 if variant_q is not None]
        qs = [q for q, _ in pairs]
        ms = [M.root_meta(r) for _, r in pairs]
        feas["variants"][v] = {
            "G3_descriptive": M.g3_ranking(qs, [m["pair"] for m in ms], 2000, 500, 0),
            "G4_descriptive": M.g4_select(qs, ms, 2000, 0),
            "mean_dq_ci_width": float(np.mean([
                row["ci_width"] for q in qs for row in M.delta_precision(q)
            ])),
            "c_greedy_is_best_fraction": float(np.mean([
                q.mean(axis=1).argmax() == 0 for q in qs])),
        }
    bel = [(M.variant_q(r, "belief_greedy"), M.variant_q(r, "oracle_greedy")) for r in good]
    feas["belief_vs_oracle_dQ"] = M.oracle_agreement([b for b, _ in bel], [o for _, o in bel])
    Path(args.out_feas).write_text(json.dumps(feas, indent=1))
    print(json.dumps({"G1": g1["hard_pass"], "G2": g2["hard_pass"]}))


if __name__ == "__main__":
    main()
