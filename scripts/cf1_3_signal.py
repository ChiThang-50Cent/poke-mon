"""CF1.3: G3 ranking + G4 selectability on the CF1.2 labels (no extra rollouts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pokeai.cf import cf1_metrics as M

SPLIT_SEED = 20260603


def block(records, variant, split_map, n_boot, n_null, seed):
    pairs = [(M.variant_q(r, variant), r) for r in records]
    pairs = [(q, r) for q, r in pairs if q is not None]
    qs = [q for q, _ in pairs]
    meta = [M.root_meta(r, split_map[r["pair"]]) for _, r in pairs]
    out = {
        "G3": M.g3_ranking(qs, [m["pair"] for m in meta], n_boot, n_null, seed),
        "G4": M.g4_select(qs, meta, n_boot, seed),
    }
    prec = [row for q in qs for row in M.delta_precision(q)]
    out["delta_q_precision"] = {
        "pairs": len(prec), "mean_se": float(np.mean([p["se"] for p in prec])),
        "mean_ci_width": float(np.mean([p["ci_width"] for p in prec])),
        "fraction_ci_excluding_0": float(np.mean([
            abs(p["dq"]) > p["ci_width"] / 2 for p in prec])),
    }
    return out, qs, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-json", default="reports/w5_cf1_3_signal.json")
    ap.add_argument("--out-md", default="reports/w5_cf1_3_report.md")
    args = ap.parse_args()
    records = [r for r in M.load_run(args.run_dir) if "error" not in r]
    split_map = M.assign_split([r["pair"] for r in records], SPLIT_SEED)
    res = {"roots": len(records)}
    qs_by, pairs_by = {}, {}
    for v in ("belief_greedy", "oracle_greedy", "belief_t1"):
        res[v], qs_by[v], pairs_by[v] = block(records, v, split_map, 5000, 1000, 0)
    # belief-vs-oracle agreement on the control roots
    ctrl = [r for r in records if "oracle_greedy" in r["variants"]]
    res["belief_vs_oracle_dQ"] = M.oracle_agreement(
        [M.variant_q(r, "belief_greedy") for r in ctrl],
        [M.variant_q(r, "oracle_greedy") for r in ctrl],
    )
    res["belief_vs_t1_dQ"] = M.oracle_agreement(
        [M.variant_q(r, "belief_greedy") for r in ctrl],
        [M.variant_q(r, "belief_t1") for r in ctrl],
    )
    # belief G3/G4 restricted to the control roots, for a like-for-like comparison
    ids = {r["root"] for r in ctrl}
    primary_ctrl = [r for r in records if r["root"] in ids]
    res["belief_greedy_on_control_roots"], _, _ = block(
        primary_ctrl, "belief_greedy", split_map, 5000, 1000, 0
    )
    res["split_counts"] = {
        s: sum(1 for r in records if split_map[r["pair"]] == s) for s in ("train", "val")
    }
    res["c_greedy_is_best_fraction"] = float(np.mean([
        q.mean(axis=1).argmax() == 0 for q in qs_by["belief_greedy"]
    ]))
    g3, g4 = res["belief_greedy"]["G3"], res["belief_greedy"]["G4"]
    res["verdict"] = {"G3_pass": g3["pass"], "G4_pass": g4["pass"]}
    Path(args.out_json).write_text(json.dumps(res, indent=1))

    def line(name, g3, g4):
        return (
            f"| {name} | {g3['roots']} | {g3['mean_undefined_as_0']:.3f} "
            f"[{g3['ci95'][0]:.3f}, {g3['ci95'][1]:.3f}] | {g3['null_p95']:.3f} | "
            f"{'PASS' if g3['pass'] else 'FAIL'} | {g4['roots']} | {g4['delta']:+.4f} "
            f"[{g4['ci95'][0]:+.4f}, {g4['ci95'][1]:+.4f}] | "
            f"{'PASS' if g4['pass'] else 'FAIL'} | {g4['fraction_a_star_ne_a_c']:.2f} |"
        )

    rows = [line(v, res[v]["G3"], res[v]["G4"])
            for v in ("belief_greedy", "oracle_greedy", "belief_t1")]
    rows.append(line("belief_greedy (control roots)",
                     res["belief_greedy_on_control_roots"]["G3"],
                     res["belief_greedy_on_control_roots"]["G4"]))
    md = ["# W5.CF1.3 signal report", "",
          f"Roots: {len(records)} (train/val {res['split_counts']}).", "",
          ("| variant | G3 roots | split-half Spearman [95% CI] | null p95 | G3 | "
           "G4 roots | delta_select [95% CI] | G4 | frac a*!=a_C |"),
          "|---|---|---|---|---|---|---|---|---|", *rows, "",
          (f"C greedy is the best candidate in {res['c_greedy_is_best_fraction']:.3f}"
           " of roots (descriptive)."), "",
          "## G4 by group (belief_greedy)", "",
          "| group | roots | delta | CI | frac a*!=a_C |", "|---|---|---|---|---|"]
    for k, v in g4["by_group"].items():
        if v.get("roots"):
            md.append(f"| {k} | {v['roots']} | {v['delta']:+.4f} | "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                      f"{v['fraction_a_star_ne_a_c']:.2f} |")
    md += ["", "## Belief vs oracle dQ (control roots, diagnostic)", "",
           f"`{json.dumps(res['belief_vs_oracle_dQ'])}`", "",
           "## dQ precision (belief_greedy)", "",
           f"`{json.dumps(res['belief_greedy']['delta_q_precision'])}`", ""]
    Path(args.out_md).write_text("\n".join(md))
    print(json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
