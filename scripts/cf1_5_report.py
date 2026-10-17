# ruff: noqa: ISC004
"""Render reports/w5_cf1_5_report.md from reports/w5_cf1_5_head.json."""

from __future__ import annotations

import json
from pathlib import Path


def ci(x: dict) -> str:
    return f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] (n={x['n']})"


def main() -> None:
    r = json.loads(Path("reports/w5_cf1_5_head.json").read_text())
    col = json.loads(Path("reports/w5_cf1_4_collection.json").read_text())
    sel = r["selected"]
    md = [
        "# W5.CF1.5 head report (frozen C; offline)", "",
        "**Val was used for early stopping, (config, tau) selection: every CI below is "
        "development evidence only.**", "",
        f"Selected: {sel['arch']} / {sel['loss']} / tau={sel['tau']} (saved head = "
        f"seed {sel['saved_seed']}); train {r['n_train']} roots (400 pilot + 900 sampled "
        f"+ 300 greedy), fresh val {r['n_val']} roots (300 sampled + 100 greedy).", "",
        f"Gates: H1 {'PASS' if r['gates']['H1'] else 'FAIL'}, "
        f"H2 {'PASS' if r['gates']['H2'] else 'FAIL'}, "
        f"H4 {'PASS' if r['gates']['H4'] else 'FAIL'}; CF1.6 eligible: "
        f"{r['gates']['cf1_6_eligible']}.", "",
        "## Grid: mean-over-3-seeds val Delta_head by tau", "",
        "| config | tau=0 | 0.01 | 0.02 | 0.05 |", "|---|---|---|---|---|",
    ]
    for arch in ("linear", "mlp"):
        for loss in ("huber", "pairwise"):
            cells = [f"{r['grid'][f'{arch}/{loss}/tau={t}']['mean_over_seeds']:+.4f}"
                     for t in (0.0, 0.01, 0.02, 0.05)]
            md.append(f"| {arch}/{loss} | " + " | ".join(cells) + " |")
    md += ["", "## Fresh val (selected head, saved seed)", "",
           "| subset | roots | H2 Delta_head | switch rate | Delta given switch | "
           "control Delta (3 seeds) |", "|---|---|---|---|---|---|"]
    for name in ("pooled", "sampled", "greedy"):
        v = r["val"][name]
        given = v["delta_given_switch"]
        md.append(
            f"| {name} | {v['roots']} | {ci(v['delta_head'])} | {v['switch_rate']:.3f} "
            f"({v['switches']}) | {given['mean']:+.4f} (n={given['n']}) | "
            + ", ".join(f"{x:+.4f}" for x in v["control_delta_head"]) + " |")
    md += ["", "## H1 ranking vs shuffled-label control", "",
           "| subset | head Spearman | control Spearman | paired diff | beats control |",
           "|---|---|---|---|---|"]
    for name in ("pooled", "sampled", "greedy"):
        s = r["val"][name]["H1"]["spearman"]
        md.append(f"| {name} | {ci(s['head'])} | {ci(s['control'])} | "
                  f"{ci(s['paired_diff'])} | {s['beats_control']} |")
    sg = r["val"]["pooled"]["H1"]["sign_acc"]
    md += ["", f"2-candidate roots with a non-zero rollout dQ: {sg['roots']} (too few to "
           "evaluate sign accuracy).", "",
           f"Seed spread of val Delta_head: {r['seed_spread']['val_delta_head_per_seed']}; "
           f"switch rates {r['seed_spread']['switch_rate_per_seed']}.", "",
           f"Train split (overfit check): Delta_head {ci(r['train']['pooled']['delta_head'])}, "
           f"switch rate {r['train']['pooled']['switch_rate']:.3f}.", "",
           f"H3 (greedy val roots): Delta {ci(r['val']['greedy']['delta_head'])}; "
           f"warning={r['H3_warning']}.", "",
           f"H4: {json.dumps(r['H4'])}", "",
           "## Delta_head by group (pooled val)", "", "| group | roots | Delta | switch |",
           "|---|---|---|---|"]
    for k, v in r["val"]["by_group"].items():
        md.append(f"| {k} | {v['roots']} | {v['delta_head']:+.4f} | {v['switch_rate']:.3f} |")
    md += ["", "## Reading", "",
           "The control reaches the same per-root Spearman as the head: that ranking is "
           "carried by C's own log-prob input, not by the learned state-action link. The "
           "head overfits quickly (train loss falls while val Delta_head worsens). "
           "CF1.6 is not run because H1 and H2 fail.", "",
           f"Data: {col['roots_total']} roots from `{col['run_dir']}`.", ""]
    Path("reports/w5_cf1_5_report.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
