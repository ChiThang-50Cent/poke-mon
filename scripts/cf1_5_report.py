# ruff: noqa: ISC004
"""Render reports/w5_cf1_5_report.md from reports/w5_cf1_5_head.json."""

from __future__ import annotations

import json
from pathlib import Path


def ci(x: dict) -> str:
    return f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] (n={x['n']})"


def diagnostics_section(d: dict) -> list[str]:
    md = ["## Fail-path diagnostics (CF1.5b; offline, same fresh val, no new rollouts)", "",
          "### 1. Learning curve (nested train subsets, stratified by source, 3 seeds)", "",
          "| config | roots | val Delta tau=0 | val Delta tau_sel | val Spearman | "
          "train Delta tau_sel | train Spearman |", "|---|---|---|---|---|---|---|"]
    lc = d["learning_curve"]
    for name, cfg in lc["configs"].items():
        t = cfg["tau_selected"]
        for f, row in cfg["rows"].items():
            m = row["mean_over_seeds"]
            md.append(f"| {name} (tau_sel={t}) | {lc['subset_sizes'][f]} | "
                      f"{m['val/delta_tau=0.0']:+.4f} | {m[f'val/delta_tau={t}']:+.4f} | "
                      f"{m['val/spearman']:.3f} | {m[f'train/delta_tau={t}']:+.4f} | "
                      f"{m['train/spearman']:.3f} |")
        tr = cfg["trend"][f"val/delta_tau={t}"]
        sp = cfg["trend"]["val/spearman"]
        md.append(f"| trend {name} | | | slope {tr['slope_per_100pct']:+.4f}/100% "
                  f"(upward={tr['upward']}, monotone={tr['monotone_non_decreasing']}) | "
                  f"slope {sp['slope_per_100pct']:+.3f} (upward={sp['upward']}) | | |")
    n = d["noise"]
    md += ["", "### 2. Supervised noise", "",
           "| set | split-half Pearson dQ | SB reliability dQ | dQ signal fraction | "
           "centred-Q signal fraction | mean split-half Spearman | SB rank reliability | "
           "perfect-predictor Spearman ceiling |", "|---|---|---|---|---|---|---|---|"]
    for k, v in n.items():
        md.append(f"| {k} | {v['split_half_pearson_dq']:.3f} | "
                  f"{v['spearman_brown_reliability_dq']:.3f} | {v['dq_signal_fraction']:.3f} | "
                  f"{v['centred_q_signal_fraction']:.3f} | "
                  f"{v['split_half_spearman_mean_undefined0']:.3f} | "
                  f"{v['spearman_brown_reliability_rank']:.3f} | "
                  f"{v['perfect_predictor_spearman_ceiling']:.3f} |")
    lo = d["label_oracle"]
    cap = lo["captured_fraction_of_oracle_delta_select"]
    md += ["", "Label-oracle Delta_select on fresh val (CF1.3 cross-fit): "
           f"pooled {ci(lo['pooled']['delta_select'])}, sampled "
           f"{ci(lo['sampled']['delta_select'])}, greedy {ci(lo['greedy']['delta_select'])}; "
           f"a*!=a_C {lo['pooled']['fraction_a_star_ne_a_c']:.2f}. In-sample best-minus-a_C "
           f"(optimistic) {lo['pooled']['in_sample_best_minus_a_c_optimistic']:.4f}.", "",
           f"C's log-prob alone: Spearman vs labels {ci(lo['logp_only_spearman_vs_labels'])}; "
           f"captured fraction of the oracle Delta_select: logp-only (always the "
           f"second-highest candidate) {cap['logp_only_second_highest']:+.2f}, saved head "
           f"{cap['saved_head']:+.2f}.", "", "### 3. Action coverage (train)", ""]
    cv = d["coverage"]
    md += [f"Candidates by type: {cv['train_candidates_by_type']}; by reason: "
           f"{cv['train_candidates_by_reason']}; a_C type: {cv['train_greedy_action_type']}.",
           "", "Mean rollout dQ of non-greedy candidates by type:", ""]
    for k, v in cv["mean_dq_nongreedy"]["type"].items():
        md.append(f"- {k}: n={v['n']}, {v['mean_dq']:+.4f} "
                  f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    md += ["", "By type x phase and type x opponent (full tables incl. 3-way in the JSON):", "",
           "| cell | n | mean dQ | CI |", "|---|---|---|---|"]
    for tab in ("type_x_phase", "type_x_opponent"):
        for k, v in cv["mean_dq_nongreedy"][tab].items():
            md.append(f"| {k} | {v['n']} | {v['mean_dq']:+.4f} | "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    sw = cv["head_switches_on_val"]
    md += ["", f"Head switches on val (tau={sw['tau']}): by type {sw['by_type']}; by reason "
           f"{sw['by_reason']}.", "", "### 4. Non-learned baselines on val (fitted on train)",
           "", "| rule | pooled Delta | sampled | greedy | switch rate |", "|---|---|---|---|---|"]
    for k in ("second_highest_candidate", "train_mean_type_x_phase", "tera_when_present"):
        b = d["baselines"][k]
        md.append(f"| {k} | {ci(b['all'])} | {ci(b['sampled'])} | {ci(b['greedy'])} | "
                  f"{b['switch_rate']:.3f} |")
    return [*md, ""]


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
    diag = Path("reports/w5_cf1_5b_diagnostics.json")
    if diag.exists():
        md += diagnostics_section(json.loads(diag.read_text()))
    Path("reports/w5_cf1_5_report.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
