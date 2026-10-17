"""CF0.4: Q-signal validation over CF0.3 rollouts -> reports/w5_cf0_4_signal.json + .md."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from pokeai.cf.signal_stats import (
    binom_sf,
    bootstrap_ci,
    mean_with_undefined,
    paired_gap,
    permute_within_k,
    root_and_global_tests,
    spearman,
    split_half_stat,
    top1_agreement,
)

SEED = 20261020


def load(path: str) -> tuple[dict, list[dict], list[np.ndarray]]:
    data = json.loads(Path(path).read_text())
    roots = data["roots"]
    mats = [np.array([c["outcomes_by_k"] for c in r["candidates"]]) for r in roots]
    return data, roots, mats


def summarize(values: list[float]) -> dict:
    arr = np.array(values, float)
    n = len(arr)
    return {
        "n": n, "mean": float(arr.mean()) if n else None,
        "se": float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="reports/w5_cf0_3_rollouts.json")
    parser.add_argument("--out", default="reports/w5_cf0_4_signal.json")
    parser.add_argument("--md", default="reports/w5_cf0_4_report.md")
    parser.add_argument("--perms", type=int, default=2000)
    parser.add_argument("--boot", type=int, default=5000)
    args = parser.parse_args()
    data, roots, mats = load(args.rollouts)
    n_roots = len(roots)

    # 1. branching correctness
    branching = {
        "rollouts": data["rollouts"],
        "unexpected_error_lines": len(data["unexpected_error_lines"]),
        "illegal_actions": data["illegal_actions"],
        "pass": not data["unexpected_error_lines"] and data["illegal_actions"] == 0,
    }

    # 2. Q signal
    tests = root_and_global_tests(mats, args.perms, SEED)
    p = tests["p"]
    n_sig = int((p < 0.05).sum())
    binom_p = binom_sf(n_sig, n_roots, 0.05)
    gaps = [paired_gap(q) for q in mats]
    by: dict[str, dict[str, list]] = {"opponent": defaultdict(list), "phase": defaultdict(list)}
    for r, (gap, se) in zip(roots, gaps, strict=True):
        by["opponent"][r["opponent"]].append((gap, se))
        by["phase"][r["phase"]].append((gap, se))
    gap_table = {
        group: {
            key: {
                "roots": len(v), "mean_gap": float(np.mean([g for g, _ in v])),
                "se_across_roots": float(np.std([g for g, _ in v], ddof=1) / math.sqrt(len(v)))
                if len(v) > 1 else None,
                "mean_within_root_paired_se": float(np.nanmean([s for _, s in v])),
            }
            for key, v in sorted(table.items())
        }
        for group, table in by.items()
    }
    signal = {
        "T_global": tests["T_global"], "T_global_null_mean": tests["T_global_null_mean"],
        "p_global": tests["p_global"], "per_root_p": [float(x) for x in p],
        "fraction_p_lt_0.05": n_sig / n_roots, "roots_p_lt_0.05": n_sig,
        "binomial_p_vs_0.05": binom_p,
        "mean_best_minus_worst_gap": float(np.mean([g for g, _ in gaps])),
        "mean_gap_paired_se_across_roots": float(
            np.std([g for g, _ in gaps], ddof=1) / math.sqrt(n_roots)
        ),
        "gap_by_group": gap_table,
        "gap_note": "best/worst picked by the same means -> upward-biased (winner's curse); "
        "the permutation null, not the gap, is the evidence",
    }

    # 3. ranking stability
    rng = np.random.default_rng(SEED + 1)
    obs = [split_half_stat(q) for q in mats if len(q) >= 3]
    two = [split_half_stat(q) for q in mats if len(q) == 2]
    mean0, mean_def, n_def = mean_with_undefined(obs)
    obs0 = np.nan_to_num(np.array(obs), nan=0.0)
    ci = bootstrap_ci(obs0, args.boot, SEED + 2)
    defined = np.array([x for x in obs if not np.isnan(x)])
    ci_def = bootstrap_ci(defined, args.boot, SEED + 3) if len(defined) else (None, None)
    null_means = []
    for _ in range(1000):
        vals = [split_half_stat(permute_within_k(q, rng)) for q in mats if len(q) >= 3]
        null_means.append(float(np.nan_to_num(np.array(vals), nan=0.0).mean()))
    null_means_arr = np.array(null_means)
    top1 = [top1_agreement(q, rng) for q in mats]
    chance = [1 / len(q) for q in mats]
    stability = {
        "roots_ge3_candidates": len(obs), "roots_2_candidates": len(two),
        "mean_spearman_undefined_as_0": mean0, "ci95_bootstrap": list(ci),
        "mean_spearman_defined_only": mean_def, "n_defined": n_def,
        "ci95_defined_only": list(ci_def),
        "undefined_roots": len(obs) - n_def,
        "null_mean_spearman": float(null_means_arr.mean()),
        "null_95_interval": [
            float(np.percentile(null_means_arr, 2.5)), float(np.percentile(null_means_arr, 97.5))
        ],
        "p_vs_null": float((1 + (null_means_arr >= mean0 - 1e-12).sum()) / (1 + len(null_means_arr))),
        "two_candidate_sign_agreement_mean": float(np.mean(two)) if two else None,
        "top1_agreement": float(np.mean(top1)), "top1_chance": float(np.mean(chance)),
        "primary_statistic_note": "undefined (constant half) Spearman counted as 0; "
        "defined-only variant reported alongside",
    }

    gate = {
        "global_test_p_lt_0.05": tests["p_global"] < 0.05,
        "fraction_sig_above_0.05_binomial_p_lt_0.05": binom_p < 0.05,
        "split_half_spearman_gt0_ci_excludes_0": mean0 > 0 and ci[0] > 0,
    }
    gate["PASS"] = all(gate.values())

    # 4. descriptive
    argmax_best, rho_probs, nonarg = [], [], []
    nonarg_by = {"phase": defaultdict(int), "opponent": defaultdict(int)}
    comparisons = 0
    v_vals, q_vals, tera_effects, ties = [], [], [], 0
    for r, q in zip(roots, mats, strict=True):
        probs = np.array([c["prob"] for c in r["candidates"]])
        means = q.mean(axis=1)
        arg = int(probs.argmax())
        best = np.flatnonzero(means == means.max())
        argmax_best.append(float(arg in best) / len(best))
        rho = spearman(probs, means)
        if not np.isnan(rho):
            rho_probs.append(rho)
        found = False
        for c in range(len(q)):
            if c == arg:
                continue
            comparisons += 1
            d = q[c] - q[arg]
            se = d.std(ddof=1) / math.sqrt(len(d))
            if d.mean() > 0 and d.mean() - 1.96 * se > 0:
                found = True
        if found:
            nonarg.append(r["root"])
            nonarg_by["phase"][r["phase"]] += 1
            nonarg_by["opponent"][r["opponent"]] += 1
        w = probs / probs.sum()
        v_vals.append(r["probe_value"])
        q_vals.append(float((w * means).sum()))
        tera = [i for i, c in enumerate(r["candidates"]) if c["is_tera"]]
        if tera and len(tera) < len(q):
            rest = [i for i in range(len(q)) if i not in tera]
            tera_effects.append(float(means[tera].mean() - means[rest].mean()))
        ties += sum(c["ties"] for c in r["candidates"])
    v_arr, q_arr = np.array(v_vals, float), np.array(q_vals)
    valid = ~np.isnan(v_arr)
    descriptive = {
        "argmax_is_best_qhat_fraction": float(np.mean(argmax_best)),
        "mean_spearman_probs_vs_qhat": float(np.mean(rho_probs)) if rho_probs else None,
        "n_roots_spearman_defined": len(rho_probs),
        "roots_with_nonargmax_beating_argmax_ci_excl_0": len(nonarg),
        "comparisons_made_uncorrected": comparisons,
        "nonargmax_wins_by_phase": dict(nonarg_by["phase"]),
        "nonargmax_wins_by_opponent": dict(nonarg_by["opponent"]),
        "mean_probe_V": float(v_arr[valid].mean()) if valid.any() else None,
        "mean_prob_weighted_qhat": float(q_arr[valid].mean()) if valid.any() else None,
        "mean_V_minus_qhat": float((v_arr[valid] - q_arr[valid]).mean()) if valid.any() else None,
        "mae_V_vs_qhat": float(np.abs(v_arr[valid] - q_arr[valid]).mean()) if valid.any() else None,
        "corr_V_qhat": float(np.corrcoef(v_arr[valid], q_arr[valid])[0, 1]) if valid.sum() > 2 else None,
        "tera_candidate_effect": summarize(tera_effects),
        "tied_rollouts": ties,
    }
    out = {
        "phase": "CF0.4", "seed": SEED, "roots": n_roots, "perms": args.perms,
        "branching": branching, "signal": signal, "stability": stability,
        "gate": gate, "descriptive": descriptive,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=float) + "\n")
    Path(args.md).write_text(render(out, data))
    print(json.dumps({"gate": gate, "branching": branching["pass"]}, indent=2))


def render(o: dict, data: dict) -> str:
    s, st, d, g = o["signal"], o["stability"], o["descriptive"], o["gate"]
    yes = lambda b: "YES" if b else "NO"
    return f"""# W5 CF0.4 report: Q-signal validation (pilot, exploratory)

Rollouts: {o['branching']['rollouts']} over {o['roots']} roots (CF0.3, seed {data['seed']}).
Labels use the true simulator state (privileged); policy inputs are POV-only. One pilot.

## Q1. Is branching correct?  **{yes(o['branching']['pass'])}**
{o['branching']['unexpected_error_lines']} unexpected error lines, {o['branching']['illegal_actions']} illegal actions
over every rollout; CF0.1/0.2 gates stand.

## Q2. Is there a Q signal across candidates?  **{yes(g['global_test_p_lt_0.05'] and g['fraction_sig_above_0.05_binomial_p_lt_0.05'])}**
- Global within-k permutation test: T_global = {s['T_global']:.4f} (null mean {s['T_global_null_mean']:.4f}), p = {s['p_global']:.4g}
- Roots with p < 0.05: {s['roots_p_lt_0.05']}/{o['roots']} = {s['fraction_p_lt_0.05']:.3f}; binomial p vs 0.05 = {s['binomial_p_vs_0.05']:.4g}
- Mean best-minus-worst gap {s['mean_best_minus_worst_gap']:.3f} (SE across roots {s['mean_gap_paired_se_across_roots']:.3f}); upward-biased by selection (see JSON for opponent/phase tables).

## Q3. Is the ranking stable?  **{yes(g['split_half_spearman_gt0_ci_excludes_0'])}**
- Split-half Spearman ({st['roots_ge3_candidates']} roots with >=3 candidates; undefined as 0): mean {st['mean_spearman_undefined_as_0']:.3f}, 95% bootstrap CI [{st['ci95_bootstrap'][0]:.3f}, {st['ci95_bootstrap'][1]:.3f}]; defined-only {st['mean_spearman_defined_only']:.3f} (n={st['n_defined']})
- Permutation null mean {st['null_mean_spearman']:.3f}, 95% interval [{st['null_95_interval'][0]:.3f}, {st['null_95_interval'][1]:.3f}], p = {st['p_vs_null']:.4g}
- Top-1 agreement {st['top1_agreement']:.3f} vs chance {st['top1_chance']:.3f}; 2-candidate sign agreement {st['two_candidate_sign_agreement_mean']}

## Gate: **{'PASS' if g['PASS'] else 'FAIL'}**

## Descriptive (not gated)
- C's argmax is best Q-hat candidate: {d['argmax_is_best_qhat_fraction']:.3f}
- Mean Spearman(probs, Q-hat): {d['mean_spearman_probs_vs_qhat']}
- Non-argmax beats argmax (paired CI excl. 0): {d['roots_with_nonargmax_beating_argmax_ci_excl_0']} roots of {d['comparisons_made_uncorrected']} uncorrected comparisons; by phase {d['nonargmax_wins_by_phase']}, by opponent {d['nonargmax_wins_by_opponent']}
- Probe V vs prob-weighted Q-hat: mean V {d['mean_probe_V']}, mean Q-hat {d['mean_prob_weighted_qhat']}, MAE {d['mae_V_vs_qhat']}, corr {d['corr_V_qhat']}
- Tera candidate effect (Q-hat tera minus others): {d['tera_candidate_effect']}
- Tied rollouts: {d['tied_rollouts']}
"""


if __name__ == "__main__":
    main()
