"""Render the frozen W4.5 findings from machine-readable evidence only."""
import json
from pathlib import Path


def main():
    root = Path("reports")
    benchmark = json.loads((root/"w4_5_ablation.json").read_text())
    paired = json.loads((root/"w4_5_paired_comparison.json").read_text())
    test = json.loads((root/"w4_5_test_results.json").read_text())
    calibration = json.loads((root/"w4_5_value_calibration.json").read_text())
    comparisons = paired["comparisons"]
    def delta(x):
        ci = x["bootstrap_95"]
        return f'{100*x["win_rate_difference"]:+.2f} pp [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] — {x["verdict"]}'
    lines = ["# W4.5 final benchmark and ablation", "",
        "All choices were frozen before evaluation. C (Probe) remains the production candidate; D is an unselected ablation, rejected by the W4.4 gate (ΔL_policy 0.0155 > 0.01). No W4.5 result changes that pick.", "",
        "## Policy finding", "",
        "Primary question: does joint BC + 0.5·value training change battle strength despite lower imitation accuracy?", "",
        "| Comparison | Random | Grunt | GymLeader | Pooled |",
        "|---|---|---|---|---|"]
    for name, results in comparisons.items():
        lines.append("| "+name+" | "+" | ".join(delta(results[k]) for k in
            ("RandomBaseline", "Grunt", "GymLeader", "pooled"))+" |")
    lines += ["", "Paired percentile bootstrap: 5,000 seeded resamples clustered by team pair; both directions remain together. Pooled resampling is stratified by opponent. Ties count as non-wins. A CI containing zero is inconclusive, not grounds to pick a higher observed win rate. Multiple comparisons are reported without multiplicity correction; D−C is primary.", "",
        "| Arm | Opponent | W/L/T | Win rate (Wilson 95% CI) | Errors / illegal / retries |",
        "|---|---|---|---|---|"]
    for opp, result in benchmark["opponents"].items():
        for arm, x in result["arms"].items():
            lo, hi = x["win_rate_wilson_95"]
            lines.append(f'| {arm} | {opp} | {x["wins"]}/{x["losses"]}/{x["ties"]} | {100*x["win_rate"]:.1f}% [{100*lo:.1f}, {100*hi:.1f}] | {x["error_count"]} / {x["illegal_actions"]} / {sum(x["protocol_retries"].values())} |')
    lines += ["", "A=v0/random preview; B=v1/random preview; C=frozen v1+probe/pretrained argmax preview; D=joint/pretrained argmax preview. Every action uses greedy policy argmax; value never guides decisions. All arms use identical paired schedules, 250 team pairs/opponent played in both directions, alternating challenger side. Showdown and heuristic RNG are not seedable. The validated team pool was never used for training or selection; W4.3 only measured on it.", "",
        "C and D differ in objective, with the same scratch initialization recipe, seed 20261016 and lr 5e-4. This is only one training seed: causal attribution needs multiple seeds.", "",
        "## Value finding", "",
        "Full decision-weighted and battle-equally-weighted metrics, battle-bootstrap CIs, reliability tables, phase (decision indices 0–5, 6–15, 16–30, 31+), first/last decision and opponent splits are in `w4_5_value_calibration.json`. Prediction spread accompanies ECE. Low ECE is not evidence of better probabilities when predictions sit near 0.5. Live opponents are bots, unlike the human training distribution.", "",
        "| Arm | Weighting | Brier | AUC | ECE | Prediction std | Mean abs(p−0.5) |",
        "|---|---|---|---|---|---|---|"]
    for arm, result in calibration["pooled"].items():
        for weighting in ("decision_weighted", "trajectory_equally_weighted"):
            x = result[weighting]["all"]
            lines.append(f'| {arm} | {weighting} | {x["brier"]:.5f} | {x["roc_auc"]:.5f} | {x["ece"]:.5f} | {x["std_probability"]:.5f} | {x["mean_absolute_p_minus_half"]:.5f} |')
    difference = calibration["c_minus_d_brier"]["pooled"]
    for weighting, estimate in difference["estimate"].items():
        ci = difference["ci95"][weighting]
        lines.append(f'- C−D live Brier ({weighting}): {estimate:+.6f}, 95% battle-bootstrap CI [{ci["lower"]:+.6f}, {ci["upper"]:+.6f}]. Negative favors C.')
    lines += ["", "Both heads rank live outcomes above chance, but Probe is poorly calibrated on these bot battles. Joint has lower live Brier and ECE and stronger discrimination; its better value prediction does not establish stronger action selection. Their live Brier comparison describes each arm's own induced battle distribution, not predictions on identical states. Compare Brier with 0.25 (constant 0.5) and the empirical-base-rate references in the JSON; discrimination does not imply calibration.", "",
        "## Architecture decision", ""]
    primary = comparisons["D-C"]["pooled"]
    lines.append("Keep Probe as production. "+(
        "Joint significantly beats C in the pooled primary comparison: it is a W5 research direction only, requiring multiple seeds and a new pre-registered selection round."
        if primary["verdict"] == "positive" else
        "Joint is not significantly stronger in the pooled primary comparison. Focus W5 on action-value learning or self-play, not promotion of Joint."))
    lines += ["", "## Test-set results", "",
        "This is the first and only use of both test splits in this final frozen evaluation; neither split was used to select these models. Test outcome labels are separately gitignored; the default train/val-only guard remains intact.",
        "", "| Split | Policy | Loss | Accuracy | Non-forced accuracy |",
        "|---|---|---|---|---|"]
    for split, models in test["policy"].items():
        for name, metrics in models.items():
            lines.append(f'| {split} | {name} | {metrics["policy_loss"]:.6f} | {100*metrics["accuracy"]:.2f}% | {100*metrics["accuracy_nonforced"]:.2f}% |')
    lines += ["", "| W4 test value | BCE | Brier | AUC | ECE |",
              "|---|---|---|---|---|"]
    for name, metrics in test["value"].items():
        lines.append(f'| {name} | {metrics["bce"]:.6f} | {metrics["brier"]:.6f} | {metrics["roc_auc"]:.6f} | {metrics["ece"]:.6f} |')
    lines += ["", "## W4 DoD status", "",
        "| Item | Status and evidence |", "|---|---|",
        "| Reproducibility | Complete: checkpoint/team hashes, commit, schedule/bootstrap seeds in `w4_5_ablation.json` and `w4_5_paired_comparison.json`; frozen model configs. |",
        "| Recent dataset | Complete: `w4_dataset_validation.json`, `datasets/w4/split_ids.json`; temporal split and overlap QA. |",
        "| BC improvement | "+("Met" if comparisons["B-A"]["pooled"]["verdict"] == "positive" else "Not demonstrated")+": B−A (data) and C−A (total) in `w4_5_paired_comparison.json`; inconclusive/negative results are not improvements. |",
        "| Team preview | Complete: `w4_team_preview.json`; C−B in `w4_5_paired_comparison.json`. |",
        "| Value head | Evaluation complete, not a guaranteed calibration-gate pass: `w4_value_metrics.json` and `w4_5_value_calibration.json`; outcome never an input. |",
        "| Reliability | "+("Met" if benchmark["valid"] else "Not met")+": `w4_5_ablation.json`; classified hidden-trap retries are reported separately. |",
        "| Final evaluation | Complete: four fixed arms, paired CIs, final checkpoints, both one-shot test splits; all four W4.5 reports. |",
        "", "## Targets", ""]
    for opp in ("Grunt", "GymLeader"):
        rate = benchmark["opponents"][opp]["arms"]["C"]["win_rate"]
        lines.append(f'- C vs {opp} 40%+ stretch target: {"met" if rate >= .4 else "not met"} ({100*rate:.1f}%).')
    random_delta = comparisons["C-A"]["RandomBaseline"]
    lines.append('- Random must not degrade: '+("met (significant improvement)" if random_delta["verdict"] == "positive" else "not met (significant decline)" if random_delta["verdict"] == "negative" else "not established (inconclusive; no formal non-inferiority margin)")+"; "+delta(random_delta)+".")
    lines += ["", "## Execution evidence", "",
        f'Full benchmark wall time: {benchmark["wall_time_s"]:.3f} s. Additional command/timing evidence: `w4_5_execution.json`.',
        "", "Machine-readable sources: `w4_5_ablation.json`, `w4_5_paired_comparison.json`, `w4_5_value_calibration.json`; test intermediate `w4_5_test_results.json`."]
    if "prior_invalid_run" in benchmark:
        lines += ["", "The first full run is preserved as `w4_5_ablation_run1.json`: one D-vs-RandomBaseline timeout following a Revival Blessing target rejection; zero model-illegal actions. The fork's Pokemon HP parser retains FNT status after a healthy numeric HP update, so cached fainted flags could select an already revived target. Baseline revival now selects explicit fainted party indices from the authoritative current request and disables native default orders during revival. Exactly one corrected full rerun is reported above; the invalid run is excluded from all estimates."]
    (root/"w4_5_final_report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
