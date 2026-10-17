"""W5.3 dev-only staged AWBC benchmark and evidence-based candidate selection.

Run each --stage separately under the repository's systemd resource wrapper.
No stage accesses heldout teams or promotes the production baseline C.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from evaluate_w4_5 import bootstrap, cluster_differences, verdict
from evaluate_w4_team_preview import arm_report, sha256
from evaluate_w5 import claim_pool, run_config

from pokeai.env.collector import git_commit
from pokeai.env.runner import run_battles, wilson_interval
from pokeai.repro import source_fingerprint
from pokeai.training.awbc import select_median_seed

BETAS = ("0.5", "1.0", "2.0")
SEEDS = (1, 2, 3)
STAGES = {"smoke": 20, "screen": 200, "confirm": 400,
          "control": 400, "sampling": 200}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stage_path(args, stage):
    return Path(args.reports) / f"w5_3_{stage}.json"


def completed_stage(args, stage):
    report = load(stage_path(args, stage))
    if not report.get("valid"):
        raise ValueError(f"{stage} stage must be complete and valid")
    return report


def arm_name(experiment, beta, seed):
    return f"{experiment}-{beta}-s{seed}"


def screen_names():
    return [arm_name("E1", "none", 1)] + [
        arm_name(exp, beta, 1) for exp in ("E2", "E3") for beta in BETAS
    ]


def selected_betas(screen):
    return {exp: max(BETAS, key=lambda beta: screen["statistics"]["arms"][
        arm_name(exp, beta, 1)]["pooled"]["win_rate"])
            for exp in ("E2", "E3")}


def selected_names(betas):
    return [arm_name(exp, beta, seed)
            for exp, beta in {"E1": "none", **betas}.items() for seed in SEEDS]


def read_rows(result):
    return [json.loads(line) for line in
            Path(result["battles_path"]).read_text().splitlines()]


def pooled_bootstrap(strata, seed):
    """Resample team pairs within opponents, retaining seed dependence.

    Each stratum already averages the same schedule's differences across training
    seeds. The interval conditions on the three fitted policies; seed variation
    is reported separately, not treated as independent repeated battles.
    """
    rng = np.random.default_rng(seed)
    estimates = np.zeros(5000)
    for values in strata.values():
        for start in range(0, 5000, 128):
            n = min(128, 5000 - start)
            estimates[start:start+n] += values[
                rng.integers(len(values), size=(n, len(values)))
            ].mean(axis=1) / len(strata)
    interval = np.quantile(estimates, [.025, .975]).tolist()
    return {"win_rate_difference": float(np.mean([v.mean() for v in strata.values()])),
            "bootstrap_95": interval, "verdict": verdict(interval),
            "clusters": sum(len(v) for v in strata.values()),
            "seed": seed, "resamples": 5000}


def comparison(report, treatments, controls, seed):
    by_seed = {}
    strata = {opp: [] for opp in report["opponents"]}
    for treatment, control in zip(treatments, controls, strict=True):
        deltas = {}
        for opponent, results in report["opponents"].items():
            values = cluster_differences(read_rows(results[control]),
                                         read_rows(results[treatment]))
            deltas[opponent] = values
            strata[opponent].append(values)
        by_seed[treatment] = {
            "control": control,
            "opponents": {opp: bootstrap(v, seed=seed+i)
                          for i, (opp, v) in enumerate(deltas.items())},
            "pooled": pooled_bootstrap(deltas, seed+100),
        }
    averaged = {opp: np.stack(values).mean(axis=0)
                for opp, values in strata.items()}
    per_opponent = {opp: bootstrap(v, seed=seed+i)
                    for i, (opp, v) in enumerate(averaged.items())}
    seed_means = {name: result["pooled"]["win_rate_difference"]
                  for name, result in by_seed.items()}
    severe = [opp for opp, result in per_opponent.items()
              if result["win_rate_difference"] < -0.05
              and result["bootstrap_95"][1] < 0]
    return {"pooled": pooled_bootstrap(averaged, seed+100),
            "opponents": per_opponent, "by_seed": by_seed,
            "seed_means": seed_means,
            "positive_seed_count": sum(v > 0 for v in seed_means.values()),
            "severe_regression_opponents": severe}


def statistics(report):
    names = list(report["arms"])
    result = {"arms": {}, "comparisons": {}}
    for name in names:
        wins = sum(r[name]["wins"] for r in report["opponents"].values())
        total = sum(r[name]["completed"] for r in report["opponents"].values())
        result["arms"][name] = {
            "pooled": {"wins": wins, "completed": total,
                       "win_rate": wins / total,
                       "win_rate_wilson_95": list(wilson_interval(wins, total))},
            "opponents": {opp: {key: rows[name][key] for key in (
                "wins", "completed", "win_rate", "win_rate_wilson_95")}
                for opp, rows in report["opponents"].items()}}
    if report["stage"] == "smoke":
        return result
    for index, name in enumerate(names):
        if name != "C":
            result["comparisons"][name+"-C"] = comparison(
                report, [name], ["C"], 20263000+index*1000)
    for exp in ("E1", "E2", "E3", "E2-shuffled", "E3-shuffled"):
        treatments = [name for name in names if name.rsplit("-", 2)[0] == exp]
        if not treatments:
            continue
        result["comparisons"][exp+"-C"] = comparison(
            report, treatments, ["C"]*len(treatments), 20268000)
        if exp != "E1" and all(arm_name("E1", "none", int(n.rsplit("s", 1)[1]))
                               in names for n in treatments):
            result["comparisons"][exp+"-E1"] = comparison(
                report, treatments,
                [arm_name("E1", "none", int(n.rsplit("s", 1)[1]))
                 for n in treatments], 20269000)
        if exp in ("E2", "E3"):
            controls = [name.replace(exp+"-", exp+"-shuffled-", 1)
                        for name in treatments]
            if all(name in names for name in controls):
                result["comparisons"][exp+"-shuffled"] = comparison(
                    report, treatments, controls, 20270000)
    return result


def definitions(args, names, sample=False):
    config = yaml.safe_load(Path(args.config).read_text())
    baseline = dict(config["arms"]["C"])
    baseline.update(greedy=not sample, temperature=1.0)
    arms = {"C": baseline}
    for name in names:
        arms[name] = {**baseline, "model_dir": str(Path(args.checkpoints)/name)}
    opponents = {name: config["opponents"][name] for name in (
        "RandomBaseline", "Grunt", "GymLeader", "EmeraldKaizo")}
    opponents["v1"] = {"model_dir": "models/pokedecision-v1",
                       "greedy": True,
                       "preview_checkpoint": baseline["preview_checkpoint"],
                       "preview_argmax": True}
    return arms, opponents


async def evaluate(args):
    started = time.perf_counter()
    stage = args.stage
    output = stage_path(args, stage)
    if output.exists():
        raise FileExistsError(f"Preserve existing stage report: {output}")
    betas = {} if stage in ("smoke", "screen") else selected_betas(
        completed_stage(args, "screen"))
    names = screen_names() if not betas else selected_names(betas)
    trigger = []
    source = None
    if stage == "control":
        source = completed_stage(args, "confirm")
        means = {exp: source["statistics"]["comparisons"][exp+"-C"][
            "pooled"]["win_rate_difference"] for exp in ("E2", "E3")}
        better = max(means, key=means.get)
        trigger = [better] if means[better] > 0 else []
        names = [arm_name(exp+"-shuffled", betas[exp], seed)
                 for exp in trigger for seed in SEEDS]
    arms, opponents = definitions(args, names, stage == "sampling")
    pool = Path(args.pool)
    if json.loads((pool/"gen9ou/manifest.json").read_text())["role"] != "dev":
        raise ValueError("W5.3 selection requires the dev pool")
    claim_pool(pool, False, {})
    schedule = args.seed + {"smoke": 0, "screen": 10000,
                           "confirm": 20000, "control": 20000,
                           "sampling": 30000}[stage]
    if source is not None and schedule != source["schedule_seed"]:
        raise ValueError("Control must use the confirmation schedule seed")
    report = {"stage": stage, "valid": False, "arms": arms,
              "selected_betas": betas, "control_trigger": trigger,
              "schedule_seed": schedule, "battles_per_opponent": STAGES[stage],
              "pool": str(pool), "opponent_definitions": opponents,
              "opponents": {}, "fingerprints": {
                  "git_commit": git_commit("."), "source_sha256": source_fingerprint(),
                  "team_manifest_sha256": sha256(pool/"gen9ou/manifest.json"),
                  "checkpoints": {name: sha256(Path(arm["model_dir"])/"best.pt")
                                  for name, arm in arms.items()}},
              "methodology": {
                  "schedule": "paired teams, swapped directions, alternating challenger",
                  "bootstrap": "5000 paired team-pair resamples within opponent; seeds averaged on common schedule",
                  "seed_intervals": "conditional on fitted policies; seed means reported separately",
                  "ties": "non-wins", "selection": "dev only; no promotion",
                  "rng": "Showdown and heuristic RNG not seedable",
                  "severe_regression": "delta < -5pp and opponent paired CI upper < 0",
                  "smoke": "reliability only; no selection"}}
    if source:
        if opponents != source["opponent_definitions"]:
            raise ValueError("Control opponent configuration changed")
        if report["fingerprints"]["checkpoints"]["C"] != source[
                "fingerprints"]["checkpoints"]["C"]:
            raise ValueError("Control baseline changed since confirmation")
        if report["fingerprints"]["team_manifest_sha256"] != source[
                "fingerprints"]["team_manifest_sha256"]:
            raise ValueError("Control team pool changed since confirmation")
        report["arms"] = {**source["arms"], **arms}
        report["fingerprints"]["checkpoints"] = {
            **source["fingerprints"]["checkpoints"],
            **report["fingerprints"]["checkpoints"]}
    torch.set_num_threads(1)
    def checkpoint():
        report["wall_time_s"] = time.perf_counter()-started
        save(output, report)
    checkpoint()
    for index, (opponent_name, opponent) in enumerate(opponents.items()):
        results = report["opponents"][opponent_name] = (
            dict(source["opponents"][opponent_name]) if source else {})
        for name, arm in arms.items():
            if source and name == "C":
                continue
            seed = schedule+index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            summary = await run_battles(run_config(
                arm, opponent, pool=pool, battles=STAGES[stage],
                concurrent=args.concurrent, seed=seed))
            log = Path(summary["config"]["out"])/summary["run_id"]/"battles.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            result = arm_report(summary, rows, STAGES[stage])
            stats = summary.get("neural", {})
            result["model_illegal_by_side"] = {
                side: value["illegal_actions"] for side, value in stats.items()}
            result["valid"] = (result["valid"] and all(
                value["illegal_actions"] == 0 for value in stats.values())
                and ("model_dir" not in opponent or "opponent" in stats))
            result["battles_path"] = str(log)
            results[name] = result
            checkpoint()
            if not result["valid"]:
                raise RuntimeError(f"Invalid {stage}: {name} vs {opponent_name}")
            cluster_differences(rows, rows)
            if name != "C":
                cluster_differences(read_rows(results["C"]), rows)
            print(json.dumps({"stage": stage, "opponent": opponent_name,
                              "arm": name, "win_rate": result["win_rate"]}), flush=True)
    report["statistics"] = statistics(report)
    report["valid"] = True
    checkpoint()


def drift_flags(value):
    """Apply the pre-registered KL, entropy, and action-family drift gates."""
    flags = []
    for run, splits in value["runs"].items():
        for split, stats in splits.items():
            prefix = f"{run}/{split}"
            if stats["KL_C_to_actor"] > 0.5:
                flags.append(f"{prefix}: mean KL > 0.5 nats")
            reference_entropy = stats["reference_entropy"]
            entropy_change = abs(stats["entropy"] - reference_entropy)
            if entropy_change > 0.30 * abs(reference_entropy):
                flags.append(f"{prefix}: absolute entropy relative change > 30%")
            for mode in ("greedy",):
                current = stats[mode+"_action_frequency"]
                reference = stats["reference_"+mode+"_action_frequency"]
                for family in ("move", "switch", "tera"):
                    if abs(current[family] - reference[family]) > 0.10:
                        flags.append(f"{prefix}: {mode} {family} shift > 10pp")
    return flags


def finalize(args):
    started = time.perf_counter()
    stages = {stage: completed_stage(args, stage)
              for stage in ("smoke", "screen", "confirm", "sampling")}
    confirm = stages["confirm"]
    awbc_means = {exp: confirm["statistics"]["comparisons"][exp+"-C"][
        "pooled"]["win_rate_difference"] for exp in ("E2", "E3")}
    better = max(awbc_means, key=awbc_means.get)
    positive = [better] if awbc_means[better] > 0 else []
    if positive:
        stages["control"] = completed_stage(args, "control")
        if set(stages["control"]["control_trigger"]) != set(positive):
            raise ValueError("Control coverage does not match positive AWBC experiments")
    drift = load(args.drift)
    flags_by_experiment = {}
    for exp, beta in confirm["selected_betas"].items():
        selected_drift = {"runs": {
            arm_name(exp, beta, seed): {"val": drift["runs"][
                arm_name(exp, beta, seed)]["val"]} for seed in SEEDS}}
        flags_by_experiment[exp] = drift_flags(selected_drift)
    flags = flags_by_experiment[better]
    evidence = stages.get("control", confirm)
    comparisons = evidence["statistics"]["comparisons"]
    means = {exp: comparisons[exp+"-C"]["pooled"]["win_rate_difference"]
             for exp in ("E1", "E2", "E3")}
    winner = max(means, key=means.get)
    seed_means = comparisons[winner+"-C"]["seed_means"]
    median_seed = select_median_seed({
        int(name.rsplit("s", 1)[1]): delta for name, delta in seed_means.items()
    })
    selected = next(name for name in seed_means
                    if int(name.rsplit("s", 1)[1]) == median_seed)
    source = Path(evidence["arms"][selected]["model_dir"])
    target = Path(args.candidate)
    if target.exists():
        raise FileExistsError(f"Preserve existing candidate: {target}")
    output = Path(args.reports)/"w5_3_dev_ablation.json"
    final_report = Path(args.reports)/"w5_3_final_report.md"
    if output.exists() or final_report.exists():
        raise FileExistsError("Preserve existing combined/final reports")
    artifacts = ("best.pt", "config.yaml", "tokenizer.json")
    hashes = {name: sha256(source/name) for name in artifacts}
    if hashes["best.pt"] != evidence["fingerprints"]["checkpoints"][selected]:
        raise ValueError("Selected checkpoint changed since benchmark")
    training = yaml.safe_load((source/"config.yaml").read_text())
    provenance = training["provenance"]
    manifest = {"selected_arm": selected, "selected_experiment": winner,
                "source": str(source), "artifact_sha256": hashes,
                "selection": "highest confirmation mean among E1/E2/E3, median training seed by paired delta",
                "experiment_mean_deltas": means,
                "seed_mean_deltas": seed_means,
                "role": "candidate", "production": False,
                "dev_only": True, "production_promoted": False,
                "source_run": provenance["source_run"],
                "split_ids_sha256": provenance["split_ids_sha256"],
                "dataset_sha256": training["frozen_dataset_sha256"],
                "C_checkpoint_sha256": provenance["C_sha256"],
                "C_tokenizer_sha256": provenance["tokenizer_sha256"],
                "critic_sha256": (provenance["V_sha256"] if winner == "E2"
                                  else provenance["C_sha256"] if winner == "E3" else None),
                "beta": training["beta"], "seed": training["seed"],
                "hyperparameters": {key: training[key] for key in (
                    "epochs", "selected_epoch", "batch_size", "lr", "weight_decay",
                    "gradient_clip", "kl_coefficient", "maximum_weight", "optimizer",
                    "loss_normalization", "deterministic")},
                "git_commit": training["git_commit"], "git_dirty": training["git_dirty"],
                "evidence": {stage: {"path": str(stage_path(args, stage)),
                                     "sha256": sha256(stage_path(args, stage))}
                             for stage in stages},
                "training_config": training}
    target.mkdir(parents=True)
    for name in artifacts:
        shutil.copy2(source/name, target/name)
    save(target/"training_manifest.json", manifest)
    awbc = max(("E2", "E3"), key=means.get)
    main = comparisons[awbc+"-C"]
    stable = main["positive_seed_count"] >= 2
    severe = main["severe_regression_opponents"]
    if flags or severe:
        gate = "do_not_expand: policy drift or severe opponent regression"
    elif main["pooled"]["verdict"] == "negative":
        gate = "do_not_expand: clearly negative AWBC signal"
    elif means[awbc] > 0 and stable:
        gate = "expansion_eligible: positive, seed-stable exploratory signal; not production promotion"
    else:
        gate = "do_not_expand: no stable positive AWBC signal"
    decisions = {}
    for exp in ("E2", "E3"):
        result = comparisons[exp+"-C"]
        signal = (result["pooled"]["win_rate_difference"] > 0
                  and result["positive_seed_count"] >= 2
                  and not flags_by_experiment[exp])
        shuffled = comparisons.get(exp+"-shuffled")
        versus_e1 = comparisons[exp+"-E1"]["pooled"]
        weighting = (versus_e1["win_rate_difference"] > 0
                     and (shuffled is None or shuffled["pooled"][
                         "win_rate_difference"] > 0))
        supported = (versus_e1["bootstrap_95"][0] > 0
                     and (shuffled is None or shuffled["pooled"][
                         "bootstrap_95"][0] > 0))
        decisions[exp] = {
            "positive_signal": signal, "weighting_matters": weighting,
            "weighting_mean_positive": weighting,
            "weighting_supported": supported,
            "weighting_evidence": ("positive" if supported else
                                   "inconclusive" if weighting else "not_positive"),
            "paired_vs_E1": versus_e1,
            "paired_vs_shuffled": shuffled["pooled"] if shuffled else None,
            "severe_regression_opponents": result["severe_regression_opponents"],
            "control_evaluated": shuffled is not None,
            "selected_validation_drift_flags": flags_by_experiment[exp]}
    combined = {"valid": True, "stages": stages, "comparisons": comparisons,
                "sampling_comparisons": stages["sampling"]["statistics"]["comparisons"],
                "policy_drift": {"path": args.drift, "sha256": sha256(Path(args.drift)),
                                 "flags": flags, "diagnostics": drift},
                "candidate": manifest, "pilot_gate": gate,
                "experiment_decisions": decisions,
                "wall_time_s_by_stage": {k: v["wall_time_s"] for k, v in stages.items()},
                "production": "C unchanged; heldout untouched"}
    decision_lines = ["## Pre-registered experiment decisions", ""]
    for exp, decision in decisions.items():
        decision_lines.append(
            f"- {exp}: positive_signal={decision['positive_signal']}; "
            f"weighting_matters (mean)={decision['weighting_matters']}; "
            f"weighting supported by paired CI={decision['weighting_supported']}; "
            f"evidence={decision['weighting_evidence']}; "
            f"shuffled control evaluated={decision['control_evaluated']}.")
    decision_lines.append("")
    lines = ["# W5.3 AWBC pilot final report", "",
             "Dev-only exploratory evaluation. Production C is unchanged; heldout was not used.", "",
             "## Greedy confirmation (400 battles/opponent/arm; three training seeds)", "",
             "| Comparison | Pooled delta (pp) | Paired 95% CI (pp) | Verdict | Positive seeds | Severe regressions |",
             "|---|---:|---|---|---:|---|"]
    for name, result in comparisons.items():
        if name not in ("E1-C", "E2-C", "E3-C", "E2-E1", "E3-E1",
                        "E2-shuffled-C", "E3-shuffled-C", "E2-shuffled", "E3-shuffled"):
            continue
        pooled = result["pooled"]
        lo, hi = pooled["bootstrap_95"]
        lines.append(f"| {name} | {100*pooled['win_rate_difference']:+.2f} | "
                     f"[{100*lo:+.2f}, {100*hi:+.2f}] | {pooled['verdict']} | "
                     f"{result['positive_seed_count']}/{len(result['seed_means'])} | "
                     f"{', '.join(result['severe_regression_opponents']) or 'none'} |")
    lines += ["", "## Absolute greedy win rates (Wilson 95% intervals)", ""]
    for name, stats in confirm["statistics"]["arms"].items():
        pooled = stats["pooled"]
        lo, hi = pooled["win_rate_wilson_95"]
        lines.append(f"- {name}: {pooled['wins']}/{pooled['completed']} "
                     f"({100*pooled['win_rate']:.2f}%, "
                     f"[{100*lo:.2f}, {100*hi:.2f}]%).")
    lines += ["", "## Per-opponent and seed evidence", ""]
    for exp in ("E1", "E2", "E3"):
        result = comparisons[exp+"-C"]
        lines.append(f"### {exp} vs C")
        lines.append("Seed paired deltas (pp): " + ", ".join(
            f"{name}: {100*delta:+.2f}" for name, delta in result["seed_means"].items()))
        for opp, item in result["opponents"].items():
            lo, hi = item["bootstrap_95"]
            lines.append(f"- {opp}: {100*item['win_rate_difference']:+.2f} pp "
                         f"[{100*lo:+.2f}, {100*hi:+.2f}], {item['verdict']}.")
        lines.append("")
    lines += ["## Sampling at T=1.0 (200 battles/opponent/arm)", ""]
    for exp in ("E1", "E2", "E3"):
        item = stages["sampling"]["statistics"]["comparisons"][exp+"-C"]["pooled"]
        lo, hi = item["bootstrap_95"]
        lines.append(f"- {exp}-C: {100*item['win_rate_difference']:+.2f} pp "
                     f"[{100*lo:+.2f}, {100*hi:+.2f}], {item['verdict']}.")
    lines += ["", "## Training policy diagnostics", ""]
    for name in confirm["arms"]:
        if name == "C":
            continue
        for split, stats in drift["runs"][name].items():
            entropy = stats["entropy"]
            reference_entropy = stats["reference_entropy"]
            lines.append(f"- {name}/{split}: KL(C||actor) "
                         f"{stats['KL_C_to_actor']:.6f} nats; entropy "
                         f"{entropy:.6f} vs C {reference_entropy:.6f}; "
                         f"greedy move/switch/Tera {json.dumps(stats['greedy_action_frequency'])}; "
                         f"sampled move/switch/Tera {json.dumps(stats['sampled_action_frequency'])}.")
    lines += ["", "## Runtime diagnostics", ""]
    for stage, data in stages.items():
        errors = sum(result["error_count"] for rows in data["opponents"].values()
                     for result in rows.values())
        illegal = {side: sum(result["model_illegal_by_side"].get(side, 0)
                            for rows in data["opponents"].values()
                            for result in rows.values())
                   for side in ("player", "opponent")}
        retries = {side: sum(result["protocol_retries"].get(side, 0)
                            for rows in data["opponents"].values()
                            for result in rows.values())
                   for side in ("player", "opponent")}
        lines.append(f"- {stage}: runtime errors {errors}; model-illegal "
                     f"{json.dumps(illegal)}; protocol retries {json.dumps(retries)}. "
                     "Control-stage totals include reused confirmation records."
                     if stage == "control" else
                     f"- {stage}: runtime errors {errors}; model-illegal "
                     f"{json.dumps(illegal)}; protocol retries {json.dumps(retries)}.")
    validation = load(Path(args.reports)/"w5_3_dataset_validation.json")
    training = load(Path(args.reports)/"w5_3_training_metrics.json")
    advantage_stats = load(Path(args.reports)/"w5_3_advantage_stats.json")
    base_runs = {name: run for name, run in training["runs"].items()
                 if "-shuffled-" not in name}
    lines += ["", "## Frozen data and training", "",
              (f"C-only decisions: {validation['eligible_C_decisions']:,}; "
               f"re-encoding probability max difference "
               f"{validation['reference_probability_max_difference']:.1f}."),
              (f"Split hash: `{validation['provenance']['split_ids_sha256']}` "
               "(unchanged from W5.2; 45,080 train / 10,737 validation decisions)."),
              (f"Base actor runs: {len(base_runs)}, each exactly five epochs; "
               "final weights retained. Value heads were removed."),
              (f"Screen beta choices: {json.dumps(confirm['selected_betas'])}; "
               "highest pooled seed-1 dev delta, deterministic beta-order tie break."),
              ("W5.2 V was fitted on these same train decisions (a 385-parameter "
               "linear head; minimal but nonzero overfit); C's probe was fitted on human data.")]
    for exp, beta in confirm["selected_betas"].items():
        stats = advantage_stats[exp][beta]["all"]
        lines.append(f"- {exp}, beta={beta}: A mean "
                     f"{stats['advantage']['mean']:.6f}, std "
                     f"{stats['advantage']['std']:.6f}; normalized ESS "
                     f"{stats['ESS']:.6f}, clipped weight fraction "
                     f"{stats['clipped_fraction']:.6f}.")
    best_signal = any(item["positive_signal"] for item in decisions.values())
    lines += ["", "## Answer and limitations", "",
              ("The pilot provides a pre-registered AWBC positive signal, "
               "but does not establish that C is beaten."
               if best_signal else
               "The pilot does not establish that win/loss advantage weighting "
               "chooses better than C; the pre-registered positive-signal gate was not met."),
              "Stage-2 results are dev-selection evidence, not unbiased final estimates.",
              ("Limitations: noisy Monte Carlo advantages credit every decision in a "
               "winning battle; T=1 sampled collection differs from greedy production; "
               "only the existing 1,000-battle pilot was used; beta/config selection "
               "on dev creates selection bias; W5.2 V was fitted on the same data."),
              (f"Actor sequences retained "
               f"{validation['provenance']['fallback_steps_in_actor_sequences']} "
               "non-model observations without loss, as required. Frozen critic/reference "
               "features preserve serving parity, which skips fallback memory updates; "
               "this is an additional train/serve memory caveat."),
              ("No new data, heldout pool, W3 test, or W4 test was used. "
               "C remains production; promotion is reserved for W5.4/W5.5.")]
    lines += ["", "## Reliability, drift, and decision", "",
              "All completed stages require zero runtime errors and zero model-illegal actions on both neural sides.",
              f"Training policy-drift flags: {json.dumps(flags)}.",
              f"Pilot gate: **{gate}**.",
              f"Candidate: **{selected}**, copied to `{target}` with artifact hashes and selection manifest.",
              "Candidate selection is not evidence of production superiority. A CI containing zero is inconclusive.",
              ("Intervals use paired team-pair bootstrap within opponent, averaging the shared schedule across seeds; "
               "they condition on fitted policies. Wilson intervals and full seed/opponent results are in the JSON."),
              "Severe regression: strictly more than 5 pp loss with opponent CI upper bound below zero. Comparisons are exploratory, without multiplicity correction.",
              "", "## Wall time", ""]
    for stage, data in stages.items():
        lines.append(f"- {stage}: {data['wall_time_s']:.1f} s.")
    combined["finalization_wall_time_s"] = time.perf_counter()-started
    save(output, combined)
    lines += ["", *decision_lines]
    final_report.write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(*STAGES, "finalize"))
    parser.add_argument("--config", default="configs/w5_benchmark.yaml")
    parser.add_argument("--pool", default="data/teams_w5/dev")
    parser.add_argument("--checkpoints", default="checkpoints/w5_3")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--drift", default="reports/w5_3_policy_drift.json")
    parser.add_argument("--candidate", default="models/pokedecision-v2-candidate")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--concurrent", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    if args.stage == "finalize":
        finalize(args)
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
