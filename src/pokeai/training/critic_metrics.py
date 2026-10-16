"""Offline W5.2 critic diagnostics; never interpret unchosen Q as ground truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pokeai.training.value_metrics import value_metrics

SEED = 20261016
RESAMPLES = 5000
PHASES = (("0-5", 0, 6), ("6-15", 6, 16), ("16-30", 16, 31), ("31+", 31, None))


def _bootstrap(values):
    # The scripts directory is not installed as a package. Import the original
    # benchmark helper, including its sibling imports, rather than copying it.
    scripts = str(Path(__file__).resolve().parents[3] / "scripts")
    added = scripts not in sys.path
    if added:
        sys.path.insert(0, scripts)
    try:
        from evaluate_w4_5 import bootstrap

        result = bootstrap(np.asarray(values), seed=SEED, resamples=RESAMPLES)
    finally:
        if added:
            sys.path.remove(scripts)
    return result


def _weights(battle):
    _, inverse, counts = np.unique(battle, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _weighted_metrics(p, y, battle):
    """Every battle contributes total weight one, not one averaged prediction."""
    result = value_metrics(p, y)
    if not len(y):
        result["battles"] = 0
        return result
    weights = _weights(battle)
    weights /= weights.sum()
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for index, entry in enumerate(result["reliability"]):
        selected = bins == index
        mass = float(weights[selected].sum())
        entry["weight"] = mass
        entry["mean_probability"] = (
            float(np.dot(weights[selected], p[selected]) / mass) if mass else None
        )
        entry["observed_rate"] = (
            float(np.dot(weights[selected], y[selected]) / mass) if mass else None
        )
        ece += abs(float(np.dot(weights[selected], p[selected] - y[selected])))
    _, inverse = np.unique(p, return_inverse=True)
    positive = np.bincount(inverse, weights=weights * y)
    negative = np.bincount(inverse, weights=weights * (1 - y))
    denominator = positive.sum() * negative.sum()
    auc = (
        float(np.dot(positive, np.cumsum(negative) - negative / 2) / denominator)
        if denominator
        else None
    )
    result.update(
        battles=len(np.unique(battle)),
        brier=float(np.dot(weights, (p - y) ** 2)),
        bce=float(-np.dot(weights, y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        roc_auc=auc,
        ece=ece,
        accuracy=float(np.dot(weights, (p >= 0.5) == y)),
        mean_probability=float(np.dot(weights, p)),
        base_rate=float(np.dot(weights, y)),
    )
    return result


def _distribution(values, weights):
    if not len(values):
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10_p50_p90": None,
        }
    weights = weights / weights.sum()
    mean = float(np.dot(weights, values))
    order = np.argsort(values)
    quantiles = np.interp([0.1, 0.5, 0.9], np.cumsum(weights[order]), values[order])
    return {
        "mean": mean,
        "std": float(np.sqrt(np.dot(weights, (values - mean) ** 2))),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10_p50_p90": quantiles.tolist(),
    }


def _both(p, y, battle):
    result = {
        "decision_weighted": value_metrics(p, y),
        "battle_equal": _weighted_metrics(p, y, battle),
    }
    for name, weights in (
        ("decision_weighted", np.ones(len(p))),
        ("battle_equal", _weights(battle)),
    ):
        result[name]["prediction_distribution"] = _distribution(p, weights)
        result[name]["prediction_std"] = result[name]["prediction_distribution"]["std"]
        result[name]["mean_absolute_distance_from_0.5"] = (
            float(np.dot(weights, abs(p - 0.5)) / weights.sum()) if len(p) else None
        )
    return result


def _breakdown(p, data, selected):
    y, battle = data["y"], data["battle"]
    result = _both(p[selected], y[selected], battle[selected])
    groups = {"phase": {}, "opponent": {}}
    for name, lower, upper in PHASES:
        mask = selected & (data["phase_index"] >= lower)
        if upper is not None:
            mask &= data["phase_index"] < upper
        groups["phase"][name] = _both(p[mask], y[mask], battle[mask])
    for opponent in np.unique(data["opponent"]):
        mask = selected & (data["opponent"] == opponent)
        groups["opponent"][str(opponent)] = _both(p[mask], y[mask], battle[mask])
    for name, key in (("first", "first"), ("last", "last")):
        mask = selected & data[key]
        groups[name] = _both(p[mask], y[mask], battle[mask])
    result["breakdowns"] = groups
    return result


def _pair_sums(values, pairs):
    _, inverse = np.unique(pairs, return_inverse=True)
    return np.stack(
        [np.bincount(inverse, weights=column) for column in np.asarray(values).T],
        axis=1,
    )


def _nonlinear_bootstrap(sufficient, statistic):
    """Cluster-resample sufficient statistics for ratios and correlations.

    The W4.5 scalar-mean bootstrap cannot evaluate nonlinear statistics.
    This path avoids treating decisions as independent and keeps memory bounded.
    """
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(RESAMPLES):
        sampled = sufficient[rng.integers(len(sufficient), size=len(sufficient))]
        value = statistic(sampled.sum(axis=0))
        if value is not None:
            estimates.append(value)
    return (
        np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else None
    ), len(estimates)


def _delta(left, right, y, battle, pairs):
    difference = (left - y) ** 2 - (right - y) ** 2
    weights = _weights(battle)
    sufficient = _pair_sums(np.column_stack((weights * difference, weights)), pairs)
    point = float(sufficient[:, 0].sum() / sufficient[:, 1].sum())
    if np.allclose(sufficient[:, 1], sufficient[0, 1]):
        raw = _bootstrap(sufficient[:, 0] / sufficient[:, 1])
        interval = raw["bootstrap_95"]
    else:
        interval, _ = _nonlinear_bootstrap(sufficient, lambda x: x[0] / x[1])
    return {
        "battle_equal_delta": point,
        "decision_weighted_delta": float(difference.mean()),
        "bootstrap_95": interval,
        "passes": bool(interval[1] < 0),
        "verdict": "pass" if interval[1] < 0 else "not established",
        "clusters": len(sufficient),
        "cluster_unit": "team pair",
        "seed": SEED,
        "resamples": RESAMPLES,
    }


def _correlation(advantage, residual, battle, pair):
    weights = _weights(battle)
    sufficient = _pair_sums(
        np.column_stack(
            (
                weights,
                weights * advantage,
                weights * residual,
                weights * advantage**2,
                weights * residual**2,
                weights * advantage * residual,
            )
        ),
        pair,
    )

    def statistic(sums):
        n, x, y, xx, yy, xy = sums
        vx, vy = xx - x * x / n, yy - y * y / n
        if vx <= 0 or vy <= 0:
            return None
        return float(np.clip((xy - x * y / n) / np.sqrt(vx * vy), -1, 1))

    point = statistic(sufficient.sum(axis=0))
    interval, valid = _nonlinear_bootstrap(sufficient, statistic)
    return {
        "definition": "Pearson corr(Q(s,a_chosen)-V(s), outcome-V(s))",
        "weighting": "battle_equal",
        "correlation": point,
        "bootstrap_95": interval,
        "defined_resamples": valid,
        "resamples": RESAMPLES,
        "seed": SEED,
        "clusters": len(sufficient),
        "cluster_unit": "team pair",
    }


def _coverage(data):
    result = {
        "sparse_threshold": 500,
        "phase_definition": "zero-based decision index, not simulator turn",
        "splits": {},
    }
    for split in ("train", "val"):
        selected = data["split"] == split
        counts = np.bincount(data["actions"][selected], minlength=13)
        probabilities = data["probs"][selected]
        terms = np.zeros_like(probabilities, dtype=np.float64)
        positive = probabilities > 0
        terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
        entropy = terms.sum(axis=1)
        legal = data["masks"][selected].sum(axis=0)
        unchosen = legal - counts
        grid = {}
        for opponent in np.unique(data["opponent"]):
            grid[str(opponent)] = {}
            for name, lower, upper in PHASES:
                mask = selected & (data["opponent"] == opponent)
                mask &= data["phase_index"] >= lower
                if upper is not None:
                    mask &= data["phase_index"] < upper
                chosen = np.bincount(data["actions"][mask], minlength=13)
                legal_counts = data["masks"][mask].sum(axis=0)
                grid[str(opponent)][name] = {
                    "count": int(mask.sum()),
                    "chosen_counts": chosen.tolist(),
                    "legal_counts": legal_counts.tolist(),
                    "legal_unchosen_counts": (legal_counts - chosen).tolist(),
                    "legal_unchosen_rate_per_slot": [
                        float((total - observed) / total) if total else None
                        for total, observed in zip(legal_counts, chosen)
                    ],
                    "sparse_slots": np.flatnonzero(chosen < 500).tolist(),
                }
        result["splits"][split] = {
            "decisions": int(selected.sum()),
            "chosen_counts": counts.tolist(),
            "sparse_slots": np.flatnonzero(counts < 500).tolist(),
            "legal_counts": legal.tolist(),
            "legal_unchosen_counts": unchosen.tolist(),
            "legal_unchosen_rate_per_slot": [
                float(unobserved / total) if total else None
                for unobserved, total in zip(unchosen, legal)
            ],
            "never_supervised_slots": np.flatnonzero(counts == 0).tolist(),
            "legal_unchosen_fraction": (
                float(unchosen.sum() / legal.sum()) if legal.sum() else None
            ),
            "behavior_entropy_nats": {
                "mean": float(entropy.mean()) if len(entropy) else None,
                "p10_p50_p90": (
                    np.quantile(entropy, [0.1, 0.5, 0.9]).tolist()
                    if len(entropy)
                    else None
                ),
            },
            "slot_x_opponent_x_phase": grid,
        }
    result["caveat"] = (
        "Counts refer to action slots, whose semantic moves/switches vary by state. "
        "Legal-but-unchosen actions have no observed counterfactual outcome. "
        "The 500-sample threshold is descriptive, not a power guarantee."
    )
    return result


def evaluate(data, predictions, metadata, out_dir) -> dict:
    """Write the four W5.2 reports using the frozen outer train/val split.

    ``q`` and ``shuffled`` are probabilities [N,13]; shuffled is the separately
    trained action-shuffled control, not an inference-time permutation of Q.
    No checkpoint selection or actor update occurs here.
    """
    data = {key: np.asarray(value) for key, value in data.items()}
    for key in ("masks", "first", "last", "inner_val"):
        data[key] = np.asarray(data[key], dtype=bool)
    y, actions = data["y"], data["actions"]
    v = np.asarray(predictions["v"], dtype=np.float64)
    q = np.asarray(predictions["q"], dtype=np.float64)[np.arange(len(y)), actions]
    shuffled = np.asarray(predictions["shuffled"], dtype=np.float64)[
        np.arange(len(y)), actions
    ]
    train, val = data["split"] == "train", data["split"] == "val"
    if not train.any() or not val.any():
        raise ValueError("critic evaluation requires nonempty train and val splits")
    train_rate = float(y[train].mean())
    references = {
        "train_win_rate": np.full(len(y), train_rate),
        "constant_0.5": np.full(len(y), 0.5),
        "logged_probe": np.asarray(data["probe"], dtype=np.float64),
    }
    provenance = {
        key: metadata.get(key)
        for key in (
            "source_run",
            "c_hashes",
            "counts",
            "splits",
            "feature_alignment_max_diff",
            "qa",
            "team_overlap",
        )
    }
    common = {
        "provenance": provenance,
        "train_win_rate": train_rate,
        "seed": SEED,
        "training_seeds": 1,
        "pilot_battles": 1000,
        "train_win_rate_definition": "mean binary outcome over train decisions",
        "phase_definition": "zero-based decision index, not simulator turn",
        "weighting": "decision_weighted and equal total weight per battle",
    }
    value_report = {**common, "splits": {}}
    q_report = {
        **common,
        "splits": {},
        "caveats": [
            "Only chosen actions are supervised and evaluated against an outcome.",
            "Chosen-action gains and shuffled-control gains do not establish alternative-action ranking.",
            "Q slot dispersion is descriptive only; slots have state-dependent semantics.",
            "Collection sampled C at temperature 1.0; this is not the greedy C value function.",
        ],
    }
    for split, selected in (("train", train), ("val", val)):
        value_report["splits"][split] = {
            "v": _breakdown(v, data, selected),
            "baselines": {
                name: _breakdown(p, data, selected) for name, p in references.items()
            },
        }
        q_report["splits"][split] = {
            "q_chosen": _breakdown(q, data, selected),
            "v": _breakdown(v, data, selected),
            "shuffled_q_chosen": _breakdown(shuffled, data, selected),
            "baselines": {
                name: _breakdown(p, data, selected) for name, p in references.items()
            },
        }
    arguments = (y[val], data["battle"][val], data["pair"][val])
    gates = {
        "G1": _delta(v[val], references["train_win_rate"][val], *arguments),
        "G2": _delta(q[val], v[val], *arguments),
        "G3": _delta(q[val], shuffled[val], *arguments),
    }
    for key, definition in (
        ("G1", "Brier(V) - Brier(train win rate)"),
        ("G2", "Brier(Q chosen) - Brier(V)"),
        ("G3", "Brier(Q chosen) - Brier(action-shuffled Q chosen)"),
    ):
        gates[key]["definition"] = definition
    value_report["gates"] = {"G1": gates["G1"]}
    q_report["gates"] = gates
    q_report["advantage_residual_correlation"] = _correlation(
        q[val] - v[val], y[val] - v[val], data["battle"][val], data["pair"][val]
    )
    all_q = np.asarray(predictions["q"], dtype=np.float64)[val]
    mask = data["masks"][val]
    legal_count = mask.sum(axis=1)
    legal_mean = np.where(mask, all_q, 0).sum(axis=1) / legal_count
    legal_sd = np.sqrt(
        np.where(mask, (all_q - legal_mean[:, None]) ** 2, 0).sum(axis=1) / legal_count
    )
    q_report["descriptive_legal_q_dispersion"] = {
        "mean_within_state_std": float(legal_sd.mean()),
        "p10_p50_p90": np.quantile(legal_sd, [0.1, 0.5, 0.9]).tolist(),
        "not_evidence_of_counterfactual_accuracy": True,
    }
    advantage = q[val] - v[val]
    q_report["advantage_distribution"] = {
        "decision_weighted": _distribution(advantage, np.ones(len(advantage))),
        "battle_equal": _distribution(advantage, _weights(data["battle"][val])),
    }
    q_best = np.where(mask, all_q, -np.inf).argmax(axis=1)
    c_best = np.where(mask, data["probs"][val], -np.inf).argmax(axis=1)
    unobserved = mask.copy()
    unobserved[np.arange(len(all_q)), actions[val]] = False
    train_counts = np.bincount(actions[train], minlength=13)
    q_report["descriptive_action_diagnostics"] = {
        "argmax_legal_q_differs_from_chosen_rate": float(
            np.mean(q_best != actions[val])
        ),
        "argmax_legal_q_differs_from_c_greedy_rate": float(np.mean(q_best != c_best)),
        "observed_chosen_slot_state_count": len(all_q),
        "legal_unobserved_slot_state_count": int(unobserved.sum()),
        "train_supervision_count_per_slot": train_counts.tolist(),
        "val_legal_count_per_slot": mask.sum(axis=0).tolist(),
        "val_legal_unobserved_count_per_slot": unobserved.sum(axis=0).tolist(),
        "chosen_q_distribution": _distribution(q[val], np.ones(int(val.sum()))),
        "legal_unobserved_q_distribution": _distribution(
            all_q[unobserved], np.ones(int(unobserved.sum()))
        ),
        "mean_chosen_minus_legal_unobserved_q": (
            float(
                np.mean(
                    q[val][unobserved.sum(axis=1) > 0]
                    - (
                        np.where(unobserved, all_q, 0).sum(axis=1)
                        / np.maximum(unobserved.sum(axis=1), 1)
                    )[unobserved.sum(axis=1) > 0]
                )
            )
            if unobserved.any()
            else None
        ),
        "caveat": "Each state has one observed slot; all other legal Q predictions "
        "are unvalidated counterfactuals. Argmax differences are descriptive.",
    }
    coverage = _coverage(data)
    qa = metadata.get("qa", {})
    qa_clean = isinstance(qa, dict) and qa.get("overall_status") == "PASS"
    if not qa_clean:
        verdict = "STOP: repair data QA before collecting more battles."
    elif all(gate["passes"] for gate in gates.values()):
        verdict = "Scale to 10,000 battles: QA and all three pilot gates pass."
    elif gates["G1"]["passes"]:
        rare = coverage["splits"]["train"]["sparse_slots"]
        verdict = (
            "Targeted scale-up only; V learns but action signal is not established. "
            f"Prioritize train slots below 500 samples: {rare} "
            "(slots 9-12 are tera-moves); do not broadly scale to 10,000 yet."
        )
    else:
        verdict = "Inspect labels, rewards and critic architecture before scaling."
    report = [
        "# W5.2 pilot critic report",
        "",
        verdict,
        "",
        "## Gates",
        "",
        (
            "Primary: battle-equal validation Brier differences; "
            "negative is better. A pass requires the entire 95% CI below zero. "
            "5,000 seeded bootstrap resamples cluster by team pair."
        ),
        "",
        "| Gate | Difference | Estimate | 95% CI | Verdict |",
        "|---|---|---:|---|---|",
    ]
    for name, gate in gates.items():
        low, high = gate["bootstrap_95"]
        report.append(
            f"| {name} | {gate['definition']} | "
            f"{gate['battle_equal_delta']:.6f} | [{low:.6f}, {high:.6f}] | "
            f"{gate['verdict']} |"
        )
    report.extend(
        [
            "",
            "## Validation summary",
            "",
            "| Predictor | Battle-equal Brier | BCE | ECE | AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    summary = {
        "V": value_report["splits"]["val"]["v"],
        "Q chosen": q_report["splits"]["val"]["q_chosen"],
        **value_report["splits"]["val"]["baselines"],
    }
    for name, metrics in summary.items():
        row = metrics["battle_equal"]
        cells = [
            "undefined" if row[key] is None else f"{row[key]:.6f}"
            for key in ("brier", "bce", "ece", "roc_auc")
        ]
        report.append(f"| {name} | " + " | ".join(cells) + " |")
    correlation = q_report["advantage_residual_correlation"]
    report.extend(
        [
            "",
            (
                "Advantage/residual correlation: "
                f"{correlation['correlation']}; 95% CI "
                f"{correlation['bootstrap_95']}."
            ),
        ]
    )
    for split, details in coverage["splits"].items():
        report.append(
            f"- {split}: {details['decisions']} decisions; slots below "
            f"500 samples: {details['sparse_slots']}; mean behavior "
            f"entropy {details['behavior_entropy_nats']['mean']} nats."
        )
    report.extend(
        [
            "",
            "## Answers to the pilot questions",
            "",
            "1. Does V learn outcome signal on unseen battles? "
            + (
                "Yes under G1."
                if gates["G1"]["passes"]
                else "Not established by the pre-registered G1 confidence interval."
            ),
            "2. Does Q add action-dependent signal beyond V? "
            + (
                "G2 and G3 both pass for observed actions only."
                if gates["G2"]["passes"] and gates["G3"]["passes"]
                else "Not established: both G2 and G3 must pass."
            ),
            "3. Is coverage and consistency sufficient to scale? " + verdict,
            "",
            "This is one training seed on the 1,000-battle pilot. "
            "Team reuse can limit generalization despite team-pair-disjoint splits; "
            "individual teams may occur in multiple pairs. Team-overlap metadata: "
            + json.dumps(metadata.get("team_overlap", "not supplied"))
            + ".",
        ]
    )
    report.extend(
        [
            "",
            "## Evidence and limitations",
            "",
            (
                f"- Dataset QA: {'PASS' if qa_clean else 'NOT PASS'}. "
                f"Train-only decision win-rate reference: {train_rate:.6f}."
            ),
            (
                "- Feature alignment max difference: "
                f"{metadata['feature_alignment_max_diff']} on "
                f"{metadata['feature_alignment_decisions']} C decisions. "
                f"POV/outcome checks: {metadata['pov_outcome_checks']} across "
                f"{metadata['two_pov_battles_checked']} two-POV battles; "
                f"ties excluded: {metadata['ties_battles']} battles."
            ),
            (
                "- JSON reports include BCE, Brier, AUC, ECE, reliability bins, "
                "first/last decisions, phase and opponent breakdowns under both weightings."
            ),
            (
                "- Advantage–outcome-residual correlation has a team-pair clustered CI; "
                "it is diagnostic, not proof that changing the policy improves win rate."
            ),
            (
                "- Coverage enumerates every slot × opponent × phase cell, sparse "
                "slots (<500), behavior entropy and legal-but-unchosen actions."
            ),
            *[f"- {text}" for text in q_report["caveats"]],
            (
                "- No actor update, new battles, dev-team selection or heldout evaluation "
                "is performed. W5.3 must test actual policy improvement in the simulator."
            ),
            (
                "- These gates share the pilot validation set and are not independent "
                "replications. A CI containing zero is inconclusive, not evidence of no effect."
            ),
            "",
        ]
    )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "w5_2_value_metrics.json": value_report,
        "w5_2_q_metrics.json": q_report,
        "w5_2_action_coverage.json": coverage,
    }
    for name, content in artifacts.items():
        (output / name).write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n"
        )
    (output / "w5_2_pilot_report.md").write_text("\n".join(report))
    return {
        "value": value_report,
        "q": q_report,
        "coverage": coverage,
        "gates": gates,
        "qa_clean": qa_clean,
        "verdict": verdict,
    }
