"""W5 Experiment A2.3: GAE(gamma=1, lambda) advantages for AWBC. Offline only.

Pre-registered (user plan A2.3). Only the advantage estimator changes vs A2.1:
- Critic: frozen C probe V. Reward 0 per step, 1/0 at the terminal step.
- lambda 0.95 (primary), 0.90 (sensitivity), 1.00 (Monte Carlo reference, which
  must reproduce Y - V exactly). Fallback steps stay in the GAE chain (their V
  comes from C's value head on the stored actor sequence) but get no actor loss.
- Actor/training identical to A2.1 (batch 64, lr 3e-4, lambda_KL 0.1, beta 2,
  w_max 5, max 30 epochs, seeds 1/2/3, checkpoints at first epoch with val
  KL >= 0.05 / 0.10). A2.1's E1, E3 (= MC lambda 1) and E3-shuffled runs are
  reused as references, not retrained.
- Shuffled control: battle-level outcome permutation inside (split, opponent)
  groups (as A2.1); TD residuals V(s_t+1) - V(s_t) are untouched, so the
  control removes only the outcome label's contribution.
- Evaluation advantage FIXED for every model: A_eval = Y - V_probe;
  S_MC(theta) = E_val[A_eval * (log pi_theta(a_t) - log pi_C(a_t))], 5,000-resample
  bootstrap clustered by team pair, seed-averaged Delta per decision.
- Gradient alignment per lambda: cos(g^A_train, g^A_val) at theta = C for the
  real outcome vs 100 battle-level permutation nulls (per-batch normalisation;
  A2.1 showed global normalisation is equivalent, cos 0.98).
- Gate (KL 0.05): S_MC(GAE0.95) - S_MC(E1) > 0 and S_MC(GAE0.95) -
  S_MC(GAE0.95-shuffled) > 0, both CIs excluding 0; GAE0.95 real cosine above the
  null 95th percentile. Otherwise A2.3 is negative and no battles are run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w5_a2_offline as a2

from pokeai.models.pokedecision import load_pokedecision
from pokeai.training.awbc import (
    AdvantageSequences,
    awbc_loss,
    awbc_weights,
    make_arrays,
)
from pokeai.training.gae import gae_advantages
from pokeai.training.train_bc import pad_sequences

OUT = Path("checkpoints/w5_a2_3")
REPORTS = Path("reports")
LAMBDAS = (1.0, 0.95, 0.90)
NULL_SEEDS = range(100, 200)


def write(name: str, value) -> None:
    (REPORTS / name).write_text(json.dumps(value, indent=2) + "\n")


@torch.no_grad()
def fallback_values(frozen, meta) -> tuple[dict[int, np.ndarray], float]:
    """C value head over each stored actor sequence; used only at fallback steps.

    Also returns the max |sigmoid(V) - logged probe| over sequences without any
    fallback step, proving this path equals the serving-matched critic there.
    """
    model, _, _ = load_pokedecision(a2.C_DIR, "cuda")
    model.eval()
    values, parity = {}, 0.0
    for index, seq in enumerate(frozen["sequences"]):
        tokens = torch.as_tensor(seq["tokens"]).long().unsqueeze(0).cuda()
        numbers = torch.as_tensor(seq["numbers"]).float().unsqueeze(0).cuda()
        _, value, _ = model.forward_with_value(tokens, numbers)
        v = torch.sigmoid(value[0]).double().cpu().numpy()
        values[index] = v
        rows = seq["row"]
        if (rows >= 0).all():
            parity = max(parity, float(np.abs(v - meta["probe"][rows]).max()))
    return values, parity


def advantages(frozen, meta, fb: dict[int, np.ndarray], outcomes: np.ndarray,
               lam: float) -> np.ndarray:
    result = np.full(len(outcomes), np.nan)
    for index, seq in enumerate(frozen["sequences"]):
        rows = seq["row"]
        decided = rows >= 0
        if not decided.any():
            continue
        values = np.where(decided, meta["probe"][rows.clip(min=0)], fb[index])
        outcome = outcomes[rows[decided]]
        if not (outcome == outcome[0]).all():
            raise ValueError(f"trajectory {index} mixes outcomes")
        result[rows[decided]] = gae_advantages(values, float(outcome[0]), lam)[decided]
    if np.isnan(result).any():
        raise ValueError("some C decisions received no advantage")
    return result


def validation(frozen, meta, fb, parity) -> dict:
    mc = meta["y"] - meta["probe"]
    shuffled = a2.shuffled_outcomes(meta, 1)
    seqs = frozen["sequences"]
    ends_fallback = sum(int(s["row"][-1] < 0) for s in seqs)
    report = {
        "lambda1_real_max_abs_diff_vs_MC": float(
            np.abs(advantages(frozen, meta, fb, meta["y"], 1.0) - mc).max()),
        "lambda1_shuffled_max_abs_diff_vs_MC": float(np.abs(
            advantages(frozen, meta, fb, shuffled, 1.0) - (shuffled - meta["probe"])).max()),
        "value_path_parity_max_abs_diff": parity,
        "trajectories": len(seqs),
        "fallback_steps_in_chain": int(sum(int((s["row"] < 0).sum()) for s in seqs)),
        "trajectories_ending_in_fallback_step": ends_fallback,
        "terminal_rule": "r_T = outcome at the last stored step, next value 0",
        "provenance": {k: frozen["provenance"][k] for k in
                       ("C_sha256", "feature_cache_sha256", "split_ids_sha256")},
        "pov_outcome_checks": "inherited from A2.1 load_inputs (records == W5.2 cache)",
    }
    report["pass"] = (report["lambda1_real_max_abs_diff_vs_MC"] < 1e-9
                      and report["lambda1_shuffled_max_abs_diff_vs_MC"] < 1e-9
                      # float32 GRU drift (batch-1 vs cached batch): median 7e-5,
                      # grows with length; a semantic mismatch would be ~1e-2.
                      and parity < 1e-3)
    return report


def advantage_stats(meta, adv: dict[str, np.ndarray]) -> dict:
    mc = adv["lambda_1.00"]
    phases = np.array(sorted(set(meta["phase"])))
    out = {}
    for name, a in adv.items():
        w = awbc_weights(torch.as_tensor(a), a2.BETA).numpy()
        entry = {"mean": float(a.mean()), "var": float(a.var()), "std": float(a.std()),
                 "corr_with_MC": float(np.corrcoef(a, mc)[0, 1]),
                 "sign_flip_rate_vs_MC": float((np.sign(a) != np.sign(mc)).mean()),
                 "ESS": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                 "clipped_fraction": float((a / a2.BETA >= np.log(5)).mean()),
                 "by_phase": {}}
        for p in phases:
            m = meta["phase"] == p
            entry["by_phase"][str(p)] = {
                "n": int(m.sum()), "mean": float(a[m].mean()), "std": float(a[m].std()),
                "corr_with_MC": float(np.corrcoef(a[m], mc[m])[0, 1]),
                "sign_flip_rate_vs_MC": float((np.sign(a[m]) != np.sign(mc[m])).mean())}
        out[name] = entry
    return out


def matched_kl(frozen, meta, refs: dict[str, list[dict]]) -> dict:
    rows = a2.val_rows(frozen)
    c_logp, c_greedy, forced = a2.chosen_logp(a2.fresh_actor(), frozen)
    if len(c_logp) != len(rows):
        raise ValueError("eligible val steps do not align with C decision records")
    a_eval = (meta["y"] - meta["probe"])[rows]
    pairs = meta["pair"][rows]
    report = {"evaluation_advantage": "A_eval = Y - V_probe (fixed for every model)"}
    for target in a2.TARGETS:
        deltas, per_run = defaultdict(list), {}
        for variant, runs in refs.items():
            for run in sorted(runs, key=lambda r: r["seed"]):
                hit = run["matched"].get(str(target)) or run["matched"].get(target)
                if hit is None:
                    continue
                model, _, _ = load_pokedecision(hit["dir"], "cuda")
                logp, greedy, _ = a2.chosen_logp(model, frozen)
                delta = logp - c_logp
                deltas[variant].append(delta)
                per_run[f"{variant}-s{run['seed']}"] = {
                    "epoch": hit["epoch"], "val_KL": hit["val_KL"],
                    "S_MC": float(np.mean(a_eval * delta)),
                    "greedy_flip_rate_nonforced": float((greedy != c_greedy)[~forced].mean())}
        complete = all(len(deltas[v]) == len(a2.SEEDS) for v in refs)
        entry = {"runs": per_run, "all_matched": complete}
        if complete:
            mean = {v: np.mean(d, axis=0) for v, d in deltas.items()}
            entry["S_MC_seed_mean"] = {v: float(np.mean(a_eval * m)) for v, m in mean.items()}
            contrasts = [("GAE0.95", "E1"), ("GAE0.95", "GAE0.95-shuffled"),
                         ("GAE0.90", "E1"), ("GAE0.90", "GAE0.95-shuffled"),
                         ("MC", "E1"), ("MC", "MC-shuffled"), ("GAE0.95", "MC")]
            entry["contrasts"] = {
                f"{a}_minus_{b}": a2.cluster_ci(a_eval * (mean[a] - mean[b]), pairs)
                for a, b in contrasts}
            entry["per_seed"] = {
                f"s{s}": {f"{a}_minus_{b}": a2.cluster_ci(
                    a_eval * (deltas[a][i] - deltas[b][i]), pairs)
                    for a, b in contrasts[:2]}
                for i, s in enumerate(a2.SEEDS)}
        report[f"kl_{target:.2f}"] = entry
    return report


def per_batch_grad(model, frozen, weights, split) -> torch.Tensor:
    loader = DataLoader(AdvantageSequences(make_arrays(frozen, split, weights)),
                        batch_size=a2.BATCH, shuffle=False, collate_fn=pad_sequences)
    params = [p for p in model.parameters() if p.requires_grad]
    total, n = None, 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        logits = model(batch["tokens"].long(), batch["numbers"])[0]
        _, bc, _ = awbc_loss(logits, batch["actions"], batch["masks"], batch["valid"],
                             batch["weights"], batch["reference"])
        g = a2.flat(torch.autograd.grad(bc, params, allow_unused=True)).cpu()
        total = g if total is None else total + g
        n += 1
    return total / n


def gradient_alignment(frozen, meta, fb) -> dict:
    model = a2.fresh_actor()
    model.train(True)
    ones = np.ones(len(meta["y"]), np.float32)
    base = {s: per_batch_grad(model, frozen, ones, s) for s in ("train", "val")}
    cos = torch.nn.functional.cosine_similarity

    def cosine(outcomes, lam):
        w = awbc_weights(torch.as_tensor(advantages(frozen, meta, fb, outcomes, lam)),
                         a2.BETA).numpy()
        comp = {s: per_batch_grad(model, frozen, w, s) - base[s] for s in ("train", "val")}
        return float(cos(comp["train"], comp["val"], dim=0))

    nulls = {s: a2.shuffled_outcomes(meta, s) for s in NULL_SEEDS}
    report = {"null_permutations": len(nulls), "normalisation": "per_batch"}
    for lam in LAMBDAS:
        real = cosine(meta["y"], lam)
        values = np.array([cosine(nulls[s], lam) for s in NULL_SEEDS])
        p95 = float(np.quantile(values, 0.95))
        report[f"lambda_{lam:.2f}"] = {
            "real_cos_train_val": real, "null_mean": float(values.mean()),
            "null_std": float(values.std()), "null_p95": p95,
            "fraction_null_at_or_above_real": float((values >= real).mean()),
            "real_above_p95": bool(real > p95)}
    return report


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(4)
    frozen, meta = a2.load_inputs()
    meta["phase"] = np.array([r["phase"] for r in frozen["records"]])
    fb, parity = fallback_values(frozen, meta)

    gae_validation = validation(frozen, meta, fb, parity)
    write("w5_a2_3_gae_validation.json", gae_validation)
    if not gae_validation["pass"]:
        raise SystemExit("GAE validation failed; stopping before training")

    adv = {f"lambda_{lam:.2f}": advantages(frozen, meta, fb, meta["y"], lam)
           for lam in LAMBDAS}
    adv["lambda_0.95_shuffled_s1"] = advantages(
        frozen, meta, fb, a2.shuffled_outcomes(meta, 1), 0.95)
    write("w5_a2_3_advantage_stats.json", {"beta": a2.BETA, "max_weight": 5.0,
                                          "stats": advantage_stats(meta, adv)})

    a21 = json.loads((REPORTS / "w5_a2_offline.json").read_text())["runs"]
    refs: dict[str, list[dict]] = {
        "E1": [r for r in a21 if r["variant"] == "E1"],
        "MC": [r for r in a21 if r["variant"] == "E3"],
        "MC-shuffled": [r for r in a21 if r["variant"] == "E3-shuffled"],
    }
    training = []
    for name, lam, shuffled in (("GAE0.95", 0.95, False),
                                ("GAE0.95-shuffled", 0.95, True),
                                ("GAE0.90", 0.90, False)):
        refs[name] = []
        for seed in a2.SEEDS:
            outcomes = a2.shuffled_outcomes(meta, seed) if shuffled else meta["y"]
            weights = awbc_weights(torch.as_tensor(
                advantages(frozen, meta, fb, outcomes, lam)), a2.BETA).numpy()
            run = a2.train(frozen, meta, name, seed, weights=weights, out_root=OUT)
            refs[name].append(run)
            training.append({k: run[k] for k in ("variant", "seed", "matched",
                                                 "epochs_run", "wall_s")}
                            | {"final": run["history"][-1]})

    matched = matched_kl(frozen, meta, refs)
    matched["training_runs"] = training
    write("w5_a2_3_matched_kl.json", matched)

    grads = gradient_alignment(frozen, meta, fb)
    write("w5_a2_3_gradient_alignment.json", grads)

    primary = matched["kl_0.05"]
    c = primary.get("contrasts", {})

    def positive(key):
        return bool(c.get(key, {}).get("excludes_zero") and c[key]["estimate"] > 0)

    reliability = all(t["final"]["finite"] and not t["final"]["drift_flags"]["entropy"]
                      for t in training)
    gate = {"matched": primary["all_matched"],
            "GAE0.95_gt_E1": positive("GAE0.95_minus_E1"),
            "GAE0.95_gt_shuffled": positive("GAE0.95_minus_GAE0.95-shuffled"),
            "gradient_alignment_above_null": grads["lambda_0.95"]["real_above_p95"],
            "reliability": reliability}
    gate["pass"] = all(gate.values())
    render_report(gae_validation, matched, grads, gate,
                  time.perf_counter() - started)
    print(json.dumps({"gate": gate, "wall_s": time.perf_counter() - started}))


def fmt(ci: dict) -> str:
    lo, hi = ci["ci95"]
    verdict = "excludes 0" if ci["excludes_zero"] else "inconclusive"
    return f"{ci['estimate']:+.5f} [{lo:+.5f}, {hi:+.5f}] ({verdict})"


def render_report(validation_report, matched, grads, gate, wall) -> None:
    stats = json.loads((REPORTS / "w5_a2_3_advantage_stats.json").read_text())["stats"]
    lines = ["# W5 A2.3 — GAE advantages for AWBC (offline)", "",
             ("Only the advantage estimator changed vs A2.1. Evaluation advantage fixed "
             "to A_eval = Y − V_probe for every model. Validation split reused many "
             "times: exploratory."), "",
             "## GAE validation", "",
             (f"- λ=1 vs Monte Carlo max |diff|: real "
             f"{validation_report['lambda1_real_max_abs_diff_vs_MC']:.2e}, shuffled "
             f"{validation_report['lambda1_shuffled_max_abs_diff_vs_MC']:.2e}."),
             (f"- Value path parity (no-fallback trajectories): "
             f"{validation_report['value_path_parity_max_abs_diff']:.2e}."),
             (f"- Fallback steps kept in the chain: "
             f"{validation_report['fallback_steps_in_chain']}; trajectories: "
             f"{validation_report['trajectories']}."), "",
             "## Advantage statistics", "",
             "| Estimator | std | corr with MC | sign flips vs MC | ESS |",
             "|---|---|---|---|---|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['std']:.4f} | {s['corr_with_MC']:.4f} | "
                     f"{s['sign_flip_rate_vs_MC']:.2%} | {s['ESS']:.3f} |")
    for target in ("kl_0.05", "kl_0.10"):
        entry = matched[target]
        lines += ["", f"## Matched {target.replace('_', ' ')} (S_MC, team-pair cluster CI)",
                  ""]
        if not entry["all_matched"]:
            lines.append("Not all variants reached this KL in 30 epochs.")
            continue
        lines += ["| Variant | S_MC (seed mean) |", "|---|---|"]
        lines += [f"| {v} | {s:.5f} |" for v, s in entry["S_MC_seed_mean"].items()]
        lines += ["", "| Contrast | Estimate [95% CI] |", "|---|---|"]
        lines += [f"| {k} | {fmt(ci)} |" for k, ci in entry["contrasts"].items()]
        lines += ["", "Per seed (primary contrasts):", ""]
        for seed, contrasts in entry["per_seed"].items():
            lines.append(f"- {seed}: " + "; ".join(f"{k} {fmt(ci)}"
                                                    for k, ci in contrasts.items()))
    lines += ["", "## Gradient alignment at θ = C (100 battle-level permutation nulls)",
              "", "| λ | real cos(train, val) | null mean ± std | null p95 | null ≥ real |",
              "|---|---|---|---|---|"]
    for lam in LAMBDAS:
        g = grads[f"lambda_{lam:.2f}"]
        lines.append(f"| {lam:.2f} | {g['real_cos_train_val']:+.4f} | "
                     f"{g['null_mean']:+.4f} ± {g['null_std']:.4f} | {g['null_p95']:+.4f} | "
                     f"{g['fraction_null_at_or_above_real']:.0%} |")
    lines += ["", "## Gate (KL 0.05)", "",
              *[f"- {k}: {v}" for k, v in gate.items()], "",
              "Decision: " + ("move to dev battle evaluation." if gate["pass"] else
                              "A2.3 is a negative experiment; no battles. Stop tuning "
                              "lr/batch/λ on this pilot (user rule)."), "",
              ("Caveats: shuffled outcomes keep TD residuals, so the control isolates only "
              "the terminal label; GAE bias depends on the weak probe critic (val Brier "
              "0.2235); single pilot dataset; validation reused."), "",
              f"Wall time: {wall:.1f} s."]
    (REPORTS / "w5_a2_3_final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
