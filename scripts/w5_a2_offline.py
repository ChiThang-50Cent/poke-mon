"""W5 Experiment A2.1: offline AWBC signal test at matched KL. No battles.

Pre-registered (user plan, W5.3 audit follow-up):
- Data: W5.1 pilot C decisions, W5.2/W5.3 team-pair split (train 80 / val 20).
- Actor: copy of C, batch 64 full trajectories, AdamW lr 3e-4 wd 1e-4, clip 1.0,
  lambda_KL 0.1, max 30 epochs, seeds 1/2/3. Loss = existing `awbc_loss`.
- Variants: E1 (w = 1), E3 (probe advantage, beta 2, max weight 5), E3-shuffled
  (battle outcomes permuted at battle level inside (split, opponent type) groups,
  self-play complementarity preserved, seed-specific permutation).
- Matched KL: checkpoint at the FIRST epoch whose val KL(C||actor) >= 0.05
  (primary) and >= 0.10 (exploratory sensitivity only).
- Primary metric: S(theta) = E_val[A_t * (log pi_theta(a_t|s_t) - log pi_C(a_t|s_t))]
  with the real probe advantage; differences E3-E1 and E3-shuffled, 95% CI from a
  5,000-resample bootstrap clustered by team pair (seed-averaged Delta per decision).
- Cross-split gradient: at theta = C, advantage component g^A = g_w - g_1 of the
  BC term on train and on val; cos(g^A_train, g^A_val) for the real advantage vs a
  20-permutation battle-level null. Both per-batch and global weight normalisation.
- Gate (KL 0.05): S(E3)-S(E1) > 0 and S(E3)-S(shuffled) > 0 with CIs excluding 0;
  real cross-split cosine above the null 95th percentile; no NaN / mask violation /
  entropy collapse (> 30 % change); all three seeds reported.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pokeai.models.pokedecision import load_pokedecision, save_checkpoint
from pokeai.training.awbc import (
    AdvantageSequences,
    awbc_loss,
    awbc_weights,
    make_arrays,
    run_epoch,
)
from pokeai.training.train_bc import eligible_steps, pad_sequences, seed_everything

C_DIR = Path("models/pokedecision-v1-value")
FROZEN = Path("checkpoints/w5_3/frozen_dataset.pt")
CACHE = Path("models/pokedecision-w5-critic-pilot/feature_cache.pt")
OUT = Path("checkpoints/w5_a2")
BETA, LR, BATCH, MAX_EPOCHS, SEEDS = 2.0, 3e-4, 64, 30, (1, 2, 3)
TARGETS = (0.05, 0.10)
NULL_PERMUTATIONS = range(100, 120)
RESAMPLES, BOOT_SEED = 5000, 20261016


# ---------------------------------------------------------------- data ----------
def load_inputs() -> tuple[dict, dict]:
    frozen = torch.load(FROZEN, map_location="cpu", weights_only=False)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)["data"]
    records = frozen["records"]
    y = np.array([r["Y"] for r in records])
    probe = np.array([r["Probe"] for r in records])
    if not (np.array_equal(y, np.asarray(cache["y"], dtype=float))
            and np.allclose(probe, np.asarray(cache["probe"]), atol=2e-5)):
        raise ValueError("frozen records do not align with the W5.2 feature cache")
    if not np.allclose(frozen["advantages"]["E3"], y - probe, atol=1e-6):
        raise ValueError("stored E3 advantage != Y - Probe")
    meta = {
        "y": y, "probe": probe,
        "pair": np.asarray(cache["pair"]),
        "split": np.array([r["split"] for r in records]),
        "opponent": np.array([r["opponent"] for r in records]),
        "battle": np.array([r["key"][0] for r in records]),
        "pov": np.array([r["key"][1] for r in records]),
    }
    return frozen, meta


def shuffled_outcomes(meta: dict, seed: int) -> np.ndarray:
    """Permute battle outcomes inside (split, opponent) groups.

    A battle's scalar outcome is its p1-POV result for self-play (p2 = 1 - p1,
    so complementarity survives) and the single C POV's result otherwise.
    """
    scalar: dict[str, float] = {}
    group: dict[str, tuple[str, str]] = {}
    for b, pov, y, s, o in zip(meta["battle"], meta["pov"], meta["y"],
                               meta["split"], meta["opponent"]):
        group[b] = (s, o)
        if o != "selfplay" or pov == "p1":
            scalar[b] = float(y)
        elif b not in scalar:
            scalar[b] = 1.0 - float(y)
    rng = np.random.default_rng(seed)
    mapped: dict[str, float] = {}
    members = defaultdict(list)
    for b in sorted(scalar):
        members[group[b]].append(b)
    for key in sorted(members):
        battles = members[key]
        perm = rng.permutation(len(battles))
        for b, j in zip(battles, perm):
            mapped[b] = scalar[battles[j]]
    result = np.array([
        mapped[b] if (o != "selfplay" or pov == "p1") else 1.0 - mapped[b]
        for b, pov, o in zip(meta["battle"], meta["pov"], meta["opponent"])
    ])
    for b in np.unique(meta["battle"][meta["opponent"] == "selfplay"]):
        rows = meta["battle"] == b
        by_pov = {p: result[rows & (meta["pov"] == p)] for p in ("p1", "p2")}
        if all(len(v) for v in by_pov.values()) and by_pov["p1"][0] + by_pov["p2"][0] != 1:
            raise AssertionError(f"self-play complementarity broken: {b}")
    return result


def weights_for(variant: str, meta: dict, seed: int) -> np.ndarray:
    if variant == "E1":
        return np.ones(len(meta["y"]), np.float32)
    y = meta["y"] if variant == "E3" else shuffled_outcomes(meta, seed)
    return awbc_weights(torch.as_tensor(y - meta["probe"]), BETA).numpy()


def loaders(frozen, weights, seed):
    return {
        split: DataLoader(
            AdvantageSequences(make_arrays(frozen, split, weights)), batch_size=BATCH,
            shuffle=split == "train", collate_fn=pad_sequences,
            generator=torch.Generator().manual_seed(seed),
        )
        for split in ("train", "val")
    }


def fresh_actor():
    model, _, _ = load_pokedecision(C_DIR, "cuda")
    model.value_head = None
    model.config.value_head = False
    return model


# ------------------------------------------------------------- training --------
def train(frozen, meta, variant: str, seed: int) -> dict:
    seed_everything(seed)
    weights = weights_for(variant, meta, seed)
    data = loaders(frozen, weights, seed)
    model = fresh_actor()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    out = OUT / f"{variant}-s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    history, matched, started = [], {}, time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        training = run_epoch(model, data["train"], optimizer)
        with torch.no_grad():
            val = run_epoch(model, data["val"])
        finite = all(torch.isfinite(p).all() for p in model.parameters())
        row = {"epoch": epoch, "train_loss": training["loss"],
               "val_KL": val["KL_C_to_actor"], "val_KL_p95": val["KL_C_to_actor_p95"],
               "val_entropy": val["entropy"], "C_entropy": val["reference_entropy"],
               "val_greedy_agreement": val["greedy_action_agreement"],
               "drift_flags": val["drift_flags"], "finite": bool(finite)}
        history.append(row)
        if not finite:
            break
        for target in TARGETS:
            if target not in matched and val["KL_C_to_actor"] >= target:
                target_dir = out / f"kl{target:.2f}"
                target_dir.mkdir(exist_ok=True)
                save_checkpoint(model, target_dir / "best.pt")
                (target_dir / "config.yaml").write_text(yaml.safe_dump({
                    "model": asdict(model.config), "experiment": "W5-A2.1",
                    "variant": variant, "seed": seed, "beta": BETA, "lr": LR,
                    "batch_trajectories": BATCH, "kl_coefficient": 0.1,
                    "kl_target": target, "epoch": epoch,
                    "val_KL": val["KL_C_to_actor"], "production": False,
                }, sort_keys=False))
                (target_dir / "tokenizer.json").write_bytes(
                    (C_DIR / "tokenizer.json").read_bytes())
                matched[target] = {"epoch": epoch, "val_KL": val["KL_C_to_actor"],
                                   "dir": str(target_dir)}
        if len(matched) == len(TARGETS):
            break
    return {"variant": variant, "seed": seed, "matched": matched,
            "epochs_run": len(history), "history": history,
            "wall_s": time.perf_counter() - started}


# ------------------------------------------------------------- metrics ---------
@torch.no_grad()
def chosen_logp(model, frozen) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per val C decision (record order): log pi(a_t), greedy action, forced flag."""
    ones = np.ones(len(frozen["records"]), np.float32)
    loader = DataLoader(AdvantageSequences(make_arrays(frozen, "val", ones)),
                        batch_size=BATCH, shuffle=False, collate_fn=pad_sequences)
    model.eval()
    out_logp, out_greedy, out_forced, illegal = [], [], [], 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        logits = model(batch["tokens"].long(), batch["numbers"])[0]
        sel = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
        mask = batch["masks"][sel].bool()
        logp = logits[sel].masked_fill(~mask, -torch.inf).log_softmax(-1)
        greedy = logp.argmax(-1)
        illegal += int((~mask.gather(-1, greedy.unsqueeze(-1))).sum())
        out_logp.append(logp.gather(-1, batch["actions"][sel].unsqueeze(-1))
                        .squeeze(-1).cpu())
        out_greedy.append(greedy.cpu())
        out_forced.append((mask.sum(-1) == 1).cpu())
    if illegal:
        raise ValueError(f"{illegal} greedy actions outside the legal mask")
    return (torch.cat(out_logp).numpy().astype(np.float64),
            torch.cat(out_greedy).numpy(), torch.cat(out_forced).numpy())


def val_rows(frozen) -> np.ndarray:
    rows = np.concatenate([s["row"] for s in frozen["sequences"] if s["split"] == "val"])
    return rows[rows >= 0]


def cluster_ci(contrib: np.ndarray, clusters: np.ndarray) -> dict:
    """Mean over decisions; percentile CI resampling whole team-pair clusters."""
    ids, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=contrib)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    stats = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    estimate = float(contrib.mean())
    return {"estimate": estimate, "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "clusters": len(ids)}


def alignment(frozen, meta, runs: list[dict]) -> dict:
    rows = val_rows(frozen)
    c_logp, c_greedy, forced = chosen_logp(fresh_actor(), frozen)
    if len(c_logp) != len(rows):
        raise ValueError("eligible val steps do not align with C decision records")
    adv = (meta["y"] - meta["probe"])[rows]
    pairs = meta["pair"][rows]
    report = {"val_decisions": len(rows), "val_team_pairs": len(set(pairs))}
    for target in TARGETS:
        deltas, per_run = defaultdict(list), {}
        for run in runs:
            if target not in run["matched"]:
                continue
            model, _, _ = load_pokedecision(run["matched"][target]["dir"], "cuda")
            logp, greedy, _ = chosen_logp(model, frozen)
            delta = logp - c_logp
            deltas[run["variant"]].append(delta)
            per_run[f"{run['variant']}-s{run['seed']}"] = {
                "epoch": run["matched"][target]["epoch"],
                "val_KL": run["matched"][target]["val_KL"],
                "S": float(np.mean(adv * delta)),
                "greedy_flip_rate_nonforced": float((greedy != c_greedy)[~forced].mean()),
            }
        entry: dict = {"runs": per_run}
        complete = all(len(deltas[v]) == len(SEEDS) for v in ("E1", "E3", "E3-shuffled"))
        entry["all_variants_matched_all_seeds"] = complete
        if complete:
            mean = {v: np.mean(deltas[v], axis=0) for v in deltas}
            entry["S_seed_mean"] = {v: float(np.mean(adv * mean[v])) for v in mean}
            entry["E3_minus_E1"] = cluster_ci(adv * (mean["E3"] - mean["E1"]), pairs)
            entry["E3_minus_shuffled"] = cluster_ci(
                adv * (mean["E3"] - mean["E3-shuffled"]), pairs)
            entry["per_seed"] = {
                f"s{s}": {
                    "E3_minus_E1": cluster_ci(
                        adv * (deltas["E3"][i] - deltas["E1"][i]), pairs),
                    "E3_minus_shuffled": cluster_ci(
                        adv * (deltas["E3"][i] - deltas["E3-shuffled"][i]), pairs),
                }
                for i, s in enumerate(SEEDS)
            }
        report[f"kl_{target:.2f}"] = entry
    return report


def flat(grads) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def bc_gradients(model, frozen, weights: np.ndarray, split: str,
                 global_mean: float) -> dict[str, torch.Tensor]:
    """Mean-over-batches BC-term gradient at the current weights (fixed order)."""
    loader = DataLoader(AdvantageSequences(make_arrays(frozen, split, weights)),
                        batch_size=BATCH, shuffle=False, collate_fn=pad_sequences)
    params = [p for p in model.parameters() if p.requires_grad]
    totals: dict[str, torch.Tensor] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        logits = model(batch["tokens"].long(), batch["numbers"])[0]
        _, bc_batch, _ = awbc_loss(logits, batch["actions"], batch["masks"],
                                   batch["valid"], batch["weights"], batch["reference"])
        sel = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
        mask = batch["masks"][sel].bool()
        logp = logits[sel].masked_fill(~mask, -torch.inf).log_softmax(-1)
        nll = -logp.gather(-1, batch["actions"][sel].unsqueeze(-1)).squeeze(-1)
        bc_global = (batch["weights"][sel].detach() / global_mean * nll).mean()
        for name, loss in (("per_batch", bc_batch), ("global", bc_global)):
            g = flat(torch.autograd.grad(loss, params, retain_graph=True,
                                         allow_unused=True)).cpu()
            totals[name] = g if name not in totals else totals[name] + g
        n += 1
    return {k: v / n for k, v in totals.items()}


def gradient_tests(frozen, meta) -> dict:
    model = fresh_actor()
    model.train(True)
    train_mask = meta["split"] == "train"
    ones = np.ones(len(meta["y"]), np.float32)
    base = {s: bc_gradients(model, frozen, ones, s, 1.0) for s in ("train", "val")}
    cos = torch.nn.functional.cosine_similarity

    def component(weights):
        mean_w = float(weights[train_mask].mean())
        grads = {s: bc_gradients(model, frozen, weights, s, mean_w) for s in ("train", "val")}
        result = {}
        for norm in ("per_batch", "global"):
            a_train = grads["train"][norm] - base["train"][norm]
            a_val = grads["val"][norm] - base["val"][norm]
            result[norm] = {"cos_train_val": float(cos(a_train, a_val, dim=0)),
                            "norm_train": float(a_train.norm()),
                            "norm_val": float(a_val.norm())}
        result["normalisation_full_grad_cos"] = float(
            cos(grads["train"]["per_batch"], grads["train"]["global"], dim=0))
        result["normalisation_component_cos"] = float(cos(
            grads["train"]["per_batch"] - base["train"]["per_batch"],
            grads["train"]["global"] - base["train"]["global"], dim=0))
        result["normalisation_norm_ratio_global_over_batch"] = float(
            grads["train"]["global"].norm() / grads["train"]["per_batch"].norm())
        return result

    real = component(weights_for("E3", meta, 0))
    null = [component(awbc_weights(
        torch.as_tensor(shuffled_outcomes(meta, s) - meta["probe"]), BETA).numpy())
        for s in NULL_PERMUTATIONS]
    report = {"real_E3_beta2": real, "null_permutations": len(null)}
    for norm in ("per_batch", "global"):
        values = np.array([n[norm]["cos_train_val"] for n in null])
        report[f"null_{norm}"] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "p95": float(np.quantile(values, 0.95)),
            "real_rank_fraction_above": float((values >= real[norm]["cos_train_val"]).mean()),
            "real_above_p95": bool(real[norm]["cos_train_val"] > np.quantile(values, 0.95)),
        }
    return report


def gate(align: dict, grads: dict, runs: list[dict]) -> dict:
    primary = align["kl_0.05"]
    reliability = {
        "all_finite": all(r["history"][-1]["finite"] for r in runs),
        "entropy_collapse": any(
            h["drift_flags"]["entropy"] for r in runs for h in r["history"]),
        "mask_violations": 0,
    }
    checks = {
        "matched": primary["all_variants_matched_all_seeds"],
        "E3_gt_E1": bool(primary.get("E3_minus_E1", {}).get("excludes_zero")
                         and primary["E3_minus_E1"]["estimate"] > 0),
        "E3_gt_shuffled": bool(primary.get("E3_minus_shuffled", {}).get("excludes_zero")
                               and primary["E3_minus_shuffled"]["estimate"] > 0),
        "cross_split_gradient": grads["null_per_batch"]["real_above_p95"],
        "reliability": reliability["all_finite"] and not reliability["entropy_collapse"],
    }
    return {"checks": checks, "reliability": reliability, "pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/w5_a2_offline.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    frozen, meta = load_inputs()
    runs = [train(frozen, meta, v, s) for v in ("E1", "E3", "E3-shuffled") for s in SEEDS]
    align = alignment(frozen, meta, runs)
    grads = gradient_tests(frozen, meta)
    report = {"contract": __doc__, "train_trajectories":
              sum(s["split"] == "train" for s in frozen["sequences"]),
              "runs": runs, "alignment": align, "gradients_at_C": grads,
              "gate_kl_0.05": gate(align, grads, runs),
              "wall_s": time.perf_counter() - started}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate_kl_0.05"], "wall_s": report["wall_s"]}))


if __name__ == "__main__":
    main()
