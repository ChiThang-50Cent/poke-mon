"""Offline audit of the W5.3 AWBC pilot: no battles, no new training artefacts.

1. Replays selected actor trainings with per-step gradient instrumentation and
   proves the replay is faithful (final weights bit-identical to the saved run).
2. Gradient geometry at the initial policy C: full-train gradient cosines between
   E1 / AWBC / shuffled weightings and gradient signal-to-noise.
3. Policy change of every saved actor on the 20% val C decisions: greedy flip
   rate vs C (all / non-forced states) and whether log-prob changes of the
   chosen action follow the advantage sign.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from pokeai.models.pokedecision import load_pokedecision
from pokeai.training.awbc import (
    AdvantageSequences,
    awbc_loss,
    awbc_weights,
    make_arrays,
    run_epoch,
    shuffle_advantages,
)
from pokeai.training.train_bc import eligible_steps, pad_sequences, seed_everything

C_DIR = Path("models/pokedecision-v1-value")
CKPT = Path("checkpoints/w5_3")
REPLAYS = ["E1-none-s1", "E2-0.5-s1", "E3-2.0-s1", "E3-shuffled-2.0-s1"]


def parse(run_id: str) -> tuple[str, float | None, int, bool]:
    parts = run_id.split("-")
    control = "shuffled" in parts
    parts = [p for p in parts if p != "shuffled"]
    beta = None if parts[1] == "none" else float(parts[1])
    return parts[0], beta, int(parts[2].removeprefix("s")), control


def weights_for(frozen, experiment, beta, seed, control) -> np.ndarray:
    records = frozen["records"]
    if experiment == "E1":
        return np.ones(len(records), np.float32)
    advantage = frozen["advantages"][experiment].copy()
    if control:
        advantage = shuffle_advantages(
            advantage, [r["opponent"] for r in records],
            [r["phase"] for r in records], [r["split"] for r in records], seed,
        )
    return awbc_weights(torch.as_tensor(advantage), beta).numpy()


def fresh_actor():
    model, _, _ = load_pokedecision(C_DIR, "cuda")
    model.value_head = None
    model.config.value_head = False
    return model


def flat(grads) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def replay(frozen, run_id: str) -> dict:
    """Exactly mirrors awbc.train_actor; adds grad norms via autograd.grad only."""
    experiment, beta, seed, control = parse(run_id)
    seed_everything(seed)
    weights = weights_for(frozen, experiment, beta, seed, control)
    model = fresh_actor()
    loaders = {
        split: DataLoader(
            AdvantageSequences(make_arrays(frozen, split, weights)), batch_size=16,
            shuffle=split == "train", collate_fn=pad_sequences,
            generator=torch.Generator().manual_seed(seed),
        )
        for split in ("train", "val")
    }
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    start = flat([p.detach().clone() for p in params])
    epochs = []
    for epoch in range(1, 6):
        model.train(True)
        steps = []
        for batch in loaders["train"]:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            logits = model(batch["tokens"].long(), batch["numbers"])[0]
            losses = awbc_loss(logits, batch["actions"], batch["masks"],
                               batch["valid"], batch["weights"], batch["reference"])
            if losses is None:
                continue
            loss, bc, kl = losses
            g_bc = flat(torch.autograd.grad(bc, params, retain_graph=True,
                                            allow_unused=True))
            g_kl = flat(torch.autograd.grad(0.1 * kl, params, retain_graph=True,
                                            allow_unused=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            steps.append({
                "bc_norm": float(g_bc.norm()), "kl_norm": float(g_kl.norm()),
                "total_norm_preclip": total, "clipped": total > 1.0,
                "cos_bc_kl": float(torch.nn.functional.cosine_similarity(
                    g_bc, g_kl, dim=0)) if float(g_kl.norm()) > 0 else None,
            })
        with torch.no_grad():
            validation = run_epoch(model, loaders["val"])
        bc_n = np.array([s["bc_norm"] for s in steps])
        kl_n = np.array([s["kl_norm"] for s in steps])
        tot = np.array([s["total_norm_preclip"] for s in steps])
        cos = np.array([s["cos_bc_kl"] for s in steps if s["cos_bc_kl"] is not None])
        epochs.append({
            "epoch": epoch, "optimizer_steps": len(steps),
            "bc_grad_norm_mean": float(bc_n.mean()),
            "kl_grad_norm_mean": float(kl_n.mean()),
            "kl_over_bc_grad_ratio_mean": float((kl_n / bc_n).mean()),
            "total_grad_norm_preclip_mean": float(tot.mean()),
            "clipped_step_fraction": float(np.mean([s["clipped"] for s in steps])),
            "cos_bc_vs_kl_grad_mean": float(cos.mean()) if len(cos) else None,
            "val_KL_C_to_actor": validation["KL_C_to_actor"],
            "val_greedy_agreement": validation["greedy_action_agreement"],
            "param_distance_from_C": float(
                (flat([p.detach() for p in params]) - start).norm()),
        })
    saved = torch.load(CKPT / run_id / "best.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
    current = {k: v.cpu() for k, v in model.state_dict().items()}
    identical = all(torch.equal(saved[k], current[k]) for k in saved)
    return {"run": run_id, "replay_bit_identical_to_saved": identical,
            "parameter_norm_C": float(start.norm()), "epochs": epochs}


def gradient_geometry(frozen) -> dict:
    """Full-train BC-term gradients at theta = C, fixed batch order."""
    variants: dict[str, tuple[str, float | None, int, bool]] = {
        "E1": ("E1", None, 1, False)
    }
    for exp in ("E2", "E3"):
        for beta in (0.5, 1.0, 2.0):
            variants[f"{exp}-{beta}"] = (exp, beta, 1, False)
    variants["E3-shuffled-2.0"] = ("E3", 2.0, 1, True)
    variants["E3-shuffled-0.5"] = ("E3", 0.5, 1, True)
    model = fresh_actor()
    model.train(True)
    params = [p for p in model.parameters() if p.requires_grad]
    full, batch_grads = {}, {}
    for name, (exp, beta, seed, control) in variants.items():
        weights = weights_for(frozen, exp, beta, seed, control)
        loader = DataLoader(AdvantageSequences(make_arrays(frozen, "train", weights)),
                            batch_size=16, shuffle=False, collate_fn=pad_sequences)
        total, per_batch, kl_norms = None, [], []
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            logits = model(batch["tokens"].long(), batch["numbers"])[0]
            _, bc, kl = awbc_loss(logits, batch["actions"], batch["masks"],
                                  batch["valid"], batch["weights"], batch["reference"])
            g = flat(torch.autograd.grad(bc, params, retain_graph=True,
                                         allow_unused=True))
            kl_norms.append(float(flat(torch.autograd.grad(
                0.1 * kl, params, allow_unused=True)).norm()))
            per_batch.append(g.cpu())
            total = g if total is None else total + g
        full[name] = (total / len(per_batch)).cpu()
        batch_grads[name] = per_batch
        norms = torch.stack([b.norm() for b in per_batch])
        full[name + "/stats"] = {
            "full_grad_norm": float(full[name].norm()),
            "mean_batch_grad_norm": float(norms.mean()),
            "signal_to_noise": float(full[name].norm() / norms.mean()),
            "kl_grad_norm_at_C_max": max(kl_norms),
        }
    cos = torch.nn.functional.cosine_similarity

    def per_batch_cos(a, b):
        return float(np.mean([float(cos(x, y, dim=0))
                              for x, y in zip(batch_grads[a], batch_grads[b])]))

    names = [n for n in variants]
    report = {"stats": {n: full[n + "/stats"] for n in names},
              "full_cosine_vs_E1": {n: float(cos(full[n], full["E1"], dim=0))
                                    for n in names if n != "E1"},
              "per_batch_cosine_vs_E1_mean": {n: per_batch_cos(n, "E1")
                                              for n in names if n != "E1"}}
    # Advantage-specific direction: what weighting adds beyond plain BC.
    for real, shuffled in (("E3-2.0", "E3-shuffled-2.0"),
                           ("E3-0.5", "E3-shuffled-0.5")):
        d_real = full[real] - full["E1"]
        d_shuf = full[shuffled] - full["E1"]
        report[f"delta_{real}"] = {
            "norm_weighting_component": float(d_real.norm()),
            "norm_ratio_to_E1_full_grad": float(d_real.norm() / full["E1"].norm()),
            "cosine_real_vs_shuffled_component": float(cos(d_real, d_shuf, dim=0)),
            "norm_shuffled_component": float(d_shuf.norm()),
        }
    return report


@torch.no_grad()
def policy_change(frozen) -> dict:
    arrays = make_arrays(frozen, "val", np.ones(len(frozen["records"]), np.float32))
    rows = np.concatenate([s["row"] for s in frozen["sequences"] if s["split"] == "val"])
    loader = DataLoader(AdvantageSequences(arrays), batch_size=16, shuffle=False,
                        collate_fn=pad_sequences)

    def outputs(model):
        model.eval()
        logp_all, greedy, forced = [], [], []
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            logits = model(batch["tokens"].long(), batch["numbers"])[0]
            sel = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
            mask = batch["masks"][sel].bool()
            logp = logits[sel].masked_fill(~mask, -torch.inf).log_softmax(-1)
            logp_all.append(logp.gather(-1, batch["actions"][sel].unsqueeze(-1))
                            .squeeze(-1).cpu())
            greedy.append(logp.argmax(-1).cpu())
            forced.append((mask.sum(-1) == 1).cpu())
        return torch.cat(logp_all), torch.cat(greedy), torch.cat(forced)

    # Row order of eligible steps equals record order of val C decisions.
    eligible_rows = rows[rows >= 0]
    c_logp, c_greedy, forced = outputs(fresh_actor())
    if len(c_logp) != len(eligible_rows):
        raise ValueError("Eligible val steps do not align with C decision records")
    adv = {k: torch.as_tensor(frozen["advantages"][k][eligible_rows])
           for k in ("E2", "E3")}
    result = {"val_decisions": len(c_logp), "nonforced": int((~forced).sum()), "runs": {}}
    for run_dir in sorted(p for p in CKPT.iterdir() if (p / "best.pt").exists()):
        model, _, _ = load_pokedecision(run_dir, "cuda")
        logp, greedy, _ = outputs(model)
        delta = logp - c_logp
        flips = greedy != c_greedy
        entry = {"greedy_flip_rate_all": float(flips.float().mean()),
                 "greedy_flip_rate_nonforced": float(flips[~forced].float().mean()),
                 "mean_delta_logp_chosen": float(delta.mean())}
        for key, a in adv.items():
            entry[f"corr_delta_logp_vs_A_{key}"] = float(
                np.corrcoef(delta.numpy(), a.numpy())[0, 1])
            entry[f"delta_logp_A_pos_minus_neg_{key}"] = float(
                delta[a > 0].mean() - delta[a <= 0].mean())
        result["runs"][run_dir.name] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/w5_3_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    frozen = torch.load(CKPT / "frozen_dataset.pt", map_location="cpu",
                        weights_only=False)
    report = {"train_trajectories": sum(s["split"] == "train" for s in frozen["sequences"]),
              "policy_change_val": policy_change(frozen),
              "gradient_geometry_at_C": gradient_geometry(frozen),
              "replays": [replay(frozen, run) for run in REPLAYS]}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({r["run"]: r["replay_bit_identical_to_saved"]
                      for r in report["replays"]}))


if __name__ == "__main__":
    main()
