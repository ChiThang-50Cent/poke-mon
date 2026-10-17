"""CF1.5: train the small candidate head on frozen C features (offline only).

Grid (pre-registered): {linear, mlp} x {huber, pairwise} x seeds {1,2,3}; early stopping
on fresh-val Delta_head (tau=0); tau in {0, .01, .02, .05} and the config are selected
jointly on fresh val (so val CIs are development evidence only). Shuffled-label control:
selected config, 3 seeds, candidate labels permuted within roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pokeai.cf.head import TAUS, Arrays, load_arrays, save_head
from pokeai.cf.head_train import fit, per_root_h1, pooled_ci, predict, root_gains

ARCHS = ("linear", "mlp")
LOSSES = ("huber", "pairwise")
SEEDS = (1, 2, 3)
C_DIR = "models/pokedecision-v1-value"


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain_block(delta: np.ndarray, arr: Arrays, tau: float) -> dict:
    gain, pick = root_gains(delta, arr, tau)
    switched = pick != 0
    out = {"delta_head": pooled_ci(gain, arr), "switch_rate": float(switched.mean()),
           "switches": int(switched.sum())}
    out["delta_given_switch"] = (
        pooled_ci(gain[switched], arr.subset(switched)) if switched.sum() >= 5
        else {"mean": float(gain[switched].mean()) if switched.any() else None,
              "n": int(switched.sum())}
    )
    out["pass_gt0_ci_excl0"] = bool(out["delta_head"]["mean"] > 0
                                    and out["delta_head"]["ci95"][0] > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-json", default="reports/w5_cf1_5_head.json")
    ap.add_argument("--ckpt-dir", default="checkpoints/w5_cf1_5")
    ap.add_argument("--max-epochs", type=int, default=60)
    args = ap.parse_args()
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_arrays(args.dataset)
    split = data.meta["split"]
    train, val = data.subset(split == "train"), data.subset(split == "val")
    mask = np.load(args.dataset, allow_pickle=False)["mask"]
    mean = train.h.mean(0)
    std = np.maximum(train.h.std(0), 1e-6)

    runs: dict[str, dict] = {}
    for arch in ARCHS:
        for loss in LOSSES:
            for seed in SEEDS:
                r = fit(arch, loss, seed, train, val, mean, std, device,
                        max_epochs=args.max_epochs)
                runs[f"{arch}/{loss}/{seed}"] = r
                print(f"{arch}/{loss}/{seed} best_epoch={r['best_epoch']} "
                      f"val_delta0={r['best_val_delta_tau0']:+.4f}", flush=True)

    def val_delta(key: str, tau: float, arr: Arrays = val) -> float:
        d = predict(runs[key]["head"], arr, mean, std, device)
        return float(root_gains(d, arr, tau)[0].mean())

    grid = {}
    for arch in ARCHS:
        for loss in LOSSES:
            for tau in TAUS:
                vals = [val_delta(f"{arch}/{loss}/{s}", tau) for s in SEEDS]
                grid[f"{arch}/{loss}/tau={tau}"] = {
                    "mean_over_seeds": float(np.mean(vals)), "per_seed": vals}
    best_key = max(grid, key=lambda k: grid[k]["mean_over_seeds"])
    arch, loss, tau_s = best_key.split("/")
    tau = float(tau_s.split("=")[1])
    print("selected", best_key, flush=True)

    controls = {}
    for seed in SEEDS:
        controls[seed] = fit(arch, loss, seed, train, val, mean, std, device,
                             shuffle_labels=True, max_epochs=args.max_epochs)

    def deltas(rs: dict, arr: Arrays) -> list[np.ndarray]:
        return [predict(r["head"], arr, mean, std, device) for r in rs.values()]

    sel_runs = {s: runs[f"{arch}/{loss}/{s}"] for s in SEEDS}
    saved_seed = SEEDS[0]
    sel_val = deltas(sel_runs, val)[0]
    ctrl_val = deltas(controls, val)

    def metrics(arr: Arrays, d_head: np.ndarray, d_ctrl: list[np.ndarray]) -> dict:
        sp, sg = per_root_h1(d_head, arr)
        cps = [per_root_h1(c, arr) for c in d_ctrl]
        sp_c = np.mean([np.nan_to_num(c[0]) for c in cps], axis=0)
        sg_c = np.nanmean([c[1] for c in cps], axis=0) if len(cps) else sg
        ok3 = ~np.isnan(sp) | (arr.valid.sum(1) >= 3)
        out = {"spearman": {}, "sign_acc": {}}
        if ok3.any():
            a = np.nan_to_num(sp)[ok3]
            b = sp_c[ok3]
            sub = arr.subset(ok3)
            out["spearman"] = {"roots": int(ok3.sum()), "head": pooled_ci(a, sub),
                               "control": pooled_ci(b, sub),
                               "paired_diff": pooled_ci(a - b, sub)}
            out["spearman"]["beats_control"] = bool(
                out["spearman"]["paired_diff"]["ci95"][0] > 0)
        ok2 = ~np.isnan(sg)
        if ok2.any():
            sub = arr.subset(ok2)
            ctrl_acc = np.nan_to_num(sg_c[ok2], nan=0.5)
            out["sign_acc"] = {"roots": int(ok2.sum()), "head": pooled_ci(sg[ok2], sub),
                               "control": pooled_ci(ctrl_acc, sub),
                               "paired_diff": pooled_ci(sg[ok2] - ctrl_acc, sub)}
        return out

    sources = val.meta["source"]
    report: dict = {
        "device": device, "selected": {"arch": arch, "loss": loss, "tau": tau,
                                       "saved_seed": saved_seed, "key": best_key},
        "selection_rule": "max over (arch, loss, tau) of mean-over-3-seeds pooled fresh-val "
                          "Delta_head; early stopping per run on val Delta_head(tau=0); "
                          "saved head = pre-declared seed 1 of the selected config. "
                          "VAL USED FOR SELECTION: CIs are development evidence only.",
        "grid": grid, "n_train": len(train.h), "n_val": len(val.h),
        "val": {"pooled": {}, "sampled": {}, "greedy": {}},
        "learning_curves": {k: v["curve"] for k, v in runs.items()}
        | {f"control/{s}": c["curve"] for s, c in controls.items()},
        "best_epochs": {k: v["best_epoch"] for k, v in runs.items()},
        "seed_spread": {
            "val_delta_head_per_seed": [val_delta(f"{arch}/{loss}/{s}", tau) for s in SEEDS],
            "switch_rate_per_seed": [
                float((root_gains(d, val, tau)[1] != 0).mean())
                for d in deltas(sel_runs, val)],
        },
    }
    for name, sel in (("pooled", np.ones(len(sources), bool)),
                      ("sampled", sources == "sampled"), ("greedy", sources == "greedy")):
        sub = val.subset(sel)
        d_head = sel_val[sel]
        block = {"roots": int(sel.sum()), **gain_block(d_head, sub, tau),
                 "H1": metrics(sub, d_head, [c[sel] for c in ctrl_val])}
        block["control_delta_head"] = [
            float(root_gains(c[sel], sub, tau)[0].mean()) for c in ctrl_val]
        report["val"][name] = block
    pooled = val
    gain, pick = root_gains(sel_val, pooled, tau)
    groups = {}
    for key in ("opponent", "phase", "forced_switch"):
        for value in sorted({str(x) for x in pooled.meta[key]}):
            sel = np.array([str(x) == value for x in pooled.meta[key]])
            groups[f"{key}={value}"] = {
                "roots": int(sel.sum()), "delta_head": float(gain[sel].mean()),
                "switch_rate": float((pick[sel] != 0).mean())}
    report["val"]["by_group"] = groups
    train_delta = deltas(sel_runs, train)[0]
    report["train"] = {"pooled": gain_block(train_delta, train, tau)}
    # H4 reliability
    chosen_slot = val.slots[np.arange(len(pick)), pick]
    val_rows = np.flatnonzero(split == "val")
    legal = mask[val_rows, chosen_slot] > 0
    report["H4"] = {
        "invalid_actions": int((~legal).sum()),
        "outside_candidate_set": int((~val.valid[np.arange(len(pick)), pick]).sum()),
        "finite_outputs": bool(np.isfinite(sel_val[val.valid]).all()),
        "hidden_state_input": "none: head inputs are (h, slot one-hot, logp) only",
    }
    g = report["val"]["greedy"]["delta_head"]
    report["H3_warning"] = bool(g["mean"] < 0 and g["ci95"][1] < 0)
    h1_pass = report["val"]["pooled"]["H1"]["spearman"].get("beats_control", False) and (
        report["val"]["pooled"]["H1"]["spearman"]["head"]["ci95"][0] > 0)
    report["gates"] = {
        "H1": bool(h1_pass),
        "H2": report["val"]["pooled"]["pass_gt0_ci_excl0"],
        "H4": report["H4"]["invalid_actions"] == 0 and report["H4"]["finite_outputs"],
    }
    report["gates"]["cf1_6_eligible"] = all(report["gates"].values())

    save_head(Path(args.ckpt_dir), sel_runs[saved_seed]["head"], {
        "arch": arch, "loss": loss, "tau": tau, "seed": saved_seed,
        "h_mean": mean.tolist(), "h_std": std.tolist(),
        "c_checkpoint_sha256": sha(f"{C_DIR}/best.pt"),
        "c_dir": C_DIR, "dataset": args.dataset, "dataset_sha256": sha(args.dataset),
        "split": "train = 400 pilot + 900 sampled + 300 greedy; val = 300 sampled + "
                 "100 greedy (team-disjoint)",
        "grid": {"archs": ARCHS, "losses": LOSSES, "seeds": SEEDS, "taus": TAUS},
        "selection_rule": report["selection_rule"], "best_epoch": sel_runs[saved_seed]["best_epoch"],
    })
    report["wall_s"] = time.perf_counter() - started
    Path(args.out_json).write_text(json.dumps(report, indent=1))
    print(json.dumps({"selected": report["selected"], "gates": report["gates"],
                      "val_delta": report["val"]["pooled"]["delta_head"]}))


if __name__ == "__main__":
    main()
