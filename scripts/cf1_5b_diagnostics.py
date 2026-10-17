"""CF1.5b fail-path diagnostics (offline, no rollouts, same fresh val).

1 learning curve by #train roots; 2 label noise / ceilings; 3 action coverage;
4 non-learned baselines fitted on TRAIN only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pokeai.cf import cf1_metrics as M
from pokeai.cf.head import Arrays, load_arrays, make_head
from pokeai.cf.head_train import fit, per_root_h1, pooled_ci, predict, root_gains
from pokeai.cf.signal_stats import spearman

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (1, 2, 3)
SUBSET_SEED = 20261020
CONFIGS = {  # arch, loss, tau used for the "selected tau" column
    "mlp/pairwise": ("mlp", "pairwise", 0.02),
    "linear/pairwise": ("linear", "pairwise", 0.05),
}


def slot_type(slot: int) -> str:
    return "move" if slot <= 3 else "switch" if slot <= 8 else "move+tera"


def types_of(arr: Arrays) -> np.ndarray:
    out = np.full(arr.slots.shape, "", dtype=object)
    for i in range(arr.slots.shape[0]):
        for c in range(arr.slots.shape[1]):
            if arr.valid[i, c]:
                out[i, c] = slot_type(int(arr.slots[i, c]))
    return out


def ci_of(values: np.ndarray, arr: Arrays) -> dict:
    return pooled_ci(np.asarray(values, float), arr)


def nested_subsets(train: Arrays) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(SUBSET_SEED)
    source = train.meta["source"]
    perms = {s: rng.permutation(np.flatnonzero(source == s)) for s in sorted(set(source))}
    out = {}
    for f in FRACTIONS:
        keep = np.concatenate([p[: math.ceil(f * len(p))] for p in perms.values()])
        out[f] = np.sort(keep)
    return out


def head_metrics(delta: np.ndarray, arr: Arrays, taus: list[float]) -> dict:
    sp, _ = per_root_h1(delta, arr)
    ok = arr.valid.sum(1) >= 3
    out = {"spearman": ci_of(np.nan_to_num(sp)[ok], arr.subset(ok))}
    for tau in taus:
        gain, pick = root_gains(delta, arr, tau)
        out[f"delta_tau={tau}"] = ci_of(gain, arr)
        out[f"switch_rate_tau={tau}"] = float((pick != 0).mean())
    return out


def learning_curve(train: Arrays, val: Arrays, device: str, max_epochs: int) -> dict:
    subsets = nested_subsets(train)
    res: dict = {"subset_sizes": {str(f): len(i) for f, i in subsets.items()},
                 "subset_by_source": {
                     str(f): {s: int((train.meta["source"][i] == s).sum())
                              for s in sorted(set(train.meta["source"]))}
                     for f, i in subsets.items()},
                 "configs": {}}
    for name, (arch, loss, tau) in CONFIGS.items():
        taus = sorted({0.0, tau})
        rows = {}
        for f, idx in subsets.items():
            sub = train.subset(idx)
            mean, std = sub.h.mean(0), np.maximum(sub.h.std(0), 1e-6)
            per_seed = []
            for seed in SEEDS:
                r = fit(arch, loss, seed, sub, val, mean, std, device, max_epochs=max_epochs)
                dv = predict(r["head"], val, mean, std, device)
                dt = predict(r["head"], sub, mean, std, device)
                per_seed.append({"seed": seed, "best_epoch": r["best_epoch"],
                                 "val": head_metrics(dv, val, taus),
                                 "train": head_metrics(dt, sub, taus)})
            agg = {}
            for split in ("val", "train"):
                for key in per_seed[0][split]:
                    vals = [s[split][key] for s in per_seed]
                    agg[f"{split}/{key}"] = (
                        float(np.mean([v["mean"] for v in vals]))
                        if isinstance(vals[0], dict) else float(np.mean(vals)))
            rows[str(f)] = {"mean_over_seeds": agg, "per_seed": per_seed}
        # trend
        xs = np.array(FRACTIONS)
        trend = {}
        for key in (f"val/delta_tau={tau}", "val/delta_tau=0.0", "val/spearman"):
            ys = np.array([rows[str(f)]["mean_over_seeds"][key] for f in FRACTIONS])
            slope = float(np.polyfit(xs, ys, 1)[0])
            trend[key] = {"values": ys.tolist(), "slope_per_100pct": slope,
                          "upward": bool(ys[-1] > ys[0] and slope > 0),
                          "monotone_non_decreasing": bool(np.all(np.diff(ys) >= 0))}
        res["configs"][name] = {"tau_selected": tau, "rows": rows, "trend": trend}
    return res


def noise_block(arr: Arrays, y: np.ndarray, label: str) -> dict:
    a, b = np.nanmean(y[:, :, :8], axis=2), np.nanmean(y[:, :, 8:], axis=2)
    valid = arr.valid
    ng = valid.copy()
    ng[:, 0] = False
    da, db = (a - a[:, :1])[ng], (b - b[:, :1])[ng]
    r = float(np.corrcoef(da, db)[0, 1])
    full = (da + db) / 2
    noise_var = float(np.var(da - db) / 4)
    total = float(np.var(full))
    # within-root centred Q-hat across all candidates
    def centred(x: np.ndarray) -> np.ndarray:
        m = np.where(valid, x, np.nan)
        return (m - np.nanmean(m, axis=1, keepdims=True))[valid]

    ca, cb = centred(a), centred(b)
    c_noise = float(np.var(ca - cb) / 4)
    c_total = float(np.var((ca + cb) / 2))
    sp = np.array([spearman(a[i][valid[i]], b[i][valid[i]]) if valid[i].sum() >= 3
                   else np.nan for i in range(len(a))])
    ok = valid.sum(1) >= 3
    rbar = float(np.nan_to_num(sp)[ok].mean())
    rho = 2 * rbar / (1 + rbar) if rbar > -1 else float("nan")
    return {
        "label": label, "roots": len(a), "candidates_non_greedy": int(ng.sum()),
        "split_half_pearson_dq": r, "spearman_brown_reliability_dq": 2 * r / (1 + r),
        "dq_total_var_full": total, "dq_noise_var_full": noise_var,
        "dq_signal_fraction": max(0.0, 1 - noise_var / total),
        "centred_q_total_var": c_total, "centred_q_noise_var": c_noise,
        "centred_q_signal_fraction": max(0.0, 1 - c_noise / c_total),
        "split_half_spearman_mean_undefined0": rbar,
        "spearman_brown_reliability_rank": rho,
        "perfect_predictor_spearman_ceiling": float(math.sqrt(max(rho, 0.0))),
    }


def oracle_select(arr: Arrays, y: np.ndarray) -> dict:
    deltas, moved = [], []
    for i in range(len(y)):
        q = y[i][arr.valid[i]]
        d, m = M.select_delta(q)
        deltas.append(d)
        moved.append(m)
    deltas = np.array(deltas)
    insample = np.array([np.nanmax(np.nanmean(y[i][arr.valid[i]], axis=1)
                                   - np.nanmean(y[i][0])) for i in range(len(y))])
    return {"delta_select": ci_of(deltas, arr), "fraction_a_star_ne_a_c": float(np.mean(moved)),
            "in_sample_best_minus_a_c_optimistic": float(insample.clip(min=0).mean())}


def baselines(train: Arrays, val: Arrays, val_types: np.ndarray, tr_types: np.ndarray) -> dict:
    # B2 table: train-mean dQ by (type, phase) over non-greedy candidates
    cell: dict[tuple[str, str], list[float]] = {}
    for i in range(len(train.h)):
        for c in range(1, 4):
            if train.valid[i, c]:
                cell.setdefault((tr_types[i, c], train.meta["phase"][i]), []).append(
                    float(train.q[i, c] - train.q[i, 0]))
    table = {k: float(np.mean(v)) for k, v in cell.items()}
    q = np.nan_to_num(val.q)

    def pick_all(rule) -> np.ndarray:
        return np.array([rule(i) for i in range(len(q))])

    def b1(i: int) -> int:
        return 1 if val.valid[i, 1] else 0

    def b2(i: int) -> int:
        best, best_v = 0, 0.0
        for c in range(1, 4):
            if val.valid[i, c]:
                v = table.get((val_types[i, c], val.meta["phase"][i]), 0.0)
                if v > best_v:
                    best, best_v = c, v
        return best

    def b3(i: int) -> int:
        for c in range(1, 4):
            if val.valid[i, c] and val_types[i, c] == "move+tera":
                return c
        return 0

    out = {"train_cell_means": {f"{k[0]}|{k[1]}": v for k, v in table.items()}}
    for name, rule in (("second_highest_candidate", b1), ("train_mean_type_x_phase", b2),
                       ("tera_when_present", b3)):
        pick = pick_all(rule)
        gain = q[np.arange(len(q)), pick] - q[:, 0]
        block = {"all": ci_of(gain, val), "switch_rate": float((pick != 0).mean()),
                 "pass_gt0_ci_excl0": False}
        block["pass_gt0_ci_excl0"] = bool(block["all"]["mean"] > 0
                                          and block["all"]["ci95"][0] > 0)
        for src in ("sampled", "greedy"):
            sel = val.meta["source"] == src
            block[src] = ci_of(gain[sel], val.subset(sel))
        out[name] = block
    return out


def coverage(train: Arrays, val: Arrays, tr_types, val_types, head_delta: np.ndarray,
             tau: float) -> dict:
    reasons = train.meta["reasons"]
    counts_type, counts_reason, counts_tr = {}, {}, {}
    for i in range(len(train.h)):
        for c in range(4):
            if train.valid[i, c]:
                t = tr_types[i, c]
                counts_type[t] = counts_type.get(t, 0) + 1
                counts_reason[str(reasons[i, c])] = counts_reason.get(str(reasons[i, c]), 0) + 1
                key = f"{t}|{reasons[i, c]}"
                counts_tr[key] = counts_tr.get(key, 0) + 1
    # mean rollout dQ by type x phase / opponent (non-greedy candidates)
    flat = []
    for i in range(len(train.h)):
        for c in range(1, 4):
            if train.valid[i, c]:
                flat.append((tr_types[i, c], train.meta["phase"][i], train.meta["opponent"][i],
                             train.meta["pair"][i], float(train.q[i, c] - train.q[i, 0])))
    table = {}
    for axes, name in (((0, 1), "type_x_phase"), ((0, 2), "type_x_opponent"),
                       ((0,), "type"), ((0, 1, 2), "type_x_phase_x_opponent")):
        cells: dict[tuple, list] = {}
        for rec in flat:
            cells.setdefault(tuple(rec[a] for a in axes), []).append(rec)
        out = {}
        for key, recs in sorted(cells.items()):
            v = np.array([r[4] for r in recs])
            lo, hi = M.cluster_bootstrap(v, [r[3] for r in recs], 2000, 0)
            out["|".join(key)] = {"n": len(v), "mean_dq": float(v.mean()), "ci95": [lo, hi]}
        table[name] = out
    # what the head switches to on val
    gain, pick = root_gains(head_delta, val, tau)
    sw = pick != 0
    by_type = {}
    for t in ("move", "switch", "move+tera"):
        sel = np.array([sw[i] and val_types[i, pick[i]] == t for i in range(len(pick))])
        by_type[t] = {"switches": int(sel.sum()),
                      "mean_delta": float(gain[sel].mean()) if sel.any() else None}
    by_reason = {}
    for r in sorted({str(x) for x in reasons.ravel() if x}):
        sel = np.array([sw[i] and str(val.meta["reasons"][i, pick[i]]) == r
                        for i in range(len(pick))])
        by_reason[r] = {"switches": int(sel.sum()),
                        "mean_delta": float(gain[sel].mean()) if sel.any() else None}
    return {"train_candidates_by_type": counts_type, "train_candidates_by_reason": counts_reason,
            "train_candidates_by_type_x_reason": counts_tr,
            "train_greedy_action_type": {
                t: int(sum(1 for i in range(len(train.h)) if tr_types[i, 0] == t))
                for t in ("move", "switch", "move+tera")},
            "mean_dq_nongreedy": table,
            "head_switches_on_val": {"tau": tau, "by_type": by_type, "by_reason": by_reason}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", default="checkpoints/w5_cf1_5")
    ap.add_argument("--out", default="reports/w5_cf1_5b_diagnostics.json")
    ap.add_argument("--max-epochs", type=int, default=60)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_arrays(args.dataset)
    z = np.load(args.dataset, allow_pickle=False)
    y_all = z["y"]
    split = data.meta["split"]
    train, val = data.subset(split == "train"), data.subset(split == "val")
    y_tr, y_val = y_all[split == "train"], y_all[split == "val"]
    tr_types, val_types = types_of(train), types_of(val)

    cfg = json.loads((Path(args.ckpt) / "config.json").read_text())
    head = make_head(cfg["arch"], len(cfg["h_mean"]))
    head.load_state_dict(torch.load(Path(args.ckpt) / "head.pt", map_location="cpu",
                                    weights_only=True))
    head.to(device)
    mean, std = np.array(cfg["h_mean"], np.float32), np.array(cfg["h_std"], np.float32)
    saved_delta = predict(head, val, mean, std, device)

    report: dict = {"n_train": len(train.h), "n_val": len(val.h),
                    "saved_head": {k: cfg[k] for k in ("arch", "loss", "tau", "seed")}}
    report["learning_curve"] = learning_curve(train, val, device, args.max_epochs)
    noise = {"val": noise_block(val, y_val, "val"), "train": noise_block(train, y_tr, "train")}
    for src in ("sampled", "greedy"):
        sel = val.meta["source"] == src
        noise[f"val_{src}"] = noise_block(val.subset(sel), y_val[sel], f"val_{src}")
    report["noise"] = noise
    oracle = {"pooled": oracle_select(val, y_val)}
    for src in ("sampled", "greedy"):
        sel = val.meta["source"] == src
        oracle[src] = oracle_select(val.subset(sel), y_val[sel])
    base = baselines(train, val, val_types, tr_types)
    gain_head, _ = root_gains(saved_delta, val, cfg["tau"])
    lp_sp = np.array([spearman(val.logp[i][val.valid[i]], val.q[i][val.valid[i]])
                      if val.valid[i].sum() >= 3 else np.nan for i in range(len(val.h))])
    ok = val.valid.sum(1) >= 3
    orc = oracle["pooled"]["delta_select"]["mean"]
    report["label_oracle"] = oracle | {
        "logp_only_spearman_vs_labels": ci_of(np.nan_to_num(lp_sp)[ok], val.subset(ok)),
        "captured_fraction_of_oracle_delta_select": {
            "logp_only_second_highest": base["second_highest_candidate"]["all"]["mean"] / orc,
            "saved_head": float(gain_head.mean()) / orc,
        },
    }
    report["coverage"] = coverage(train, val, tr_types, val_types, saved_delta, cfg["tau"])
    report["baselines"] = base
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
