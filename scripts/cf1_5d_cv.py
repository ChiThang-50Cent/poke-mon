"""CF1.5d: one-shot 5-fold CV (grouped by team pair) of the semantic action head, train roots only.

Primary F2 (locked), baselines F0/F1/logpi, matched control F2_shuffled. Recipe: MLP 128,
pairwise loss, exactly 3 epochs, seeds 1-3, all fitted statistics from the training folds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from cf1_5c_train import (
    C_DIR,
    FULL_RUN,
    embeddings,
    load_descriptors,
    paired,
)

from pokeai.cf.head import load_arrays
from pokeai.cf.head_sem import SemData, fit_sem, predict_sem, root_spearman
from pokeai.cf.head_train import pooled_ci, root_gains
from pokeai.cf.signal_stats import spearman

FOLD_SEED = 20261030
SEEDS = (1, 2, 3)
EPOCHS = 3
K = 5


def make_folds(arr) -> np.ndarray:
    pairs = arr.meta["pair"]
    strata = np.array([f"{s}|{o}" for s, o in zip(arr.meta["source"], arr.meta["opponent"])])
    names = sorted(set(strata))
    target = {k: float((strata == k).sum()) / K for k in names}
    count = [dict.fromkeys(names, 0) for _ in range(K)]
    size = [0] * K
    fold_of = {}
    uniq = sorted(set(pairs))
    for i in np.random.default_rng(FOLD_SEED).permutation(len(uniq)):
        p = uniq[i]
        mine = strata[pairs == p]
        add = {k: int((mine == k).sum()) for k in set(mine)}
        costs = []
        for f in range(K):
            c = sum(abs(count[f][k] + v - target[k]) - abs(count[f][k] - target[k])
                    for k, v in add.items())
            costs.append((c, size[f], f))
        f = min(costs)[2]
        fold_of[p] = f
        size[f] += len(mine)
        for k, v in add.items():
            count[f][k] += v
    return np.array([fold_of[p] for p in pairs])


def logp_sp(arr) -> np.ndarray:
    out = np.full(len(arr.h), np.nan)
    for i in range(len(arr.h)):
        n = int(arr.valid[i].sum())
        if n >= 3:
            out[i] = spearman(arr.logp[i, :n], arr.q[i, :n])
    return out


def two_cand_sign(delta, arr):
    ok, hit = [], []
    for i in range(len(arr.h)):
        if arr.valid[i].sum() == 2:
            truth = arr.q[i, 1] - arr.q[i, 0]
            if truth != 0:
                ok.append(i)
                hit.append(float(np.sign(delta[i, 1]) == np.sign(truth)))
    return {"roots": len(ok), "sign_accuracy": float(np.mean(hit)) if hit else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{FULL_RUN}/dataset.npz")
    ap.add_argument("--desc-dir", default="runs/w5-cf1-5c")
    ap.add_argument("--out-json", default="reports/w5_cf1_5d_cv.json")
    ap.add_argument("--ckpt-dir", default="checkpoints/w5_cf1_5d")
    args = ap.parse_args()
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_arrays(args.dataset)
    arr = full.subset(full.meta["split"] == "train")
    zx, cand_ids = load_descriptors(arr, args.desc_dir)
    emb, emb_cov = embeddings(cand_ids)
    fold = make_folds(arr)
    teams = lambda m: {t for p in set(arr.meta["pair"][m]) for t in p.split("|")}
    fold_info = {"seed": FOLD_SEED, "sizes": [int((fold == f).sum()) for f in range(K)],
                 "pairs": [len(set(arr.meta["pair"][fold == f])) for f in range(K)],
                 "pair_overlap": sum(len(set(arr.meta["pair"][fold == a])
                                         & set(arr.meta["pair"][fold == b]))
                                     for a in range(K) for b in range(a + 1, K)),
                 "team_overlap_fraction_of_eval_teams_in_train": [
                     len(teams(fold == f) & teams(fold != f)) / len(teams(fold == f))
                     for f in range(K)]}
    print(fold_info, flush=True)
    n = len(arr.h)
    variants = ("F0", "F1", "F2", "F2_shuffled")
    oof = {v: [np.zeros((n, 4), dtype=np.float32) for _ in SEEDS] for v in variants}
    for f in range(K):
        tr, ev = fold != f, fold == f
        mean = arr.h[tr].mean(0)
        std = np.maximum(arr.h[tr].std(0), 1e-6)
        rows = emb[tr][arr.valid[tr]]
        e_mean, e_std = rows.mean(0), np.maximum(rows.std(0), 1e-6)
        emb_n = np.where(arr.valid[:, :, None], (emb - e_mean) / e_std, 0.0).astype(np.float32)
        x1 = zx
        x2 = np.concatenate([zx, emb_n], axis=2)
        for v in variants:
            data = SemData(arr, x1 if v in ("F0", "F1") else x2)
            for si, seed in enumerate(SEEDS):
                r = fit_sem(seed, data.subset(tr), None, mean, std, device,
                            shuffle_labels=(v == "F2_shuffled"), fixed_epochs=EPOCHS,
                            use_x=(v != "F0"), perm_seed=1000 * seed + f + 77)
                d = predict_sem(r["head"], SemData(data.arr, data.x[:, :, :r["dim_x"]]).subset(ev),
                                mean, std, device)
                oof[v][si][ev] = d
        print(f"fold {f} done", flush=True)
    assert all(np.isfinite(o).all() for v in oof.values() for o in v)
    m3 = arr.valid.sum(1) >= 3
    ens = {v: np.mean(oof[v], axis=0) for v in variants}
    sp = {v: np.nan_to_num(root_spearman(ens[v], arr)) for v in variants}
    sp["logp"] = np.nan_to_num(logp_sp(arr))
    per_seed = {v: [np.nan_to_num(root_spearman(o, arr))[m3].mean() for o in oof[v]]
                for v in variants}

    def sub(a, mask):
        return a.subset(mask)

    def diff(a, b, mask=None):
        mk = m3 if mask is None else (m3 & mask)
        x, y = sp[a].copy(), sp[b].copy()
        x[~mk], y[~mk] = np.nan, np.nan
        return paired(x, y, arr)

    out = {"fold": fold_info, "embedding_coverage": emb_cov, "epochs": EPOCHS,
           "seeds": list(SEEDS), "roots": n, "roots_ge3": int(m3.sum()),
           "primary_mean_spearman_undefined0": {k: float(v[m3].mean()) for k, v in sp.items()},
           "secondary": {
               "per_seed_mean_spearman": {v: [float(x) for x in per_seed[v]] for v in variants},
               "defined_only_mean": {v: float(np.nanmean(np.where(
                   m3, root_spearman(ens[v], arr), np.nan))) for v in variants},
               "ensemble_delta_head_tau0": {
                   v: pooled_ci(root_gains(ens[v], arr, 0.0)[0], arr) for v in variants},
               "switch_rate_tau0": {v: float((root_gains(ens[v], arr, 0.0)[1] != 0).mean())
                                    for v in variants}}}
    out["gates"] = {
        "CV-G1_F2_minus_F0": diff("F2", "F0"), "CV-G1_F2_minus_logp": diff("F2", "logp"),
        "CV-G2_F2_minus_F2_shuffled": diff("F2", "F2_shuffled"),
    }
    g = out["gates"]
    g["CV-G1"] = bool(g["CV-G1_F2_minus_F0"]["gt0_ci_excl0"]
                      and g["CV-G1_F2_minus_logp"]["gt0_ci_excl0"])
    g["CV-G2"] = bool(g["CV-G2_F2_minus_F2_shuffled"]["gt0_ci_excl0"])
    g["reliability"] = {"oof_complete": True, "finite": True, "folds": K,
                        "seeds": len(SEEDS), "variants": list(variants)}
    meta = arr.meta
    groups = {
        "source": {s: meta["source"] == s for s in ("pilot", "sampled", "greedy")},
        "opponent": {s: meta["opponent"] == s for s in ("C", "Grunt", "GymLeader")},
        "phase": {s: meta["phase"] == s for s in ("0-5", "6-15", "16-30", "31+")},
    }
    diag = {"F1_minus_F0": diff("F1", "F0"), "F2_minus_F1": diff("F2", "F1"), "by_group": {}}
    for gname, d in groups.items():
        for k, mk in d.items():
            m = mk & m3
            if m.sum() < 5:
                continue
            diag["by_group"][f"{gname}={k}"] = {
                "roots_ge3": int(m.sum()),
                "means": {kk: float(v[m].mean()) for kk, v in sp.items()},
                "F2_minus_F0": diff("F2", "F0", mk), "F2_minus_logp": diff("F2", "logp", mk),
                "F2_minus_shuffled": diff("F2", "F2_shuffled", mk)}
    diag["two_candidate_roots"] = {v: two_cand_sign(ens[v], arr) for v in variants}
    diag["per_fold_mean_spearman"] = {
        f"fold{f}": {"roots_ge3": int((m3 & (fold == f)).sum()),
                     **{k: float(v[m3 & (fold == f)].mean()) for k, v in sp.items()}}
        for f in range(K)}
    out["diagnostics"] = diag
    out["note"] = ("CIs reflect this development design only (overlapping training folds, "
                   "teams recurring across pairs), not a fresh-set confirmation.")
    if g["CV-G1"] and g["CV-G2"]:
        mean, std = arr.h.mean(0), np.maximum(arr.h.std(0), 1e-6)
        rows = emb[arr.valid]
        e_mean, e_std = rows.mean(0), np.maximum(rows.std(0), 1e-6)
        emb_n = np.where(arr.valid[:, :, None], (emb - e_mean) / e_std, 0.0).astype(np.float32)
        data = SemData(arr, np.concatenate([zx, emb_n], axis=2))
        ck = Path(args.ckpt_dir)
        ck.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            r = fit_sem(s, data, None, mean, std, device, fixed_epochs=EPOCHS)
            torch.save(r["head"].state_dict(), ck / f"head_seed{s}.pt")
        (ck / "config.json").write_text(json.dumps({
            "variant": "F2", "tau": 0.0, "epochs": EPOCHS, "seeds": list(SEEDS),
            "h_mean": mean.tolist(), "h_std": std.tolist(), "emb_mean": e_mean.tolist(),
            "emb_std": e_std.tolist(), "c_dir": C_DIR}, indent=1))
        out["locked"] = {"checkpoint_dir": str(ck), "roots": n}
    out["wall_s"] = time.perf_counter() - started
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    print(json.dumps({"primary": out["primary_mean_spearman_undefined0"],
                      "gates": {k: g[k] for k in ("CV-G1", "CV-G2")}}))


if __name__ == "__main__":
    main()
