"""CF1.5c-B/C: F0/F1/F2 semantic-head variants on an internal train/val split (train roots only).

Gates F-G1 (selected variant beats F0 and logp-only) and F-G2 (beats shuffled-label control),
paired per-root Spearman, team-pair cluster bootstrap. If both pass: lock (variant, tau, epochs)
and retrain on all train roots, 3 seeds. No fresh roots are touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pokeai.cf.cf1_metrics import cluster_bootstrap
from pokeai.cf.descriptors import DIM, FEATURE_NAMES
from pokeai.cf.head import TAUS, load_arrays
from pokeai.cf.head_sem import SemData, fit_sem, predict_sem, root_spearman
from pokeai.cf.head_train import pooled_ci, root_gains
from pokeai.cf.signal_stats import spearman

SEEDS = (1, 2, 3)
SPLIT_SEED = 20261022
C_DIR = "models/pokedecision-v1-value"
FULL_RUN = "runs/w5-cf1-4-20261017-104148"


def internal_split(arr, frac: float = 0.2) -> np.ndarray:
    """Boolean internal-val mask: whole team pairs, stratified by source x opponent."""
    pairs = arr.meta["pair"]
    strata = np.array([f"{s}|{o}" for s, o in zip(arr.meta["source"], arr.meta["opponent"])])
    uniq = sorted(set(pairs))
    order = np.random.default_rng(SPLIT_SEED).permutation(len(uniq))
    names = sorted(set(strata))
    total = {k: int((strata == k).sum()) for k in names}
    target = {k: frac * v for k, v in total.items()}
    count = dict.fromkeys(names, 0)
    val_pairs = set()
    for i in order:
        p = uniq[i]
        mine = strata[pairs == p]
        add = {k: int((mine == k).sum()) for k in set(mine)}
        before = sum(abs(count[k] - target[k]) for k in add)
        after = sum(abs(count[k] + add[k] - target[k]) for k in add)
        if after < before:
            val_pairs.add(p)
            for k, v in add.items():
                count[k] += v
    return np.array([p in val_pairs for p in pairs])


def load_descriptors(arr, desc_dir: str):
    z = np.load(Path(desc_dir) / "descriptors.npz")
    ids = json.loads((Path(desc_dir) / "ids.json").read_text())
    index = {f"{g}:{r}": i for i, (g, r) in enumerate(zip(z["group"], z["root"]))}
    n = len(arr.h)
    zx = np.zeros((n, 4, DIM), dtype=np.float32)
    cand_ids: list[list] = []
    for i in range(n):
        key = f"{arr.meta['group'][i]}:{arr.meta['root'][i]}"
        j = index[key]
        row_ids = []
        for c in range(4):
            slot = int(arr.slots[i, c])
            if slot < 0:
                row_ids.append(None)
                continue
            zx[i, c] = z["x"][j, slot]
            row_ids.append(ids[key][slot])
        cand_ids.append(row_ids)
    return zx, cand_ids


def embeddings(cand_ids):
    """Frozen C token embedding of the candidate's move / incoming species + mapping stats."""
    from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name

    from pokeai.models.pokedecision import load_pokedecision

    model, encoder, _ = load_pokedecision(C_DIR, "cpu")
    table = model.tokens.weight.detach().float().numpy()
    out = np.zeros((len(cand_ids), 4, table.shape[1]), dtype=np.float32)
    stats = {"move": [0, 0], "species": [0, 0]}
    unknown: dict[str, int] = {}
    for i, row in enumerate(cand_ids):
        for c, item in enumerate(row):
            if item is None:
                continue
            kind, name = item
            word = move_name(name) if kind == "move" else pokemon_name(name)
            tok = encoder.tokenizer.tokenize(word)
            tok = int(np.asarray(tok).reshape(-1)[0])
            stats[kind][0] += 1
            if tok < 0:
                stats[kind][1] += 1
                unknown[f"{kind}:{name}"] = unknown.get(f"{kind}:{name}", 0) + 1
                tok = encoder.unknown_id
            out[i, c] = table[tok]
    cov = {k: {"candidates": v[0], "unknown": v[1],
               "unknown_rate": v[1] / max(v[0], 1)} for k, v in stats.items()}
    cov["unknown_names"] = dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:30])
    cov["d_model"] = int(table.shape[1])
    cov["vocab_unknown_id"] = int(encoder.unknown_id)
    return out, cov


def paired(a: np.ndarray, b: np.ndarray, arr, seed: int = 0) -> dict:
    """Paired per-root difference a - b over roots where both are defined."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    diff = (a - b)[ok]
    lo, hi = cluster_bootstrap(diff, list(arr.meta["pair"][ok]), 5000, seed)
    return {"mean_diff": float(diff.mean()), "ci95": [lo, hi], "n_roots": int(ok.sum()),
            "gt0_ci_excl0": bool(diff.mean() > 0 and lo > 0)}


def logp_spearman(arr) -> np.ndarray:
    out = np.full(len(arr.h), np.nan)
    for i in range(len(arr.h)):
        n = int(arr.valid[i].sum())
        if n >= 3:
            out[i] = spearman(arr.logp[i, :n], arr.q[i, :n])
    return out


def run_variant(name, train, val, mean, std, device, use_x, shuffle=False):
    runs = []
    for seed in SEEDS:
        r = fit_sem(seed, train, val, mean, std, device, shuffle_labels=shuffle, use_x=use_x)
        runs.append(r)
        print(f"{name} seed {seed}: epoch {r['best_epoch']} val_spearman "
              f"{r['best_val_spearman']:.4f}", flush=True)
    return runs


def evaluate(runs, val, mean, std, device, use_x):
    dim_x = runs[0]["dim_x"]
    deltas = [predict_sem(r["head"], SemData(val.arr, val.x[:, :, :dim_x]), mean, std, device)
              for r in runs]
    sp = np.stack([root_spearman(d, val.arr) for d in deltas])
    ens = np.mean(deltas, axis=0)
    out = {"seed_best_epochs": [r["best_epoch"] for r in runs],
           "seed_val_spearman_selected": [r["best_val_spearman"] for r in runs],
           "per_seed_mean_spearman": [float(np.nanmean(s)) for s in sp],
           "per_root_spearman_seedmean": np.nanmean(sp, axis=0)}
    out["gain"] = {}
    for tau in TAUS:
        per_seed = []
        for d in deltas:
            g, pick = root_gains(d, val.arr, tau)
            per_seed.append({"delta": float(g.mean()), "switch_rate": float((pick != 0).mean())})
        g, pick = root_gains(ens, val.arr, tau)
        out["gain"][str(tau)] = {
            "per_seed": per_seed,
            "ensemble": {**pooled_ci(g, val.arr), "switch_rate": float((pick != 0).mean())},
        }
    out["ensemble_delta"] = ens
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{FULL_RUN}/dataset.npz")
    ap.add_argument("--desc-dir", default="runs/w5-cf1-5c")
    ap.add_argument("--out-json", default="reports/w5_cf1_5c_internal_val.json")
    ap.add_argument("--ckpt-dir", default="checkpoints/w5_cf1_5c")
    args = ap.parse_args()
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_arrays(args.dataset)
    tr_all = full.subset(full.meta["split"] == "train")
    zx, cand_ids = load_descriptors(tr_all, args.desc_dir)
    emb, emb_cov = embeddings(cand_ids)
    is_val = internal_split(tr_all)
    t_pairs = {p for p in tr_all.meta["pair"][~is_val]}
    v_pairs = {p for p in tr_all.meta["pair"][is_val]}
    teams = lambda m: {t for p in set(tr_all.meta["pair"][m]) for t in p.split("|")}
    split_info = {
        "internal_train_roots": int((~is_val).sum()), "internal_val_roots": int(is_val.sum()),
        "pair_overlap": len(t_pairs & v_pairs),
        "team_overlap": len(teams(~is_val) & teams(is_val)),
        "val_teams": len(teams(is_val)), "train_teams": len(teams(~is_val)),
        "by_source_opponent_val": {
            k: int(((np.array([f'{s}|{o}' for s, o in zip(tr_all.meta['source'],
                                                          tr_all.meta['opponent'])]) == k)
                    & is_val).sum())
            for k in sorted({f'{s}|{o}' for s, o in zip(tr_all.meta['source'],
                                                         tr_all.meta['opponent'])})},
        "seed": SPLIT_SEED,
    }
    print(split_info)
    # F2 standardisation: train-only statistics of the candidate embedding rows
    rows = emb[~is_val][tr_all.valid[~is_val]]
    e_mean, e_std = rows.mean(0), np.maximum(rows.std(0), 1e-6)
    emb_n = np.where(tr_all.valid[:, :, None], (emb - e_mean) / e_std, 0.0).astype(np.float32)
    x_f1 = zx
    x_f2 = np.concatenate([zx, emb_n], axis=2)
    d1 = SemData(tr_all, x_f1)
    d2 = SemData(tr_all, x_f2)
    sel_tr, sel_val = ~is_val, is_val
    mean = tr_all.h[sel_tr].mean(0)
    std = np.maximum(tr_all.h[sel_tr].std(0), 1e-6)

    results = {}
    ctx = {"F0": (d1, False, False), "F1": (d1, True, False), "F2": (d2, True, False),
           "F1_shuffled": (d1, True, True)}
    fitted = {}
    for name, (data, use_x, shuffle) in ctx.items():
        runs = run_variant(name, data.subset(sel_tr), data.subset(sel_val), mean, std,
                           device, use_x, shuffle)
        fitted[name] = runs
        results[name] = evaluate(runs, data.subset(sel_val), mean, std, device, use_x)
    val = d1.subset(sel_val)
    base = logp_spearman(val.arr)
    sp_mean = {k: float(np.nanmean(v["per_root_spearman_seedmean"])) for k, v in results.items()}
    sp_mean["logp_only"] = float(np.nanmean(base))
    selected = "F2" if sp_mean["F2"] > sp_mean["F1"] else "F1"
    sel = results[selected]["per_root_spearman_seedmean"]
    gates = {
        "selected_variant": selected, "mean_per_root_spearman": sp_mean,
        "F-G1_vs_F0": paired(sel, results["F0"]["per_root_spearman_seedmean"], val.arr),
        "F-G1_vs_logp_only": paired(sel, base, val.arr),
        "F-G2_vs_shuffled_control": paired(
            sel, results["F1_shuffled"]["per_root_spearman_seedmean"], val.arr),
        "also_F1_vs_F0": paired(results["F1"]["per_root_spearman_seedmean"],
                                results["F0"]["per_root_spearman_seedmean"], val.arr),
        "also_F1_vs_shuffled": paired(results["F1"]["per_root_spearman_seedmean"],
                                      results["F1_shuffled"]["per_root_spearman_seedmean"],
                                      val.arr),
        "also_F2_vs_F1": paired(results["F2"]["per_root_spearman_seedmean"],
                                results["F1"]["per_root_spearman_seedmean"], val.arr),
        "roots_with_defined_spearman": int((~np.isnan(base)).sum()),
    }
    gates["F-G1"] = bool(gates["F-G1_vs_F0"]["gt0_ci_excl0"]
                         and gates["F-G1_vs_logp_only"]["gt0_ci_excl0"])
    gates["F-G2"] = bool(gates["F-G2_vs_shuffled_control"]["gt0_ci_excl0"])
    # tau on internal val: seed-mean (ensemble) dQ, best Delta over the grid
    tau_grid = {t: results[selected]["gain"][t]["ensemble"]["mean"] for t in
                results[selected]["gain"]}
    tau_sel = float(max(tau_grid, key=lambda k: tau_grid[k]))
    epochs = int(np.median([r["best_epoch"] for r in fitted[selected]]))
    lock = {"variant": selected, "tau": tau_sel, "epochs": epochs,
            "tau_grid_ensemble_delta": tau_grid}
    out = {
        "split": split_info, "embedding_coverage": emb_cov, "feature_dim": DIM,
        "feature_names": FEATURE_NAMES, "seeds": list(SEEDS),
        "variants": {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ("per_root_spearman_seedmean", "ensemble_delta")}
            for k, v in results.items()},
        "gates": gates, "lock": lock,
        "internal_val_label_stats": {
            "roots": int(sel_val.sum()),
            "oracle_dQ_best_nongreedy_mean": float(
                np.maximum(val.arr.d[:, 1:].max(1), 0).mean()),
        },
    }
    if gates["F-G1"] and gates["F-G2"]:
        data = d2 if selected == "F2" else d1
        full_h = tr_all
        mean_all = full_h.h.mean(0)
        std_all = np.maximum(full_h.h.std(0), 1e-6)
        locked = [fit_sem(s, data, None, mean_all, std_all, device, fixed_epochs=epochs)
                  for s in SEEDS]
        ck = Path(args.ckpt_dir)
        ck.mkdir(parents=True, exist_ok=True)
        for s, r in zip(SEEDS, locked):
            torch.save(r["head"].state_dict(), ck / f"head_seed{s}.pt")
        (ck / "config.json").write_text(json.dumps({
            **lock, "h_mean": mean_all.tolist(), "h_std": std_all.tolist(),
            "emb_mean": e_mean.tolist(), "emb_std": e_std.tolist(), "feature_dim": DIM,
            "seeds": list(SEEDS)}, indent=1))
        out["locked"] = {"checkpoint_dir": str(ck), "retrained_on_roots": len(full_h.h)}
    out["wall_s"] = time.perf_counter() - started
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    print(json.dumps({"gates": {k: gates[k] for k in ("F-G1", "F-G2")}, "lock": lock,
                      "spearman": sp_mean}))


if __name__ == "__main__":
    main()
