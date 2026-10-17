"""CF1.3 statistics: ranking (G3) and selectability (G4) from world-split halves.

q has shape (candidates, 16); rollout k = 2*world + seed so H1 = worlds 0..3 = k<8 and
H2 = worlds 4..7 = k>=8. Candidate 0 is always C's greedy action (a_C).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pokeai.cf.signal_stats import permute_within_k, spearman

HALF = 8


def world_halves(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return q[:, :HALF], q[:, HALF:]


def root_spearman(q: np.ndarray) -> float:
    h1, h2 = world_halves(q)
    return spearman(h1.mean(axis=1), h2.mean(axis=1))


def cluster_bootstrap(
    values: np.ndarray, clusters: list[str], n: int, seed: int
) -> tuple[float, float]:
    """95% CI of the pooled mean, resampling whole clusters (team pairs)."""
    names = sorted(set(clusters))
    sums = np.array([values[[c == x for c in clusters]].sum() for x in names])
    counts = np.array([sum(c == x for c in clusters) for x in names], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), size=(n, len(names)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def g3_ranking(
    qs: list[np.ndarray], clusters: list[str], n_boot: int = 5000, n_null: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    keep = [i for i, q in enumerate(qs) if len(q) >= 3]
    if not keep:
        return {"roots": 0, "pass": False}
    qk = [qs[i] for i in keep]
    ck = [clusters[i] for i in keep]
    stats = np.array([root_spearman(q) for q in qk])
    primary = np.nan_to_num(stats, nan=0.0)
    defined = stats[~np.isnan(stats)]
    lo, hi = cluster_bootstrap(primary, ck, n_boot, seed)
    rng = np.random.default_rng(seed + 1)
    null = np.array([
        float(np.mean([
            np.nan_to_num(root_spearman(permute_within_k(q, rng)), nan=0.0) for q in qk
        ]))
        for _ in range(n_null)
    ])
    real = float(primary.mean())
    p95 = float(np.percentile(null, 95))
    top1, chance = [], []
    for q in qk:
        a, b = (x.mean(axis=1) for x in world_halves(q))
        best_a = np.flatnonzero(a == a.max())
        best_b = np.flatnonzero(b == b.max())
        overlap = len(set(best_a) & set(best_b))
        top1.append(overlap / (len(best_a) * len(best_b)))
        chance.append(1 / len(q))
    out = {
        "roots": len(qk), "mean_undefined_as_0": real, "ci95": [lo, hi],
        "mean_defined_only": float(defined.mean()) if len(defined) else None,
        "n_defined": len(defined),
        "null_mean": float(null.mean()), "null_p95": p95,
        "null_p_value": float((1 + (null >= real - 1e-12).sum()) / (1 + len(null))),
        "top1_agreement": float(np.mean(top1)), "top1_chance": float(np.mean(chance)),
    }
    out["pass"] = bool(lo > 0 and real > p95)
    return out


def select_delta(q: np.ndarray) -> tuple[float, float]:
    """(delta averaged over both split directions, fraction of directions a* != a_C)."""
    h1, h2 = world_halves(q)
    deltas, moved = [], 0
    for sel, ev in ((h1, h2), (h2, h1)):
        means = sel.mean(axis=1)
        best = np.flatnonzero(means == means.max())
        star = 0 if 0 in best else int(best[0])
        deltas.append(float(ev[star].mean() - ev[0].mean()))
        moved += int(star != 0)
    return float(np.mean(deltas)), moved / 2


def _summary(
    deltas: np.ndarray, clusters: list[str], n_boot: int, seed: int
) -> dict[str, Any]:
    if len(deltas) == 0:
        return {"roots": 0}
    lo, hi = cluster_bootstrap(deltas, clusters, n_boot, seed)
    return {
        "roots": len(deltas), "delta": float(deltas.mean()), "ci95": [lo, hi],
        "pass": bool(deltas.mean() > 0 and lo > 0),
    }


def g4_select(
    qs: list[np.ndarray], meta: list[dict[str, Any]], n_boot: int = 5000, seed: int = 0
) -> dict[str, Any]:
    pairs = [select_delta(q) if len(q) >= 2 else (0.0, 0.0) for q in qs]
    deltas = np.array([p[0] for p in pairs])
    moved = np.array([p[1] for p in pairs])
    clusters = [m["pair"] for m in meta]
    out = _summary(deltas, clusters, n_boot, seed)
    out["fraction_a_star_ne_a_c"] = float(moved.mean()) if len(moved) else None
    groups: dict[str, dict[str, Any]] = {}
    for key in ("opponent", "phase", "split", "forced_switch"):
        for value in sorted({str(m[key]) for m in meta}):
            idx = [i for i, m in enumerate(meta) if str(m[key]) == value]
            groups[f"{key}={value}"] = {
                **_summary(deltas[idx], [clusters[i] for i in idx], n_boot, seed),
                "fraction_a_star_ne_a_c": float(moved[idx].mean()),
            }
    out["by_group"] = groups
    return out


def delta_precision(q: np.ndarray) -> list[dict[str, float]]:
    """Per non-a_C candidate: dQ-hat, paired SE over the 16 rollouts, 95% CI width."""
    rows = []
    for c in range(1, len(q)):
        diff = q[c] - q[0]
        se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
        rows.append({"candidate": c, "dq": float(diff.mean()), "se": se,
                     "ci_width": 3.92 * se})
    return rows


def oracle_agreement(
    belief: list[np.ndarray], oracle: list[np.ndarray]
) -> dict[str, Any]:
    b, o = [], []
    for qb, qo in zip(belief, oracle, strict=True):
        for c in range(1, min(len(qb), len(qo))):
            b.append(qb[c].mean() - qb[0].mean())
            o.append(qo[c].mean() - qo[0].mean())
    b, o = np.array(b), np.array(o)
    both = (b != 0) & (o != 0)
    corr = float(np.corrcoef(b, o)[0, 1]) if len(b) > 2 and b.std() and o.std() else None
    return {
        "pairs": len(b), "pearson": corr,
        "sign_agreement": float((np.sign(b[both]) == np.sign(o[both])).mean())
        if both.any() else None,
        "pairs_both_nonzero": int(both.sum()),
    }


def load_run(run_dir: str) -> list[dict[str, Any]]:
    """Per-root records of a cf1_run.py output directory, in root order."""
    import json
    from pathlib import Path

    rows = [json.loads(x) for x in (Path(run_dir) / "roots.jsonl").read_text().splitlines()]
    last = {r["root"]: r for r in rows}  # re-run records supersede earlier errors
    return sorted(last.values(), key=lambda r: r["root"])


def variant_q(record: dict[str, Any], variant: str) -> np.ndarray | None:
    data = record.get("variants", {}).get(variant)
    return None if data is None else np.array(data["outcomes"], float)


def root_meta(record: dict[str, Any], split: str = "all") -> dict[str, Any]:
    return {
        "pair": record["pair"], "opponent": record["opponent"],
        "phase": record["phase"], "forced_switch": record["forced_switch"],
        "split": split,
    }


def assign_split(pairs: list[str], seed: int, val_fraction: float = 0.2) -> dict[str, str]:
    """Seeded train/val split by team pair (all roots of a pair share a partition)."""
    unique = sorted(set(pairs))
    order = np.random.default_rng(seed).permutation(len(unique))
    n_val = round(val_fraction * len(unique))
    val = {unique[i] for i in order[:n_val]}
    return {p: "val" if p in val else "train" for p in unique}
