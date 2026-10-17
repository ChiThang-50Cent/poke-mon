"""CF0.4 statistics: within-k permutation tests and split-half ranking stability.

`Q` for a root is an array [m candidates, K paired rollouts] of terminal outcomes in
{0, 0.5, 1}. Rollout k shares sim/opponent/learner seeds across candidates, so the null
permutes candidate labels independently inside each column k (pairing preserved).
"""

from __future__ import annotations

import math

import numpy as np


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho; NaN when either side is constant (undefined)."""
    ra, rb = rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = math.sqrt(float((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def statistic_T(q: np.ndarray) -> float:
    """Variance (ddof 0) across candidates of the per-candidate mean over k."""
    return float(np.var(q.mean(axis=1)))


def permute_within_k(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute candidate labels inside every column."""
    order = np.argsort(rng.random(q.shape), axis=0)
    return np.take_along_axis(q, order, axis=0)


def permuted_T(q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([statistic_T(permute_within_k(q, rng)) for _ in range(n)])


def permutation_p(observed: float, null: np.ndarray) -> float:
    return float((1 + (null >= observed - 1e-12).sum()) / (1 + len(null)))


def root_and_global_tests(
    roots: list[np.ndarray], n_perm: int, seed: int
) -> dict:
    """Per-root p-values and the global sum-of-T test (same permutation draws)."""
    rng = np.random.default_rng(seed)
    observed = np.array([statistic_T(q) for q in roots])
    nulls = np.stack([permuted_T(q, n_perm, rng) for q in roots])  # [roots, perms]
    p_roots = np.array([permutation_p(o, n) for o, n in zip(observed, nulls)])
    global_obs = float(observed.sum())
    global_null = nulls.sum(axis=0)
    return {
        "T": observed, "p": p_roots, "T_global": global_obs,
        "p_global": permutation_p(global_obs, global_null),
        "T_global_null_mean": float(global_null.mean()),
    }


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def half_estimates(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Even-k and odd-k candidate means (8 vs 8 for K=16)."""
    return q[:, 0::2].mean(axis=1), q[:, 1::2].mean(axis=1)


def split_half_stat(q: np.ndarray) -> float:
    """Spearman for >=3 candidates; sign-agreement (+1/-1/0) for 2 candidates."""
    a, b = half_estimates(q)
    if len(a) >= 3:
        return spearman(a, b)
    da, db = a[0] - a[1], b[0] - b[1]
    return float(np.sign(da) * np.sign(db))


def top1_agreement(q: np.ndarray, rng: np.random.Generator) -> float:
    """Whether halves pick the same top candidate (ties broken at random)."""
    a, b = half_estimates(q)
    pick = []
    for x in (a, b):
        best = np.flatnonzero(x == x.max())
        pick.append(int(rng.choice(best)))
    return float(pick[0] == pick[1])


def mean_with_undefined(values: list[float]) -> tuple[float, float, int]:
    """(mean treating undefined as 0, mean over defined, #defined)."""
    arr = np.array(values, float)
    defined = arr[~np.isnan(arr)]
    return (
        float(np.nan_to_num(arr, nan=0.0).mean()),
        float(defined.mean()) if len(defined) else float("nan"),
        len(defined),
    )


def bootstrap_ci(values: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_gap(q: np.ndarray) -> tuple[float, float]:
    """Best-minus-worst mean gap (by overall means) and its paired SE over k."""
    means = q.mean(axis=1)
    best, worst = int(means.argmax()), int(means.argmin())
    diff = q[best] - q[worst]
    se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return float(means[best] - means[worst]), se
