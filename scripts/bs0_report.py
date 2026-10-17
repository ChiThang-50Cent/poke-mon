"""Assemble reports/w5_bs0_*.json from the per-root runs (stages reached only)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RUNS = Path("runs")
REPORTS = Path("reports")


def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def stage1() -> dict[str, Any] | None:
    d = RUNS / "w5-bs0-1"
    if not (d / "summary.json").exists():
        return None
    summary = json.loads((d / "summary.json").read_text())
    roots = [json.loads(x) for x in (d / "roots.jsonl").read_text().splitlines()]
    sup = [w for r in roots for w in r["worlds"] if w["supported"]]
    reasons_roots = Counter(
        w["reason"].split(" (")[0] for r in roots for w in r["worlds"][:1] if not w["supported"])
    cover = defaultdict(lambda: [0, 0])
    for r in roots:
        ok = all(w["supported"] for w in r["worlds"])
        cover[r["phase"]][0] += int(ok)
        cover[r["phase"]][1] += 1
    summary["supported_fraction_by_phase"] = {k: v[0] / v[1] for k, v in sorted(cover.items())}
    summary["unsupported_root_reasons"] = dict(reasons_roots.most_common())
    summary["mean_reconstruct_s"] = float(np.mean([w["timing"]["reconstruct_s"] for w in sup]))
    summary["mean_sample_s"] = float(np.mean([w["timing"]["sample_s"] for w in sup]))
    summary["gate_S1"] = {
        "a_request_100pct": summary["request_mismatch_worlds"] == 0,
        "b_mask_100pct": summary["mask_mismatch_worlds"] == 0,
        "c_public_view_100pct": summary["view_mismatch_worlds"] == 0,
        "supported_fraction_ge_0.8": summary["supported_root_fraction"] >= 0.8,
    }
    summary["gate_S1"]["pass"] = all(summary["gate_S1"].values())
    return summary


def stage2() -> dict[str, Any] | None:
    f = RUNS / "w5-bs0-2" / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    s["gate_S2"] = {
        "deterministic_all": s["deterministic_worlds"] == s["supported_worlds"],
        "root_inputs_invariant_and_equal_true": s["root_inputs_equal_true_worlds"]
        == s["supported_worlds"],
        "pov_violations_0": s["pov_violation_worlds"] == 0,
        "illegal_actions_0": s["illegal_actions"] == 0,
        "unexpected_errors_0": s["unexpected_error_count"] == 0,
    }
    s["gate_S2"]["pass"] = all(s["gate_S2"].values())
    return s


def stage3() -> dict[str, Any] | None:
    d = RUNS / "w5-bs0-3"
    files = sorted(d.glob("decisions_*.json"))
    if not files:
        return None
    out: dict[str, Any] = {"budgets": {}}
    for f in files:
        rows = json.loads(f.read_text())
        budget = rows[0]["budget"]
        walls = [r["wall_s"] for r in rows]
        has4 = [r["n_pairs"] >= 4 for r in rows]
        comp = {}
        for key in ("load_s", "sample_s", "reconstruct_s", "pov_s", "rollout_s"):
            comp[key] = {
                "sum_p50": pct([sum(r["timing"][key]) for r in rows], 50),
                "sum_p95": pct([sum(r["timing"][key]) for r in rows], 95),
                "critical_path_p50": pct([max(r["timing"][key]) for r in rows], 50),
                "critical_path_p95": pct([max(r["timing"][key]) for r in rows], 95),
            }
        n_roll = sum(r["n_rollouts"] for r in rows)
        legal = all(r["legal"] for r in rows)
        fallback_legal = all(
            r["legal"] and (r["override"] or r["action"] == r["a_c"]) for r in rows)
        res = {
            "roots": len(rows), "wall_p50_s": pct(walls, 50), "wall_p95_s": pct(walls, 95),
            "wall_max_s": max(walls), "dispatch_p50_s": pct([r["dispatch_s"] for r in rows], 50),
            "pairs_p50": pct([r["n_pairs"] for r in rows], 50),
            "pairs_p95": pct([r["n_pairs"] for r in rows], 95),
            "rollouts_p50": pct([r["n_rollouts"] for r in rows], 50),
            "rollouts_p95": pct([r["n_rollouts"] for r in rows], 95),
            "rollouts_mean": n_roll / len(rows),
            "frac_roots_ge4_pairs": float(np.mean(has4)),
            "frac_roots_ge4_pairs_supported_only": float(np.mean(
                [h for h, r in zip(has4, rows, strict=True) if not r["unsupported"]]))
            if any(not r["unsupported"] for r in rows) else None,
            "fallback_rate": float(np.mean([not r["override"] for r in rows])),
            "override_rate": float(np.mean([r["override"] for r in rows])),
            "fallback_reasons": dict(Counter(r["reason"] for r in rows)),
            "unsupported_roots": sum(bool(r["unsupported"]) for r in rows),
            "illegal_actions": sum(r["illegal"] for r in rows),
            "protocol_retries": sum(r["retries"] for r in rows),
            "unexpected_errors": sum(len(r["errors"]) for r in rows),
            "components": comp,
            "all_actions_legal": legal, "fallback_always_legal_c": fallback_legal,
        }
        res["gate"] = {
            "ge4_pairs_on_ge90pct_roots": res["frac_roots_ge4_pairs"] >= 0.9,
            "p95_le_budget_plus_10pct": res["wall_p95_s"] <= budget * 1.1,
            "fallback_always_legal_c": fallback_legal,
        }
        res["gate"]["pass"] = all(res["gate"].values())
        out["budgets"][str(budget)] = res
    out["gate_S3_any_budget"] = any(b["gate"]["pass"] for b in out["budgets"].values())
    return out


def cluster_bootstrap(
    delta: np.ndarray, clusters: list[str], resamples: int = 5000, seed: int = 20261500
) -> tuple[float, float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for d, c in zip(delta, clusters, strict=True):
        groups[c].append(d)
    keys = list(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, len(keys), len(keys))
        means[i] = sums[pick].sum() / counts[pick].sum()
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stage4() -> dict[str, Any] | None:
    d = RUNS / "w5-bs0-4"
    files = sorted(d.glob("decisions_*.json"))
    if not files or not (d / "payloads.pkl").exists():
        return None
    import pickle

    payloads = {p["root"]: p for p in pickle.loads((d / "payloads.pkl").read_bytes())
                if "error" not in p}
    out: dict[str, Any] = {"budgets": {}, "evaluation": (
        "simulator value with a KNOWN opponent policy (GymLeader/Grunt heuristics), measured by "
        "true-state oracle rollouts from the real snapshot (16 fresh seeds, paired), C greedy "
        "continuation; NOT Showdown win rate")}
    for f in files:
        rows = json.loads(f.read_text())
        budget = rows[0]["budget"]
        deltas, clusters, recs = [], [], []
        best_possible = []
        for r in rows:
            table = np.array(payloads[r["root"]]["oracle"])
            idx = r["candidates"].index(r["action"])
            delta = float(table[idx].mean() - table[0].mean())
            deltas.append(delta)
            clusters.append(payloads[r["root"]]["pair"])
            best_possible.append(float(table.mean(axis=1).max() - table[0].mean()))
            recs.append({"root": r["root"], "opponent": r["opponent"], "phase": r["phase"],
                         "n_cands": r["n_cands"], "override": r["override"], "delta": delta,
                         "n_pairs": r["n_pairs"]})
        arr = np.array(deltas)
        mean, lo, hi = cluster_bootstrap(arr, clusters)

        def group(key: str, recs: list[dict[str, Any]] = recs) -> dict[str, Any]:
            g: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for x in recs:
                g[str(x[key])].append(x)
            return {k: {
                "roots": len(v), "mean_delta": float(np.mean([x["delta"] for x in v])),
                "override_rate": float(np.mean([x["override"] for x in v])),
                "mean_delta_per_override": float(np.mean(
                    [x["delta"] for x in v if x["override"]])) if any(
                    x["override"] for x in v) else None,
            } for k, v in sorted(g.items())}

        ov = [x for x in recs if x["override"]]
        indep = [r["independent_delta"] for r in rows if r.get("independent_delta") is not None]
        out["budgets"][str(budget)] = {
            "roots": len(rows), "delta_search": mean, "ci95": [lo, hi],
            "ci_excludes_0": lo > 0 or hi < 0, "bootstrap": "team-pair cluster, 5000 resamples",
            "n_clusters": len(set(clusters)),
            "override_rate": len(ov) / len(rows), "fallback_rate": 1 - len(ov) / len(rows),
            "mean_delta_per_override": float(np.mean([x["delta"] for x in ov])) if ov else None,
            "overrides_with_positive_oracle_delta": sum(x["delta"] > 0 for x in ov),
            "overrides_with_negative_oracle_delta": sum(x["delta"] < 0 for x in ov),
            "by_opponent": group("opponent"), "by_phase": group("phase"),
            "by_candidate_count": group("n_cands"),
            "pairs_p50": pct([r["n_pairs"] for r in rows], 50),
            "secondary_independent_belief_delta_mean_over_overrides":
                float(np.mean(indep)) if indep else None,
            "secondary_independent_n": len(indep),
            "oracle_best_candidate_delta_upper_bound": float(np.mean(best_possible)),
            "pass": bool(budget == 2.0 and mean > 0 and lo > 0),
        }
    out["gate_S4_primary_2s"] = out["budgets"].get("2.0", {}).get("pass")
    return out


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    for name, fn in (("1_reconstruction", stage1), ("2_correctness", stage2),
                     ("3_profile", stage3), ("4_benefit", stage4)):
        data = fn()
        if data is not None:
            (REPORTS / f"w5_bs0_{name}.json").write_text(json.dumps(data, indent=1))
            print("wrote", name)


if __name__ == "__main__":
    main()
