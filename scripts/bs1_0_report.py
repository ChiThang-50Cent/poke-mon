"""Assemble reports/w5_bs1_0_{deadline,matrix}.json from runs/w5-bs1-0-*/ (evaluation code)."""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bs0_report import cluster_bootstrap, pct

JITTER_S = 0.05
BUDGET = 5.0


TRAP = ("|error|[Unavailable choice] Can't switch: The active Pokémon is trapped",
        "|error|[Invalid choice] Can't switch: The active Pokémon is trapped")


def unexpected(r: dict[str, Any]) -> list[str]:
    """Rollout errors that are not the documented hidden-trap protocol retry (CF0/CF1 class)."""
    return [e for e in r["errors"] if not e.startswith((*TRAP, "protocol-retry:"))]


def trap_retries(rows: list[dict[str, Any]]) -> int:
    return sum(len(r["errors"]) - len(unexpected(r)) + r["retries"] for r in rows)


def load_rows(stage: str) -> list[dict[str, Any]]:
    f = Path(f"runs/w5-bs1-0-{stage}/decisions.jsonl")
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def deadline() -> dict[str, Any] | None:
    rows = load_rows("deadline")
    if not rows:
        return None
    outer = [r["outer_wall_s"] for r in rows]
    sup = [r for r in rows if not r["unsupported"]]
    ok4 = [r["n_pairs"] >= 4 for r in rows]
    res: dict[str, Any] = {
        "budget_s": BUDGET, "sched_stop_s": BUDGET - 0.3, "roots": len(rows),
        "wall_p50_s": pct(outer, 50), "wall_p95_s": pct(outer, 95), "wall_max_s": max(outer),
        "pool_wall_max_s": max(r["wall_s"] for r in rows),
        "decisions_over_budget_plus_jitter": sum(w > BUDGET + JITTER_S for w in outer),
        "decisions_hitting_hard_deadline": sum(r["hard_deadline_hit"] for r in rows),
        "recycled_workers": sum(r["recycled"] for r in rows),
        "pairs_p50": pct([r["n_pairs"] for r in rows], 50),
        "pairs_p95": pct([r["n_pairs"] for r in rows], 95),
        "rollouts_p50": pct([r["n_rollouts"] for r in rows], 50),
        "supported_roots": len(sup), "unsupported_roots": len(rows) - len(sup),
        "unsupported_reasons": dict(Counter(
            u.split(" (")[0] for r in rows for u in set(r["unsupported"]))),
        "frac_ge4_pairs_all_roots": float(np.mean(ok4)),
        "frac_ge4_pairs_supported_roots": float(np.mean([r["n_pairs"] >= 4 for r in sup])),
        "override_rate": float(np.mean([r["override"] for r in rows])),
        "fallback_rate": float(np.mean([not r["override"] for r in rows])),
        "fallback_reasons": dict(Counter(r["reason"] for r in rows)),
        "fallback_always_legal_c": all(
            r["legal"] and (r["override"] or r["action"] == r["a_c"]) for r in rows),
        "illegal_actions": sum(r["illegal"] for r in rows),
        "unexpected_errors": sum(len(unexpected(r)) for r in rows),
        "protocol_retries": trap_retries(rows),
    }
    res["gate_M4"] = {
        "max_wall_le_budget_plus_50ms": res["wall_max_s"] <= BUDGET + JITTER_S,
        "fallback_always_legal": res["fallback_always_legal_c"],
        "ge4_pairs_on_ge90pct_supported": res["frac_ge4_pairs_supported_roots"] >= 0.9,
    }
    res["gate_M4"]["pass"] = all(res["gate_M4"].values())
    return res


def delta_of(row: dict[str, Any], table: np.ndarray) -> float:
    idx = row["candidates"].index(row["action"])
    return float(table[idx].mean() - table[0].mean())


def cell(rows: list[dict[str, Any]], oracle: dict[int, np.ndarray]) -> dict[str, Any]:
    deltas = np.array([delta_of(r, oracle[r["root"]]) for r in rows])
    clusters = [r["pair"] for r in rows]
    mean, lo, hi = cluster_bootstrap(deltas, clusters)
    sup = [i for i, r in enumerate(rows) if not r["unsupported"]]
    s_mean = s_lo = s_hi = None
    if sup:
        s_mean, s_lo, s_hi = cluster_bootstrap(deltas[sup], [clusters[i] for i in sup])
    ov = [d for d, r in zip(deltas, rows, strict=True) if r["override"]]

    def by(key: str) -> dict[str, Any]:
        g: dict[str, list[float]] = defaultdict(list)
        for d, r in zip(deltas, rows, strict=True):
            g[str(r[key])].append(float(d))
        return {k: {"roots": len(v), "mean_delta": float(np.mean(v))} for k, v in sorted(g.items())}

    return {
        "roots": len(rows), "delta": mean, "ci95": [lo, hi],
        "ci_excludes_0": bool(lo > 0 or hi < 0),
        "supported_only": {"roots": len(sup), "delta": s_mean, "ci95": [s_lo, s_hi]},
        "override_rate": len(ov) / len(rows), "fallback_rate": 1 - len(ov) / len(rows),
        "mean_delta_per_override": float(np.mean(ov)) if ov else None,
        "overrides_positive": int(sum(d > 0 for d in ov)),
        "overrides_negative": int(sum(d < 0 for d in ov)),
        "unsupported_roots": len(rows) - len(sup),
        "unsupported_reasons": dict(Counter(
            u.split(" (")[0] for r in rows for u in set(r["unsupported"]))),
        "by_phase": by("phase"), "by_candidate_count": by("n_cands"),
        "wall_p50_s": pct([r["outer_wall_s"] for r in rows], 50),
        "wall_p95_s": pct([r["outer_wall_s"] for r in rows], 95),
        "wall_max_s": max(r["outer_wall_s"] for r in rows),
        "pairs_p50": pct([r["n_pairs"] for r in rows], 50),
        "_deltas": deltas,
    }


def verdict(c: dict[str, Any]) -> str:
    lo, hi = c["ci95"]
    return "PASS" if lo > 0 else "NEGATIVE" if hi < 0 else "INCONCLUSIVE"


def matrix() -> dict[str, Any] | None:
    rows = load_rows("matrix")
    f = Path("runs/w5-bs1-0-matrix/payloads.pkl")
    if not rows or not f.exists():
        return None
    oracle = {p["root"]: np.array(p["oracle"]) for p in pickle.loads(f.read_bytes())
              if "oracle" in p}
    by_cell: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_cell[(r["real"], r["model"])][r["root"]] = r
    out: dict[str, Any] = {
        "evaluation": (
            "simulator value with a KNOWN real-opponent policy via true-snapshot rollouts "
            "(16 fresh seeds, a_search vs a_C paired, C-greedy continuation); NOT Showdown "
            "win rate. Real opponent C: its Python state is the real game state restored "
            "from the true snapshot (CF0.1 restore proof); the search never models C."),
        "cells": {}, "roots_per_real": {k: len({r["root"] for r in rows if r["real"] == k})
                                        for k in ("Grunt", "GymLeader", "C")},
    }
    deltas: dict[tuple[str, str], dict[int, float]] = {}
    for key, rs in sorted(by_cell.items()):
        lst = [rs[i] for i in sorted(rs)]
        c = cell(lst, oracle)
        deltas[key] = {r["root"]: d for r, d in zip(lst, c.pop("_deltas"), strict=True)}
        c["role"] = {("Grunt", "grunt"): "positive control", ("GymLeader", "gymleader"):
                     "positive control", ("Grunt", "gymleader"): "mismatch stress",
                     ("GymLeader", "grunt"): "mismatch stress", ("C", "mixture"):
                     "neural stress test"}.get(key, "mixture on a heuristic opponent")
        out["cells"][f"{key[0]}|{key[1]}"] = c
    pooled = [by_cell[(real, "mixture")][i] for real in ("Grunt", "GymLeader")
              for i in sorted(by_cell[(real, "mixture")])]
    p = cell(pooled, oracle)
    p.pop("_deltas")
    out["primary_mixture_pooled_grunt_gymleader"] = p
    out["primary_verdict"] = verdict(p)
    # matched - mismatched (and mixture - matched), paired per root, cluster CI
    pairs_of = {r["root"]: r["pair"] for r in rows}
    diffs = {}
    for real, matched, mismatched in (("Grunt", "grunt", "gymleader"),
                                      ("GymLeader", "gymleader", "grunt")):
        for name, a, b in ((f"{real}: matched - mismatched", matched, mismatched),
                           (f"{real}: mixture - matched", "mixture", matched),
                           (f"{real}: mixture - mismatched", "mixture", mismatched)):
            da, db = deltas.get((real, a)), deltas.get((real, b))
            if not da or not db:
                continue
            ids = sorted(set(da) & set(db))
            arr = np.array([da[i] - db[i] for i in ids])
            m, lo, hi = cluster_bootstrap(arr, [pairs_of[i] for i in ids])
            diffs[name] = {"roots": len(ids), "diff": m, "ci95": [lo, hi]}
    out["paired_differences"] = diffs
    out["gate_M1"] = {
        "unexpected_errors": sum(len(unexpected(r)) for r in rows),
        "protocol_retries_hidden_trap": trap_retries(rows),
        "illegal_actions": sum(r["illegal"] for r in rows),
        "illegal_chosen_actions": sum(not r["legal"] for r in rows),
        "fallback_always_legal_c": all(
            r["legal"] and (r["override"] or r["action"] == r["a_c"]) for r in rows),
    }
    out["gate_M1"]["pass"] = (out["gate_M1"]["unexpected_errors"] == 0
                              and out["gate_M1"]["illegal_actions"] == 0
                              and out["gate_M1"]["illegal_chosen_actions"] == 0
                              and out["gate_M1"]["fallback_always_legal_c"])
    out["gate_M2"] = out["primary_verdict"].replace("NEGATIVE", "FAIL")
    main_cells = {r: verdict(out["cells"][f"{r}|mixture"]) for r in ("Grunt", "GymLeader")
                  if f"{r}|mixture" in out["cells"]}
    out["gate_M3"] = {"main_group_mixture_verdicts": main_cells,
                      "pass": all(v != "NEGATIVE" for v in main_cells.values()),
                      "real_C_cell_verdict": verdict(out["cells"]["C|mixture"])
                      if "C|mixture" in out["cells"] else None}
    return out


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    for name, fn in (("deadline", deadline), ("matrix", matrix)):
        data = fn()
        if data is not None:
            Path(f"reports/w5_bs1_0_{name}.json").write_text(json.dumps(data, indent=1))
            print("wrote", name)


if __name__ == "__main__":
    main()
