"""Assemble reports/w5_bs1_1_{integration,runtime,paired_pilot}.json from evaluate_w5 run dirs.

Usage: bs1_1_report.py integration RUN_DIR      (BS1.1a gates L1-L4)
       bs1_1_report.py runtime RUN_DIR [RUN_DIR ...]   (engineering metrics, no win rates)
       bs1_1_report.py paired RUN_DIR [RUN_DIR ...]    (exploratory gameplay; run last)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_w4_5 import bootstrap, cluster_differences

D = 5.0
SEARCH_ARM = "C+search"
BASE_ARM = "C"


def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def stats(values: list[float]) -> dict[str, float | None]:
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95),
            "max": max(values) if values else None, "mean": float(np.mean(values)) if values else None}


def load_run(run_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """opponent -> arm -> battle rows (from the evaluate_w5 report written in `run_dir`)."""
    report = json.loads((run_dir / "benchmark.json").read_text())
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for opp, arms in report["opponents"].items():
        for arm, result in arms.items():
            path = Path(result["battles_path"])
            out.setdefault(opp, {})[arm] = [json.loads(x) for x in path.read_text().splitlines()]
    return out


def all_rows(runs: list[Path]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(opp, arm, row) for r in runs for opp, arms in load_run(r).items()
            for arm, rows in arms.items() for row in rows]


def decision_rows(rows: list[tuple[str, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for opp, arm, battle in rows:
        if arm != SEARCH_ARM:
            continue
        for i, d in enumerate(battle.get("search", [])):
            out.append({**d, "opponent": opp, "battle": battle["battle_id"], "index": i,
                        "matchup": battle["matchup_index"]})
    return out


def latency(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    sent = [d for d in decisions if "t_send" in d]
    e2e = [d["t_send"] - d["t_request"] for d in sent]
    searched = [d for d in sent if "t_search_end" in d]
    phases = {
        "c_inference_and_state": [d["t_c_done"] - d["t_request"] for d in sent if "t_c_done" in d],
        "c_done_to_search_start": [d["t_search_start"] - d["t_c_done"] for d in searched
                                   if "t_c_done" in d],
        "search": [d["t_search_end"] - d["t_search_start"] for d in searched],
        "search_end_to_send": [d["t_send"] - d["t_search_end"] for d in searched],
        "non_search_choice_to_send": [d["t_send"] - d["t_choice_returned"] for d in sent
                                      if "t_search_end" not in d],
    }
    return {
        "decisions": len(decisions), "with_t_send": len(sent),
        "missing_t_send": len(decisions) - len(sent), "deadline_s": D,
        "decisions_over_deadline": sum(x > D for x in e2e), "end_to_end_s": stats(e2e),
        "phases_s": {k: stats(v) for k, v in phases.items()},
    }


def availability(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    cats = Counter(d["category"] for d in decisions)
    searched = [d for d in decisions if "n_pairs" in d]
    unsupported = Counter(u for d in searched for u in d.get("unsupported", []))
    return {
        "category_counts": dict(cats), "decisions": len(decisions),
        "searched_decisions": len(searched),
        "complete_pairs": stats([d["n_pairs"] for d in searched]),
        "rollouts": stats([d["n_rollouts"] for d in searched]),
        "frac_searched_with_ge4_pairs": float(np.mean([d["n_pairs"] >= 4 for d in searched]))
        if searched else None,
        "decisions_with_pairs": sum(d["n_pairs"] > 0 for d in searched),
        "unsupported_reasons": dict(unsupported),
        "decisions_with_unsupported_reason": sum(bool(d.get("unsupported")) for d in searched),
        "reconstruction_support_rate": float(np.mean(
            [d["worlds_built"] > 0 for d in searched])) if searched else None,
        "hard_deadline_hit": sum(bool(d.get("hard_deadline_hit")) for d in searched),
        "override_rate_of_searched": float(np.mean(
            [d["category"] == "search_override" for d in searched])) if searched else None,
        "search_errors": [d.get("error", "")[-300:] for d in decisions
                          if d["category"] == "search_error"][:5],
        "illegal_in_rollouts": sum(d.get("illegal", 0) for d in searched),
        "protocol_retries_in_rollouts": sum(d.get("retries", 0) for d in searched),
        "rollout_errors": [e for d in searched for e in d.get("rollout_errors", [])][:5],
        "by_learner_role": dict(Counter(d.get("learner_role") for d in searched)),
        "worker_restarts_max": max([d.get("restarts", 0) for d in decisions] or [0]),
    }


def state_integrity(rows: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    bad_forward, bad_hidden, cross, battles = [], [], [], 0
    running: dict[tuple[str, str, str], int] = {}
    cumulative_mismatch = []
    for opp, arm, battle in rows:
        if arm != SEARCH_ARM:
            continue
        battles += 1
        run = battle["run_id"]
        key = (run, opp, arm)
        for i, d in enumerate(battle.get("search", [])):
            want = 1 if d.get("model_choice") else 0
            if d["c_forwards"] != want:
                bad_forward.append([battle["battle_id"], i, d["c_forwards"], want])
            # an exception inside the search (category search_error) skips the post-search
            # digest; every decision that completed the search must show an unchanged memory
            if d.get("hidden_after") is not None and d.get("hidden_before") != d["hidden_after"]:
                bad_hidden.append([battle["battle_id"], i])
            if d["hidden_keys"] > 1:
                cross.append([battle["battle_id"], i, d["hidden_keys"]])
            running[key] = running.get(key, 0) + d["c_forwards"]
        state = battle.get("search_state", {})
        if state.get("forward_count") != running.get(key, 0):
            cumulative_mismatch.append([battle["battle_id"], state.get("forward_count"),
                                        running.get(key, 0)])
        if state.get("hidden_after_battle") != 0 or state.get("frame_tables") != [0, 0]:
            cross.append([battle["battle_id"], "end", state])
    return {
        "battles": battles, "per_decision_forward_violations": bad_forward,
        "hidden_changed_by_search": bad_hidden, "cross_battle_state": cross,
        "cumulative_forward_counter_mismatch": cumulative_mismatch,
        "pass": not (bad_forward or bad_hidden or cross or cumulative_mismatch),
        "offline_replay_parity": "not run: live Showdown RNG is not seedable, so there is no "
        "offline replay of the logged observations; verified by per-decision forward counters, "
        "pre/post-search hidden-state digests and unit tests (tests/test_bs1_1_live.py)",
    }


def integrity_errors(rows: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    timeouts = [r["battle_id"] for _, _, r in rows if "TimeoutError" in (r["error"] or "")]
    errors = [[r["battle_id"], r["error"]] for _, _, r in rows
              if r["error"] and "TimeoutError" not in r["error"]]
    return {"battles": len(rows), "timeouts": timeouts, "battle_errors": errors,
            "protocol_retries": sum(len(r["protocol_retries"]["player"]) for _, _, r in rows)}


def integration(run_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "benchmark.json").read_text())
    rows = all_rows([run_dir])
    decisions = decision_rows(rows)
    illegal = {f"{o}|{a}": r.get("model_illegal_by_side") for o, arms in report["opponents"].items()
               for a, r in arms.items()}
    errs = integrity_errors(rows)
    avail = availability(decisions)
    lat = latency(decisions)
    l1_unexpected = len(errs["battle_errors"]) + len(avail["search_errors"])
    out = {
        "run_dir": str(run_dir), "valid_report": report["valid"], "battles": len(rows),
        "by_cell": {f"{o}|{a}": {"battles": len(rs), "winners": dict(Counter(
            r["winner"] for r in rs))} for o, arms in load_run(run_dir).items()
            for a, rs in arms.items()},
        "errors": errs, "agent_illegal_actions_by_cell": illegal,
        "latency": lat, "state_integrity": state_integrity(rows), "search": avail,
    }
    out["gates"] = {
        "L1_server_correctness": {
            "timeouts_0": not errs["timeouts"],
            "agent_illegal_0": all(sum((v or {}).values()) == 0 for v in illegal.values()),
            "unexpected_errors_0": l1_unexpected == 0,
        },
        "L2_timing": {"zero_decisions_over_D": lat["decisions_over_deadline"] == 0
                      and lat["missing_t_send"] == 0},
        "L3_state_integrity": {"pass": out["state_integrity"]["pass"]},
        "L4_search_availability": {
            "complete_pairs_recorded": avail["decisions_with_pairs"] > 0,
            "categories_reported": True,
            "frac_searched_with_ge4_pairs": avail["frac_searched_with_ge4_pairs"]},
    }
    g = out["gates"]
    out["gates"]["L1_pass"] = all(g["L1_server_correctness"].values())
    out["gates"]["L2_pass"] = g["L2_timing"]["zero_decisions_over_D"]
    out["gates"]["L3_pass"] = g["L3_state_integrity"]["pass"]
    out["gates"]["L4_pass"] = g["L4_search_availability"]["complete_pairs_recorded"]
    out["gates"]["all_pass"] = all(out["gates"][k] for k in ("L1_pass", "L2_pass", "L3_pass",
                                                              "L4_pass"))
    return out


def runtime(run_dirs: list[Path]) -> dict[str, Any]:
    rows = all_rows(run_dirs)
    decisions = decision_rows(rows)
    per_arm: dict[str, Any] = {}
    for arm in (BASE_ARM, SEARCH_ARM):
        rs = [r for _, a, r in rows if a == arm]
        per_arm[arm] = {
            "battles": len(rs), "wall_s_per_battle": stats([r["duration_s"] for r in rs]),
            "decisions_per_battle": stats([r["decisions_player"] for r in rs]),
            "turns_per_battle": stats([r["turns"] for r in rs])}
    out = {
        "runs": [str(r) for r in run_dirs], "per_arm": per_arm,
        "errors": integrity_errors(rows), "latency": latency(decisions),
        "search": availability(decisions), "state_integrity": state_integrity(rows),
        "wall_s_total_by_run": {str(r): json.loads((r / "benchmark.json").read_text()).get(
            "wall_time_s") for r in run_dirs},
    }
    for r in run_dirs:
        f = r / "resources.txt"
        if f.exists():
            out.setdefault("resources", {})[str(r)] = f.read_text().strip()
    return out


def paired(run_dirs: list[Path]) -> dict[str, Any]:
    clusters: dict[str, list[np.ndarray]] = {}
    wlt: dict[str, dict[str, Counter]] = {}
    for r in run_dirs:
        for opp, arms in load_run(r).items():
            for arm, rs in arms.items():
                wlt.setdefault(opp, {}).setdefault(arm, Counter()).update(
                    r_["winner"] for r_ in rs)
            clusters.setdefault(opp, []).append(
                cluster_differences(arms[BASE_ARM], arms[SEARCH_ARM]))
    out: dict[str, Any] = {
        "label": "EXPLORATORY (n <= 20 per arm x opponent); simulator-adjacent local server "
        "play against the named opponents, not a Showdown ladder win rate",
        "runs": [str(r) for r in run_dirs], "wlt": {
            o: {a: dict(c) for a, c in arms.items()} for o, arms in wlt.items()},
        "paired_difference_C+search_minus_C": {}}
    pooled = []
    for opp, parts in clusters.items():
        arr = np.concatenate(parts)
        pooled.append(arr)
        out["paired_difference_C+search_minus_C"][opp] = bootstrap(arr, seed=20261101 + 10000)
    out["paired_difference_C+search_minus_C"]["pooled"] = bootstrap(
        np.concatenate(pooled), seed=20261101 + 20000)
    return out


def main() -> None:
    stage, dirs = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    Path("reports").mkdir(exist_ok=True)
    if stage == "integration":
        data, name = integration(dirs[0]), "integration"
    elif stage == "runtime":
        data, name = runtime(dirs), "runtime"
    else:
        data, name = paired(dirs), "paired_pilot"
    Path(f"reports/w5_bs1_1_{name}.json").write_text(json.dumps(data, indent=1))
    print("wrote", name)


if __name__ == "__main__":
    main()
