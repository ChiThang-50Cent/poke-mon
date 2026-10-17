"""BS1.2: calibration verdict, engineering metrics and the locked paired benchmark analysis.

Usage: bs1_2_report.py calibration            -> reports/w5_bs1_2_calibration.json
       bs1_2_report.py engineering BLOCK_DIR..  -> prints engineering metrics only (no wins)
       bs1_2_report.py final BLOCK_DIR..        -> reports/w5_bs1_2_benchmark.json (win rates!)
A BLOCK_DIR holds lane0/benchmark.json, lane1/benchmark.json (+ resources.txt) of one block.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bs1_1_report as R
from evaluate_w4_5 import bootstrap, cluster_differences

BOOT_SEED = 20261201
CAL = Path("runs/w5-bs1-2-calibration")


def rows_of(report_paths: list[Path]) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for path in report_paths:
        report = json.loads(path.read_text())
        for opp, arms in report["opponents"].items():
            for arm, result in arms.items():
                for line in Path(result["battles_path"]).read_text().splitlines():
                    out.append((opp, arm, json.loads(line)))
    return out


def peak_gb(text: str) -> float | None:
    m = re.search(r"([\d.]+)([MG]) memory peak", text)
    if not m:
        return None
    return float(m.group(1)) / (1024 if m.group(2) == "M" else 1)


def engineering(rows: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    decisions = R.decision_rows(rows)
    per_arm = {}
    for arm in (R.BASE_ARM, R.SEARCH_ARM):
        rs = [r for _, a, r in rows if a == arm]
        if rs:
            per_arm[arm] = {
                "battles": len(rs), "wall_s_per_battle": R.stats([r["duration_s"] for r in rs]),
                "decisions_per_battle": R.stats([r["decisions_player"] for r in rs])}
    return {
        "per_arm": per_arm, "errors": R.integrity_errors(rows), "latency": R.latency(decisions),
        "search": R.availability(decisions), "state_integrity": R.state_integrity(rows)}


def calibration() -> dict[str, Any]:
    paths = [CAL / "Grunt.json", CAL / "GymLeader.json"]
    eng = engineering(rows_of(paths))
    pilot = json.loads(Path("reports/w5_bs1_1_runtime.json").read_text())
    resources = (CAL / "resources.txt").read_text() if (CAL / "resources.txt").exists() else ""
    peak = peak_gb(resources)
    s, p = eng["search"], pilot["search"]
    ratio = s["complete_pairs"]["p50"] / p["complete_pairs"]["p50"]
    verdict = {
        "p50_pairs_ge_80pct_of_pilot": ratio >= 0.8,
        "coverage_ge_85pct": (s["frac_searched_with_ge4_pairs"] or 0) >= 0.85,
        "zero_deadline_violations": eng["latency"]["decisions_over_deadline"] == 0
        and eng["latency"]["missing_t_send"] == 0,
        "peak_scope_memory_le_12gb": peak is not None and peak <= 12.0,
    }
    verdict["use_2_lanes_x_4_workers"] = all(verdict.values())
    return {
        "setup": "2 lanes (processes) x 4 workers, 1 battle each at a time; lane A Grunt 2 "
        "battles, lane B GymLeader 2 battles; seed 20261199; wrapper CPUQuota=1000% MemoryMax=14G",
        "calibration": eng, "pilot_reference": {
            "p50_pairs": p["complete_pairs"]["p50"],
            "frac_ge4_pairs": p["frac_searched_with_ge4_pairs"],
            "end_to_end_max_s": pilot["latency"]["end_to_end_s"]["max"]},
        "p50_pairs_ratio_to_pilot": ratio, "resources": resources.strip(),
        "peak_scope_memory_gb": peak, "rule": verdict}


def merge_block(block: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Rows of one block per opponent/arm, both lanes, re-indexed from 0 (even offset)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for opp, arm, row in rows_of(sorted(block.glob("lane*/benchmark.json"))):
        out.setdefault(opp, {}).setdefault(arm, []).append(row)
    for arms in out.values():
        for arm, rows in arms.items():
            rows.sort(key=lambda r: r["matchup_index"])
            base = rows[0]["matchup_index"]
            assert base % 2 == 0
            arms[arm] = [{**r, "matchup_index": r["matchup_index"] - base} for r in rows]
    return out


def stratified(strata: dict[str, np.ndarray], *, seed: int, resamples: int = 10000) -> dict:
    """Cluster bootstrap resampling team pairs within each opponent stratum."""
    rng = np.random.default_rng(seed)
    est = np.zeros(resamples)
    total = sum(len(v) for v in strata.values())
    for v in strata.values():
        est += v[rng.integers(len(v), size=(resamples, len(v)))].sum(axis=1)
    est /= total
    ci = np.quantile(est, [0.025, 0.975]).tolist()
    pooled = np.concatenate(list(strata.values()))
    return {"win_rate_difference": float(pooled.mean()), "bootstrap_95": ci,
            "verdict": "positive" if ci[0] > 0 else "negative" if ci[1] < 0 else "inconclusive",
            "clusters": total, "strata": {k: len(v) for k, v in strata.items()},
            "seed": seed, "resamples": resamples}


def pair_set(seeds_and_n: list[tuple[int, int]]) -> set[frozenset[str]]:
    import random
    files = sorted(Path("data/teams_w5/dev").rglob("*.gen9ou_team"))
    out = set()
    for seed, n in seeds_and_n:
        rng = random.Random(seed)
        for _ in range(n // 2):
            out.add(frozenset((rng.choice(files).name, rng.choice(files).name)))
    return out


def overlap() -> dict[str, Any]:
    runs = {"w5-bs1-1a": [(20261101 + k, 2) for k in range(3)],
            "w5-bs1-1b-aborted1": [(20261101 + k, 10) for k in range(3)],
            "w5-bs1-1b-block1": [(20261101 + k, 10) for k in range(3)],
            "w5-bs1-1b-block2": [(20261111 + k, 10) for k in range(3)]}
    new = [frozenset(x) for k in range(3) for x in [pair_set([(20261201 + k, 100)])]]
    return {name: [len(n & pair_set(v)) for n in new] for name, v in runs.items()}


def diagnostics(rows: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    search = [(o, r) for o, a, r in rows if a == R.SEARCH_ARM]
    decisions = [d for d in R.decision_rows(rows) if d["category"] != "not_searched_single_legal"]
    searched = [d for d in decisions if d["category"] in ("search_override",
                                                           "search_fallback_no_signal")]
    def rate(ds):
        return {"decisions": len(ds), "override_rate": (
            sum(d["category"] == "search_override" for d in ds) / len(ds)) if ds else None}
    def phase(turn):
        return "turn<10" if turn < 10 else "turn10-24" if turn < 25 else "turn25+"
    out: dict[str, Any] = {
        "override_by_opponent": {o: rate([d for d in searched if d["opponent"] == o])
                                 for o in sorted({d["opponent"] for d in searched})},
        "override_by_phase": {ph: rate([d for d in searched if phase(d["turn"]) == ph])
                              for ph in ("turn<10", "turn10-24", "turn25+")}}
    out["override_by_opponent_and_phase"] = {
        f"{o}/{ph}": rate([d for d in searched if d["opponent"] == o and phase(d["turn"]) == ph])
        for o in sorted({d["opponent"] for d in searched})
        for ph in ("turn<10", "turn10-24", "turn25+")}
    ov = Counter()
    for _, r in search:
        has = any(x["category"] == "search_override" for x in r.get("search", []))
        ov[(has, r["winner"] == "player")] += 1
    out["override_vs_no_override_battles"] = {
        "with_override": {"battles": ov[(True, True)] + ov[(True, False)],
                          "wins": ov[(True, True)]},
        "without_override": {"battles": ov[(False, True)] + ov[(False, False)],
                             "wins": ov[(False, True)]},
        "note": "descriptive, selection-confounded, not causal"}
    causes = Counter(
        [x for x in (d.get("error") or "").strip().splitlines() if x][-1][:160]
        for d in decisions if d["category"] == "search_error")
    out["search_exceptions_by_cause"] = dict(causes)
    out["search_exception_fallback_outcome"] = "C's own legal action sent (category search_error)"
    return out


def seat_delta(blocks: list[Path]) -> dict[str, Any]:
    stats: dict[str, Counter] = {}
    for block in blocks:
        for arms in merge_block(block).values():
            base, srch = arms[R.BASE_ARM], arms[R.SEARCH_ARM]
            for a, b in zip(base, srch, strict=True):
                role = next((x.get("learner_role") for x in b.get("search", [])), None)
                key = role or "unknown"
                c = stats.setdefault(key, Counter())
                c["n"] += 1
                c["c_wins"] += a["winner"] == "player"
                c["search_wins"] += b["winner"] == "player"
    return {k: {**dict(v), "delta": (v["search_wins"] - v["c_wins"]) / v["n"]}
            for k, v in stats.items()}


def final(blocks: list[Path]) -> dict[str, Any]:
    clusters: dict[str, list[np.ndarray]] = {}
    wlt: dict[str, dict[str, Counter]] = {}
    pairs_per_block = []
    for block in blocks:
        merged = merge_block(block)
        pairs_per_block.append({o: len(a[R.BASE_ARM]) // 2 for o, a in merged.items()})
        for opp, arms in merged.items():
            clusters.setdefault(opp, []).append(cluster_differences(
                arms[R.BASE_ARM], arms[R.SEARCH_ARM]))
            for arm, rows in arms.items():
                wlt.setdefault(opp, {}).setdefault(arm, Counter()).update(
                    r["winner"] for r in rows)
    strata = {o: np.concatenate(v) for o, v in clusters.items()}
    diff = {"PRIMARY_all_three_stratified": stratified(strata, seed=BOOT_SEED),
            "heuristics_Grunt+GymLeader_stratified": stratified(
                {o: strata[o] for o in ("Grunt", "GymLeader")}, seed=BOOT_SEED + 1)}
    for opp, v in strata.items():
        diff[opp] = bootstrap(v, seed=BOOT_SEED + 2, resamples=10000)
    all_rows = [x for d in blocks for x in rows_of(sorted(d.glob("lane*/benchmark.json")))]
    eng = engineering(all_rows)
    lane_valid = all(
        res.get("valid") is True
        for d in blocks for f in d.glob("lane*/benchmark.json")
        for arms in json.loads(f.read_text())["opponents"].values() for res in arms.values())
    illegal = sum(
        v for d in blocks for f in d.glob("lane*/benchmark.json")
        for arms in json.loads(f.read_text())["opponents"].values() for res in arms.values()
        for v in res.get("model_illegal_by_side", {}).values())
    gates = {
        "B1_reliability": not (eng["errors"]["timeouts"] or eng["errors"]["battle_errors"])
        and lane_valid and illegal == 0,
        "B2_deadline": eng["latency"]["decisions_over_deadline"] == 0
        and eng["latency"]["missing_t_send"] == 0,
        "B3_search_benefit": diff["PRIMARY_all_three_stratified"]["bootstrap_95"][0] > 0,
        "B4_no_clear_regression": all(diff[o]["bootstrap_95"][1] >= 0 for o in strata),
        "B5_reproducibility": len(blocks) == 4 and pairs_per_block == [
            {o: n for o in strata} for n in (12, 13, 12, 13)],
    }
    return {
        "label": "DEV benchmark; fixed heuristic opponents + C (greedy); no promotion claim; "
        "heldout untouched; Showdown RNG unseeded (paired by team and side)",
        "status": "COMPLETE" if len(blocks) == 4 else "INCOMPLETE",
        "blocks": [str(b) for b in blocks], "pairs_per_block": pairs_per_block,
        "config_sha256": Path("runs/w5-bs1-2-config.sha256").read_text().split()[0],
        "projections": [json.loads(x) for x in Path(
            "runs/w5-bs1-2-projections.jsonl").read_text().splitlines()],
        "wlt": {o: {a: dict(c) for a, c in arms.items()} for o, arms in wlt.items()},
        "paired_difference_C+search_minus_C": diff, "gates": gates,
        "team_pair_overlap_new_schedule_vs_bs1_1_runs (per opponent Grunt,GymLeader,C)": overlap(),
        "delta_by_learner_seat": seat_delta(blocks), "diagnostics": diagnostics(all_rows),
        "engineering": eng}


def main() -> None:
    stage, dirs = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    Path("reports").mkdir(exist_ok=True)
    if stage == "calibration":
        data = calibration()
        Path("reports/w5_bs1_2_calibration.json").write_text(json.dumps(data, indent=1))
        print(json.dumps({k: data[k] for k in ("p50_pairs_ratio_to_pilot", "peak_scope_memory_gb",
                                              "rule")}, indent=1))
        s = data["calibration"]["search"]
        print(s["complete_pairs"], s["frac_searched_with_ge4_pairs"],
              data["calibration"]["latency"]["end_to_end_s"], data["calibration"]["errors"])
    elif stage == "engineering":
        rows = [x for d in dirs for x in rows_of(sorted(d.glob("lane*/benchmark.json")))]
        print(json.dumps(engineering(rows), indent=1)[:6000])
    else:
        data = final(dirs)
        Path("reports/w5_bs1_2_benchmark.json").write_text(json.dumps(data, indent=1))
        print("wrote benchmark")


if __name__ == "__main__":
    main()
