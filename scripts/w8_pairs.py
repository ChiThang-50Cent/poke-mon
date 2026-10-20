"""Paired arm differences from an evaluate_w5.py report: per opponent (team-pair cluster
bootstrap) and pooled (stratified by opponent). Optional extra stratum: two run_metamon_h2h.py
runs on the same schedule (e.g. vs Kakuna).

    uv run python scripts/w8_pairs.py --report reports/w8_e1_dev.json --pairs e1-ctl e1-KK \
        --h2h Kakuna=runs/h2h500_kakuna_kk100,runs/w8_e1_vs_kakuna:e1-ctl \
        --out reports/w8_e1_dev_pairs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from bs1_2_report import stratified
from evaluate_w4_5 import bootstrap, cluster_differences


def rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_rows(spec: str) -> list[dict]:
    """h2h run dir, or lane dirs joined by '+' (run_metamon_h2h.py --lanes), by matchup_index."""
    out = [r for d in spec.split("+") for r in rows(Path(d) / "battles.jsonl")]
    return sorted(out, key=lambda r: r["matchup_index"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, metavar="TREATMENT-CONTROL")
    ap.add_argument("--h2h", nargs="*", default=[],
                    metavar="NAME=CONTROL_RUN,TREATMENT_RUN:PAIR",
                    help="add an opponent stratum from two h2h runs (dirs with battles.jsonl)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = json.loads(Path(args.report).read_text())
    extra: dict[str, dict[str, np.ndarray]] = {}
    for spec in args.h2h:
        name, rest = spec.split("=", 1)
        runs, pair = rest.rsplit(":", 1)
        control, treatment = runs.split(",")
        extra.setdefault(pair, {})[name] = cluster_differences(
            run_rows(control), run_rows(treatment))
    out = {}
    for pair in args.pairs:
        treatment, control = pair.split("-", 1)
        strata = {}
        for opp, arms in report["opponents"].items():
            strata[opp] = cluster_differences(rows(arms[control]["battles_path"]),
                                              rows(arms[treatment]["battles_path"]))
        result = {opp: bootstrap(v, seed=20271017) for opp, v in strata.items()}
        result["pooled"] = stratified(strata, seed=20271019)
        for name, v in extra.get(pair, {}).items():
            result[name] = bootstrap(v, seed=20271017)
            result["pooled_with_" + name] = stratified({**strata, name: v}, seed=20271019)
        out[pair] = result
        print(pair, json.dumps({k: (round(v["win_rate_difference"], 3),
                                    [round(x, 3) for x in v["bootstrap_95"]])
                                for k, v in result.items()}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
