"""Operational checks of a finished rehearsal (BS1.4-P): NO win rate is computed or read.

  bs_rehearsal_check.py NAME [--watchdog-log PATH]

Reads only: claim files, lane benchmark.json `valid` flags, the driver's exit codes, the `error`
field of every battle row, lane logs, the watchdog log. Writes runs/rehearsal/NAME/check.json and
exits 1 when any check fails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REFUSAL_MARKERS = ("release manifest mismatch", "refusing the heldout claim", "Traceback")


def check(name: str, root: Path = ROOT, watchdog_log: Path | None = None) -> dict:
    base = root / "runs/rehearsal" / name
    config = yaml.safe_load((base / "config.yaml").read_text())
    run = config["run"]
    lanes, blocks, prefix = run["lanes"], run["block_pairs"], run["out_prefix"]
    pool = root / config["pool"]
    n_arms, n_opp = len(config["arms"]), len(config["opponents"])
    expected_battles = config["battles"] * n_arms * n_opp
    out: dict = {"name": name, "checks": {}}

    expected = {f"final.window.{lane}of{lanes}.{a}-{b}.json"
                for a, b in blocks for lane in range(lanes)}
    present = {p.name for p in pool.glob("final.window.*.json")}
    out["checks"]["windows"] = {
        "expected": len(expected), "present": len(present),
        "missing": sorted(expected - present), "extra": sorted(present - expected),
        "ok": present == expected}
    out["checks"]["final_started_json"] = (pool / "final.started.json").is_file()

    reports, rows_total, bad_rows, invalid, missing_reports = 0, 0, 0, [], []
    for k in range(1, len(blocks) + 1):
        for lane in range(lanes):
            path = root / "runs" / f"{prefix}-b{k}" / f"lane{lane}" / "benchmark.json"
            try:
                report = json.loads(path.read_text())
            except (OSError, ValueError):
                missing_reports.append(str(path.relative_to(root)))
                continue
            reports += 1
            if report.get("valid") is not True:
                invalid.append(str(path.relative_to(root)))
            for arms in report.get("opponents", {}).values():
                for result in arms.values():
                    for line in Path(result["battles_path"]).read_text().splitlines():
                        rows_total += 1
                        bad_rows += json.loads(line).get("error") is not None
    out["checks"]["lane_reports"] = {
        "expected": len(blocks) * lanes, "present": reports, "missing": missing_reports,
        "invalid": invalid, "ok": not missing_reports and not invalid}
    out["checks"]["battles"] = {"expected": expected_battles, "found": rows_total,
                                "with_error": bad_rows,
                                "ok": rows_total == expected_battles and bad_rows == 0}

    codes, projections = [], root / "runs" / f"{prefix}-projections.jsonl"
    if projections.is_file():
        codes = [json.loads(line)["exit_codes"] for line in projections.read_text().splitlines()]
    out["checks"]["exit_codes"] = {
        "blocks_recorded": len(codes), "expected_blocks": len(blocks),
        "ok": len(codes) == len(blocks) and all(c == [0] * lanes for c in codes)}

    refused = []
    for log in sorted((root / "runs").glob(f"{prefix}-b*/lane*/stdout.log")):
        text = log.read_text(errors="replace")
        refused += [f"{log.relative_to(root)}: {m}" for m in REFUSAL_MARKERS if m in text]
    out["checks"]["lane_logs_without_refusal"] = {"findings": refused, "ok": not refused}

    watchdog_log = watchdog_log or root / "runs" / f"{prefix}-watchdog.log"
    if watchdog_log.is_file():
        failures = [ln for ln in watchdog_log.read_text().splitlines() if "LANE FAILURE" in ln]
        out["checks"]["watchdog_log"] = {"path": str(watchdog_log), "lane_failures": failures,
                                         "ok": not failures}
    else:
        out["checks"]["watchdog_log"] = {"path": str(watchdog_log), "ok": False,
                                         "problem": "watchdog log missing"}
    staging = pool.parent / f".{pool.name}.claim-staging"
    orphans = sorted(p.name for p in staging.iterdir()) if staging.is_dir() else []
    out["checks"]["no_orphan_claim_temp"] = {"orphans": orphans, "ok": not orphans}
    out["ok"] = all(v if isinstance(v, bool) else v["ok"] for v in out["checks"].values())
    (base / "check.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--watchdog-log", type=Path, default=None)
    args = ap.parse_args()
    result = check(args.name, watchdog_log=args.watchdog_log)
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
