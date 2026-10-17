"""BS1.2 driver: 4 pair-blocks x 2 lanes (processes, 4 search workers each); projection log."""

import json
import subprocess
import sys
import time
from pathlib import Path

BLOCKS = [(0, 12), (12, 25), (25, 37), (37, 50)]  # team pairs per opponent
BUDGET_S = 16 * 3600
SEED = 20261201
ROOT = Path("runs")


def main() -> None:
    t0 = time.time()
    log = ROOT / "w5-bs1-2-projections.jsonl"
    done_pairs = 0
    for k, (a, b) in enumerate(BLOCKS, 1):
        tb = time.time()
        procs = []
        for lane in (0, 1):
            out = ROOT / f"w5-bs1-2-b{k}" / f"lane{lane}"
            out.mkdir(parents=True, exist_ok=True)
            procs.append(subprocess.Popen(
                [sys.executable, "scripts/evaluate_w5.py", "--config",
                 "configs/w5_bs1_2_benchmark.yaml", "--battles", "100", "--pairs", str(a), str(b),
                 "--lane", str(lane), "--lanes", "2", "--concurrent", "1", "--seed", str(SEED),
                 "--out", str(out / "benchmark.json")],
                stdout=(out / "stdout.log").open("w"), stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        done_pairs += b - a
        elapsed = time.time() - t0
        projected = elapsed / done_pairs * 50
        rec = {"block": k, "pairs": [a, b], "exit_codes": codes, "block_wall_s": time.time() - tb,
               "elapsed_s": elapsed, "projected_total_s": projected,
               "budget_s": BUDGET_S, "continue": projected <= BUDGET_S and not any(codes)}
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if any(codes):
            print("lane failure: stop and report", rec, flush=True)
            sys.exit(1)
        if projected > BUDGET_S and k < len(BLOCKS):
            print("INCOMPLETE: projection over budget", rec, flush=True)
            sys.exit(2)


if __name__ == "__main__":
    main()
