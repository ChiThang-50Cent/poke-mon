"""BS1.2 driver: pair-blocks x lanes (processes, 4 search workers each); projection log.

Defaults reproduce BS1.2; a config may carry a `run:` section (seed, lanes, block_pairs,
budget_h, out_prefix, final) as the frozen BS1.3 config does.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

BLOCKS = [(0, 12), (12, 25), (25, 37), (37, 50)]  # team pairs per opponent
ROOT = Path("runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/w5_bs1_2_benchmark.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run = config.get("run", {})
    seed = run.get("seed", 20261201)
    lanes = run.get("lanes", 2)
    blocks = [tuple(b) for b in run.get("block_pairs", BLOCKS)]
    budget_s = run.get("budget_h", 16) * 3600
    prefix = run.get("out_prefix", "w5-bs1-2")
    total_pairs = blocks[-1][1]
    battles = str(config["battles"])
    final = bool(run.get("final"))
    if final:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pool_claim import verify_only

        def verify_all() -> None:  # under the pool lock: never races a lane's claim
            verify_only(config["pool"], config, args.config)

        verify_all()  # fail before any lane starts or any claim exists
    t0 = time.time()
    log = ROOT / f"{prefix}-projections.jsonl"
    done_pairs = 0
    for k, (a, b) in enumerate(blocks, 1):
        if final and k > 1:
            verify_all()  # before every further block: tamper / orphan claim temp => no launch
        tb = time.time()
        procs = []
        for lane in range(lanes):
            out = ROOT / f"{prefix}-b{k}" / f"lane{lane}"
            out.mkdir(parents=True, exist_ok=True)
            procs.append(subprocess.Popen(
                [sys.executable, "scripts/evaluate_w5.py", "--config", args.config,
                 "--battles", battles, "--pairs", str(a), str(b),
                 "--lane", str(lane), "--lanes", str(lanes), "--concurrent", "1",
                 "--seed", str(seed), "--out", str(out / "benchmark.json"),
                 *(["--final"] if run.get("final") else [])],
                stdout=(out / "stdout.log").open("w"), stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        done_pairs += b - a
        elapsed = time.time() - t0
        projected = elapsed / done_pairs * total_pairs
        rec = {"block": k, "pairs": [a, b], "exit_codes": codes, "block_wall_s": time.time() - tb,
               "elapsed_s": elapsed, "projected_total_s": projected,
               "budget_s": budget_s, "continue": projected <= budget_s and not any(codes)}
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if any(codes):
            print("lane failure: stop and report", rec, flush=True)
            sys.exit(1)
        if projected > budget_s and k < len(blocks):
            print("INCOMPLETE: projection over budget", rec, flush=True)
            sys.exit(2)


if __name__ == "__main__":
    main()
