"""W5.BS1.3-P fix QA assembly -> reports/w5_bs1_3p_fix_qa.json (R6 golden parity + R7 requal stages).

  bs1_3p_qa.py stage NAME      (prints the stage; NAME = smoke | requal)
  bs1_3p_qa.py assemble        (writes the QA json)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bs1_1_report as R
import bs1_2b_qa as Q

Q.STAGES.clear()
Q.STAGES.update({
    "smoke": ["runs/w5-bs1-3-fix/smoke/laneA.json", "runs/w5-bs1-3-fix/smoke/laneB.json"],
    "requal": ["runs/w5-bs1-3p-requal-b1/lane0/benchmark.json",
               "runs/w5-bs1-3p-requal-b1/lane1/benchmark.json"],
})
Q.RESOURCES.clear()
Q.RESOURCES.update({"smoke": "runs/w5-bs1-3-fix/smoke/resources.txt",
                    "requal": "runs/w5-bs1-3-fix/requal-resources.txt"})


def stage(name: str) -> dict:
    out = Q.stage(name)
    paths = [Path(p) for p in Q.STAGES[name]]
    import bs1_2_report as B

    decisions = R.decision_rows(B.rows_of(paths))
    aborts = [a for d in decisions for a in d.get("rollout_aborts", [])]
    out["world_rejected_state"] = sum(d.get("world_rejected_state", 0) for d in decisions)
    out["decisions_with_state_rejects"] = sum(1 for d in decisions if d.get("world_rejected_state"))
    out["state_rejects_logged"] = [s for d in decisions for s in d.get("state_rejects", [])][:30]
    out["rollout_aborts"] = dict(Counter(a.get("kind") for a in aborts))
    out["RolloutStalled"] = sum(1 for a in aborts if a.get("kind") == "stalled")
    out["RolloutHorizonExceeded"] = sum(1 for a in aborts if a.get("kind") == "horizon")
    out["unsupported_reasons"] = out["engineering"]["search"].get("unsupported_reasons", {})
    base = json.loads(Path("reports/w5_bs1_2b_qa.json").read_text())[name]["engineering"]["search"]
    mine = out["engineering"]["search"]
    out["coverage_vs_bs1_2b"] = {
        "frac_searched_with_ge4_pairs": [mine["frac_searched_with_ge4_pairs"],
                                         base["frac_searched_with_ge4_pairs"]],
        "reconstruction_support_rate": [mine["reconstruction_support_rate"],
                                        base["reconstruction_support_rate"]],
        "note": "[this run, BS1.2b]; different seeds/team pairs, descriptive only"}
    out["gates"]["no_timeouts"] = not out["engineering"]["errors"]["timeouts"]
    out["gates"]["zero_illegal"] = out["model_illegal_actions"] == 0
    return out


def main() -> None:
    if sys.argv[1] == "stage":
        res = stage(sys.argv[2])
        print(json.dumps({k: v for k, v in res.items()
                          if k not in ("engineering", "winners_descriptive_only")}, indent=1)[:6000])
        return
    qa = {n: stage(n) for n in Q.STAGES if Path(Q.STAGES[n][0]).exists()}
    qa["golden_trace_parity"] = json.loads(
        Path("runs/w5-bs1-3-fix/golden/compare.json").read_text())
    Path("reports/w5_bs1_3p_fix_qa.json").write_text(json.dumps(qa, indent=1))
    print("wrote qa")


if __name__ == "__main__":
    main()
