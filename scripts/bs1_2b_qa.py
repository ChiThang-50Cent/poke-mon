"""BS1.2b QA assembly -> reports/w5_bs1_2b_qa.json (smoke / requalification stage metrics).

  bs1_2b_qa.py stage NAME REPORT_JSON... [--resources FILE]   (prints non-win engineering gates)
  bs1_2b_qa.py assemble                                        (writes the QA json)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import bs1_1_report as R
import bs1_2_report as B

STAGES = {
    "smoke": ["runs/w5-bs1-2b-smoke/laneA.json", "runs/w5-bs1-2b-smoke/laneB.json"],
    "requal": ["runs/w5-bs1-2b-requal-b1/lane0/benchmark.json",
               "runs/w5-bs1-2b-requal-b1/lane1/benchmark.json"],
    # SUPERSEDED: run before the choice-lock root-cause fix (kept as evidence, never merged)
    "smoke_prefix": ["runs/w5-bs1-2b-smoke-prefix/laneA.json",
                     "runs/w5-bs1-2b-smoke-prefix/laneB.json"],
    "requal_prefix": ["runs/w5-bs1-2b-requal-prefix-b1/lane0/benchmark.json",
                      "runs/w5-bs1-2b-requal-prefix-b1/lane1/benchmark.json"],
}
RESOURCES = {"smoke": "runs/w5-bs1-2b-smoke/resources.txt",
             "requal": "runs/w5-bs1-2b-requal-resources.txt",
             "smoke_prefix": "runs/w5-bs1-2b-smoke-prefix/resources.txt",
             "requal_prefix": "runs/w5-bs1-2b-requal-prefix-resources.txt"}


def cpu_hours(text: str) -> float | None:
    import re

    m = re.search(r"Consumed (?:(\d+)d )?(?:(\d+)h )?(?:(\d+)min )?([\d.]+)(ms|s) CPU", text)
    if not m:
        return None
    d, h, mi, s = (float(x or 0) for x in m.groups()[:4])
    s = s / 1000 if m.group(5) == "ms" else s
    return d * 24 + h + mi / 60 + s / 3600


def stage(name: str) -> dict[str, Any]:
    paths = [Path(p) for p in STAGES[name]]
    reports = [json.loads(p.read_text()) for p in paths]
    rows = B.rows_of(paths)
    eng = B.engineering(rows)
    decisions = R.decision_rows(rows)
    illegal = sum(
        v for rep in reports for arms in rep["opponents"].values() for res in arms.values()
        for v in (res.get("model_illegal_by_side") or {}).values())
    valid = all(res.get("valid") is True for rep in reports
                for arms in rep["opponents"].values() for res in arms.values())
    cats = Counter(d["category"] for d in decisions)
    known = Counter(d.get("error_class") or "unclassified"
                    for d in decisions if d["category"] == "search_error")
    wlt: dict[str, Counter] = {}
    for opp, arm, r in rows:
        wlt.setdefault(f"{opp}|{arm}", Counter())[r["winner"]] += 1
    resources = Path(RESOURCES[name]).read_text().strip() if Path(RESOURCES[name]).exists() else ""
    mask_rejects = [{k: m.get(k) for k in ("world", "opp", "bad", "live_mask", "world_mask")}
                    | {"battle": d["battle"], "turn": d.get("turn"), "role": d.get("learner_role")}
                    for d in decisions for m in d.get("mask_rejects", [])]
    out = {
        "stage": name, "battles": len(rows), "valid_reports": valid,
        "model_illegal_actions": illegal, "engineering": eng,
        "new_class_exceptions": cats.get("search_error_new_class", 0),
        "known_class_search_errors_by_category": dict(known),
        "decisions_over_deadline": eng["latency"]["decisions_over_deadline"],
        "decisions": len(decisions),
        "worlds_rejected_by_mask": sum(d.get("world_rejected_mask", 0) for d in decisions),
        "decisions_with_rejected_worlds": sum(1 for d in decisions if d.get("world_rejected_mask")),
        "pairs_dropped_parser": sum(d.get("dropped_pairs", 0) for d in decisions),
        "dropped_reasons": dict(Counter(x[:120] for d in decisions for x in d.get("dropped", []))),
        "mask_rejects_logged": mask_rejects[:30],
        "resources": resources, "cpu_hours": cpu_hours(resources),
        "winners_descriptive_only": {k: dict(v) for k, v in wlt.items()},
    }
    out["gates"] = {
        "all_battles_valid": valid and not eng["errors"]["timeouts"]
        and not eng["errors"]["battle_errors"] and illegal == 0,
        "zero_new_class_exceptions": out["new_class_exceptions"] == 0,
        "zero_decisions_over_5s": out["decisions_over_deadline"] == 0
        and eng["latency"]["missing_t_send"] == 0,
        "state_integrity": eng["state_integrity"]["pass"],
        "fixed_classes_zero": sum(known.values()) == 0,
    }
    return out


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "stage":
        res = stage(sys.argv[2])
        print(json.dumps({k: v for k, v in res.items()
                          if k not in ("engineering", "winners_descriptive_only")}, indent=1)[:5000])
    else:
        qa = {name: stage(name) for name in STAGES if Path(STAGES[name][0]).exists()}
        qa["golden_trace_parity"] = json.loads(
            Path("runs/w5-bs1-2b-golden/compare.json").read_text())
        Path("reports/w5_bs1_2b_qa.json").write_text(json.dumps(qa, indent=1))
        print("wrote qa")


if __name__ == "__main__":
    main()
