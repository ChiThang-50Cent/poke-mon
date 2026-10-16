"""Validate a collected run and emit the machine-readable W2 acceptance report."""

import argparse
import json
from pathlib import Path

from pokeai.data.validate import validate_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/w2_validation.json"))
    args = parser.parse_args()
    run = Path(args.run)
    report = validate_run(run if run.is_dir() else Path("runs") / args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
