"""Preprocess the downloaded Gen9 OU human pilot into frozen W3 BC arrays."""

import argparse
from pathlib import Path

from pokeai.data.w3 import build_w3_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".cache/metamon/w3-human"))
    parser.add_argument("--out", type=Path, default=Path("datasets/w3"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/w3_dataset_validation.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--live-run", type=Path, default=Path("runs/20261016T074108-cd71eb26")
    )
    args = parser.parse_args()
    report = build_w3_dataset(args.root, args.out, args.report, args.seed, args.live_run)
    print(f"W3 dataset: {report['status']} ({args.report})")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
