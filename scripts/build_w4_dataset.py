"""Build and validate the 2025–2026 temporal human replay dataset (no training)."""

import argparse
from pathlib import Path

from pokeai.data.w4 import build_w4_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".cache/metamon/w4-human"))
    parser.add_argument("--out", type=Path, default=Path("datasets/w4"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/w4_dataset_validation.json")
    )
    parser.add_argument("--w3-root", type=Path, default=Path("datasets/w3"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = build_w4_dataset(
        args.root, args.out, args.report, args.seed, args.workers, args.w3_root
    )
    print(f"W4 dataset: {report['status']} ({args.report})")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
