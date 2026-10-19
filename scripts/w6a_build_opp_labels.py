"""W6-A: build opponent-action labels for W4 train/val (test is spent and never opened)."""

import argparse
import json

from pokeai.data.opp_labels import build_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/w4")
    parser.add_argument("--source", default=".cache/metamon/w4-human")
    parser.add_argument("--output", default="datasets/w6a")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = build_labels(args.root, args.source, args.output, args.workers)
    print(json.dumps({"classes": len(report["vocab"]), **{
        split: {k: v for k, v in body.items() if k not in {"outcomes_alignment", "sha256"}}
        for split, body in report["splits"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
