"""Download a bounded, rate-limited prefix sample of Gen9 OU human replays."""

import argparse
import json
import os
from pathlib import Path

from pokeai.data.download import HumanDownloadConfig, download_human_replays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("METAMON_CACHE_DIR", ".cache/metamon"))
        / "w3-human",
    )
    parser.add_argument("--target-valid-decisions", type=int)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--rate-limit-mbps", type=float, default=12.0, help="MB/s")
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    config = HumanDownloadConfig(
        max_files=args.max_files,
        target_valid_decisions=args.target_valid_decisions,
        min_rating=args.min_rating,
        rate_limit_bps=args.rate_limit_mbps * 1_000_000,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(download_human_replays(args.root, config), indent=2))


if __name__ == "__main__":
    main()
