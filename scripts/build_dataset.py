"""Build a frozen battle-grouped W2 index, including a bounded human sample."""

import argparse
import json
from pathlib import Path

from pokeai.data.dataset import default_human_root, human_entries, local_entries
from pokeai.data.download import HumanDownloadConfig, download_human_replays
from pokeai.data.split import write_dataset_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("datasets/w2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-count", type=int, default=2)
    args = parser.parse_args()
    rows = []
    for run_id in args.runs:
        path = Path(run_id)
        rows.extend(local_entries(path if path.is_dir() else Path("runs") / run_id))
    root = default_human_root()
    stats = download_human_replays(root, HumanDownloadConfig(max_files=args.human_count))
    rows.extend(human_entries(root))
    splits = write_dataset_index(rows, args.out, args.seed)
    print(
        json.dumps(
            {
                "trajectories": len(rows),
                "split_battles": {name: len(ids) for name, ids in splits.items()},
                "human_replay_sample": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
