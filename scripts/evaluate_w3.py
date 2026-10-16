"""Evaluate the frozen W3 checkpoint against three local fixed baselines."""

import argparse
import asyncio
import hashlib
import json
import platform
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from pokeai.env.runner import RunConfig, run_battles
from pokeai.repro import source_fingerprint


async def evaluate(args: argparse.Namespace) -> dict:
    if args.battles < 200:
        raise ValueError("W3 evaluation requires at least 200 battles per opponent")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = Path(args.model)
    fingerprints = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest()
        for name in ("best.pt", "config.yaml", "tokenizer.json")
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": str(model),
        "fingerprints": fingerprints,
        "source_sha256": source_fingerprint(),
        "metamon_commit": "0a00a75",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": args.seed,
        "team_pool": args.teams,
        "team_manifest_sha256": hashlib.sha256(
            (Path(args.teams) / "gen9ou" / "manifest.json").read_bytes()
        ).hexdigest(),
        "backend": {
            "selected": "metamon",
            "reason": "MetamonPlayer uses SimProtocol/ParsedReplay, the same replay "
            "parser as human trajectories. UniversalState.from_Battle reads its "
            "replay-backed attributes, matching Metamon PokeEnvWrapper's adapter.",
            "evidence": [
                "third_party/metamon/metamon/env/metamon_player.py:31-38",
                "third_party/metamon/metamon/env/metamon_battle.py:19-25,44-47,72-77",
                "third_party/metamon/metamon/env/wrappers.py:301-310,359-360",
                "third_party/metamon/README.md:1188-1200",
            ],
            "alternative": "poke-env is documented to have a replay sim2sim gap, "
            "broken PP counting and tera types; not selected.",
            "feature_diff_measurement": "Not measured against poke-env; live unknown "
            "token counts and rates are measured per opponent below.",
            "revival": "Explicit fainted-party-slot fallback; vendor online parser "
            "skips side updates on reviving requests.",
        },
        "opponents": {},
        "run_ids": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for opponent in ("RandomBaseline", "Grunt", "GymLeader"):
        summary = await run_battles(RunConfig(
            player=f"neural:{model}", opponent=opponent, battles=args.battles,
            concurrent=args.concurrent, teams=args.teams, device=args.device,
            timeout_s=args.timeout_s,
        ))
        report["opponents"][opponent] = summary
        report["run_ids"].append(summary["run_id"])
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"opponent": opponent, **summary}), flush=True)
    report["passed"] = all(
        item["completed"] >= args.battles and item["error_count"] == 0
        and item["neural"]["player"]["illegal_actions"] == 0
        for item in report["opponents"].values()
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"W3 evaluation failed; see {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/pokedecision-v0")
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--teams", default="data/teams")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", default="reports/w3_evaluation.json")
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
