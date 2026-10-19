"""W6-B: collect C+search trajectories on teams_w5/collect, then validate them.

One process = one lane (4 search workers). Blocks of `block_battles` per opponent run until
`--blocks` or the `--budget-hours` deadline (checked between blocks; never extended).
Validation = W5.1 self-play checks (schema, leakage, C probability replay parity on every
neural POV) + the exact search-row join (`pokeai.data.search_capture`).

Preflight:  scripts/w6b_collect.py --blocks 4 --out runs/w6b-preflight
Pilot lane: scripts/w6b_collect.py --budget-hours 12 --lane K --out runs/w6b-collect/laneK
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pokeai.env.runner import RunConfig, run_battles

PARITY_MINIMUM = "Offline parity requires at least 200 neural POV trajectories"


def run_config(config: dict, opponent: dict, seed: int, out: Path) -> RunConfig:
    neural_opponent = opponent["policy"].startswith("neural:")
    return RunConfig(
        player="neural:" + config["learner"], opponent=opponent["policy"],
        opponent_type=opponent["type"], battles=config["block_battles"], concurrent=1,
        teams=config["teams"], timeout_s=config["battle_timeout_s"], device="cpu",
        save_trajectories=True, sample=False,
        team_preview_checkpoint=config["team_preview_checkpoint"], team_preview_argmax=True,
        opponent_sample=False,
        opponent_team_preview_checkpoint=(
            config["team_preview_checkpoint"] if neural_opponent else None),
        opponent_team_preview_disabled=not neural_opponent,
        opponent_team_preview_argmax=True,
        schedule_seed=seed, alternate_challenger=True, search=config["search"], out=str(out),
    )


async def collect(config: dict, output: Path, lane: int, blocks: int | None,
                  deadline: float | None) -> Path:
    if Path(config["teams"]).resolve() != Path("data/teams_w5/collect").resolve():
        raise ValueError("W6-B collection must use only data/teams_w5/collect")
    from pokeai.agents.live_search import SearchSettings, start_pool, stop_pool

    output.mkdir(parents=True, exist_ok=False)
    (output / "collection_config.json").write_text(
        json.dumps({**config, "lane": lane, "blocks": blocks, "deadline": deadline}, indent=2))
    started = time.perf_counter()
    entries, records, summaries, manifest = [], [], [], None
    start_pool(SearchSettings.from_dict(config["search"]))
    try:
        block = 0
        while (blocks is None or block < blocks) and (deadline is None or time.time() < deadline):
            for index, opponent in enumerate(config["opponents"].values()):
                seed = config["seed"] + 100_000 * lane + 10 * block + index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                summary = await run_battles(run_config(config, opponent, seed, output / "parts"))
                part = output / "parts" / summary["run_id"]
                part_manifest = json.loads((part / "manifest.json").read_text())
                manifest = manifest or part_manifest.copy()
                records.extend(map(json.loads, (part / "battles.jsonl").read_text().splitlines()))
                summaries.append(summary)
                for entry in part_manifest["trajectory_files"]:
                    for key in ("file", "meta"):
                        destination = output / entry[key]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.hardlink_to(part / entry[key])
                    entries.append(entry)
                if summary["error_count"] or part_manifest["missing"]:
                    raise RuntimeError(f"part {part} has battle errors; evidence retained")
            block += 1
            print(json.dumps({"block": block, "battles": len(records),
                              "elapsed_h": (time.perf_counter() - started) / 3600}), flush=True)
    finally:
        stop_pool()
        if manifest is not None:
            manifest.update(run_id=output.name, config=config, trajectory_files=entries,
                            expected_trajectories=2 * len(records),
                            saved_trajectories=len(entries), missing=[])
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (output / "battles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (output / "summary.json").write_text(json.dumps({
                "run_id": output.name, "attempted": len(records),
                "completed": sum(r["error"] is None for r in records),
                "error_count": sum(r["error"] is not None for r in records),
                "wall_time_s": time.perf_counter() - started, "parts": summaries,
                "neural": {role: {"illegal_actions": sum(
                    s.get("neural", {}).get(role, {}).get("illegal_actions", 0)
                    for s in summaries)} for role in ("player", "opponent")},
            }, indent=2) + "\n")
    return output


def validate(run: Path, device: str) -> dict:
    from pokeai.data.search_capture import search_join
    from pokeai.data.selfplay_validate import validate_selfplay

    selfplay = validate_selfplay(run, report_path=run / "selfplay_validation.json", device=device)
    # A preflight is far below W5.1's 200-POV volume gate; every other check still applies.
    selfplay_errors = [e for e in selfplay["errors"] if e != PARITY_MINIMUM]
    join = search_join(run)
    report = {
        "status": "PASS" if not selfplay_errors and not join["errors"]
        and selfplay["native_dataset_load"] == "PASS" else "FAIL",
        "neural_parity_trajectories": selfplay["neural_parity_trajectories"],
        "neural_parity_by_side": selfplay["neural_parity_by_side"],
        "compared_decisions": selfplay["compared_decisions"],
        "max_probability_difference": selfplay["max_probability_difference"],
        "model_illegal_actions": selfplay["model_illegal_actions"],
        "battle_error_count": selfplay["battle_error_count"],
        "selfplay_errors": selfplay_errors[:50],
        **{k: v for k, v in join.items() if k != "errors"},
        "join_errors": join["errors"][:50],
        "join_error_count": len(join["errors"]),
    }
    (run / "w6b_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/w6b_collect.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lane", type=int, default=0)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--blocks", type=int)
    budget.add_argument("--budget-hours", type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load(Path(args.config).read_text())
    deadline = time.time() + 3600 * args.budget_hours if args.budget_hours else None
    run = asyncio.run(collect(config, Path(args.out), args.lane, args.blocks, deadline))
    report = validate(run, args.device)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
