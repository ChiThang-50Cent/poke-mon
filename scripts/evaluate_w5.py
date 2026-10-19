"""Configurable paired W5 benchmark; heldout is one-shot and final-only."""
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
from evaluate_w4_5 import bootstrap, cluster_differences
from evaluate_w4_team_preview import arm_report, sha256
from pool_claim import claim_unlocked, verify_and_claim

from pokeai.env.collector import git_commit
from pokeai.env.runner import RunConfig, run_battles
from pokeai.repro import source_fingerprint


def run_config(arm: dict, opponent: dict, *, pool: Path, battles: int,
               concurrent: int, seed: int, lane: list[int] | None = None,
               pair_window: list[int] | None = None) -> RunConfig:
    """Create independent neural players using the runner's per-side options."""
    return RunConfig(
        player="neural:" + arm["model_dir"],
        opponent=("neural:" + opponent["model_dir"]
                  if "model_dir" in opponent else opponent["baseline"]),
        battles=battles, concurrent=concurrent, teams=str(pool), device="cpu",
        sample=not arm.get("greedy", True),
        temperature=arm.get("temperature", 1.0),
        team_preview_checkpoint=arm.get("preview_checkpoint"),
        team_preview_argmax=arm.get("preview_argmax", True),
        opponent_sample=not opponent.get("greedy", True),
        opponent_temperature=opponent.get("temperature", 1.0),
        opponent_team_preview_checkpoint=opponent.get("preview_checkpoint"),
        opponent_team_preview_disabled=not bool(opponent.get("preview_checkpoint")),
        opponent_team_preview_argmax=opponent.get("preview_argmax", True),
        schedule_seed=seed, swap_team_pairs=True, alternate_challenger=True,
        search=arm.get("search"), timeout_s=arm.get("battle_timeout_s", 120.0),
        lane=lane, pair_window=pair_window,
    )


async def evaluate(args):
    started = time.perf_counter()
    torch.set_num_threads(1)
    config = yaml.safe_load(Path(args.config).read_text())
    pool = Path(args.pool or config["pool"])
    battles = args.battles or config["battles"]
    if battles < 2 or battles % 2:
        raise ValueError("An even positive number of battles is required")
    arms = config["arms"]
    if args.arm:
        arms = {name: arms[name] for name in args.arm}
    lane = [args.lane, args.lanes] if args.lanes > 1 else None
    pair_window = list(args.pairs) if args.pairs else None
    reference = config["reference_arm"]
    if reference not in arms:
        raise ValueError("The reference arm must be included")
    if any(not arm.get("greedy", True) for arm in arms.values()):
        raise ValueError("W5 benchmark arms must use greedy policies")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Preserve existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    effective = {**config, "arms": arms, "pool": str(pool), "battles": battles}
    window = None
    if args.lanes > 1 or args.pairs:
        window = (args.lane, args.lanes, *(args.pairs or (0, battles // 2)))
    if args.final:
        # ONE critical section: verify (code, config file, team files, pool root) then claim
        verify_and_claim(pool, config, args.config, effective, window)
    else:
        claim_unlocked(None, pool, False, effective, window)
    model_configs = [*arms.values(), *config["opponents"].values()]
    report = {
        "valid": False, "config": effective, "seed": args.seed,
        "purpose": "smoke only; no win-rate conclusions" if args.smoke else "benchmark",
        "fingerprints": {
            "git_commit": git_commit("."), "source_sha256": source_fingerprint(),
            "team_manifest_sha256": sha256(pool / "gen9ou/manifest.json"),
            "checkpoints": {p["model_dir"]: sha256(Path(p["model_dir"]) / "best.pt")
                            for p in model_configs if "model_dir" in p},
            "previews": {p["preview_checkpoint"]: sha256(Path(p["preview_checkpoint"]))
                         for p in model_configs if p.get("preview_checkpoint")},
        },
        "methodology": {
            "schedule": "seeded team pairs, both directions, alternating challenger",
            "interval": "W4.5 paired bootstrap clustered by team pair; ties are non-wins",
            "rng": "Showdown and heuristic RNG are not seedable",
            "selection": "dev only; heldout final-only, one-shot claim",
        }, "opponents": {}, "comparisons": {},
    }

    def save():
        report["wall_time_s"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    searching = [arm["search"] for arm in arms.values() if arm.get("search")]
    if searching:  # BS1.1: start the worker pool before any battle (startup is not a decision)
        from pokeai.agents.live_search import SearchSettings, start_pool

        start_pool(SearchSettings.from_dict(searching[0]))
    for index, (name, opponent) in enumerate(config["opponents"].items()):
        seed = args.seed + index
        rows_by_arm = {}
        result = report["opponents"][name] = {}
        for arm_name, arm in arms.items():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            summary = await run_battles(run_config(
                arm, opponent, pool=pool, battles=battles,
                concurrent=args.concurrent, seed=seed, lane=lane,
                pair_window=pair_window,
            ))
            log = Path(summary["config"]["out"]) / summary["run_id"] / "battles.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            rows_by_arm[arm_name] = rows
            if lane is not None or pair_window is not None:  # BS1.2: partial schedule; coverage is validated after merging
                result[arm_name] = {
                    "battles_path": str(log), "lane": lane, "completed": len(rows),
                    "valid": all(r["error"] is None for r in rows)}
                neural_stats = summary.get("neural", {})
                result[arm_name]["model_illegal_by_side"] = {
                    role: stats["illegal_actions"] for role, stats in neural_stats.items()}
                result[arm_name]["valid"] = result[arm_name]["valid"] and all(
                    stats["illegal_actions"] == 0 for stats in neural_stats.values())
                save()
                print(json.dumps({"opponent": name, "arm": arm_name,
                                  "valid": result[arm_name]["valid"]}), flush=True)
                if not result[arm_name]["valid"]:
                    raise RuntimeError(f"Invalid benchmark arm {arm_name} vs {name}")
                continue
            result[arm_name] = arm_report(summary, rows, battles)
            neural_stats = summary.get("neural", {})
            result[arm_name]["model_illegal_by_side"] = {
                role: stats["illegal_actions"] for role, stats in neural_stats.items()
            }
            result[arm_name]["valid"] = (
                result[arm_name]["valid"]
                and all(stats["illegal_actions"] == 0 for stats in neural_stats.values())
                and ("model_dir" not in opponent or "opponent" in neural_stats)
            )
            result[arm_name]["battles_path"] = str(log)
            # Also validate exact paired direction coverage for a single-arm smoke.
            cluster_differences(rows, rows)
            save()
            print(json.dumps({"opponent": name, "arm": arm_name,
                              "valid": result[arm_name]["valid"]}), flush=True)
            if not result[arm_name]["valid"]:
                raise RuntimeError(f"Invalid benchmark arm {arm_name} vs {name}")
        if not args.smoke and lane is None and pair_window is None:
            for name_arm in arms:
                if name_arm != reference:
                    clusters = cluster_differences(rows_by_arm[reference],
                                                   rows_by_arm[name_arm])
                    report["comparisons"].setdefault(name_arm + "-" + reference, {})[
                        name
                    ] = bootstrap(clusters, seed=seed + 10000)
    report["valid"] = True
    save()
    if searching:
        from pokeai.agents.live_search import stop_pool

        stop_pool()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/w5_benchmark.yaml")
    parser.add_argument("--pool")
    parser.add_argument("--battles", type=int)
    parser.add_argument("--arm", action="append")
    parser.add_argument("--concurrent", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="reports/w5_benchmark.json")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--pairs", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and (args.final or args.battles != 20):
        parser.error("--smoke requires --battles 20 and cannot use --final")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
