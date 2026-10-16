"""Collect a sampled W5 opponent mixture, then validate the joined native dataset."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from pokeai.env.runner import RunConfig, run_battles


def allocate_battles(opponents: list[dict], battles: int) -> list[int]:
    """Largest-remainder allocation preserves the requested total deterministically."""
    total = sum(float(item["weight"]) for item in opponents)
    if battles < 1 or total <= 0 or any(item["weight"] < 0 for item in opponents):
        raise ValueError("positive battle count and nonnegative mixture required")
    exact = [battles * item["weight"] / total for item in opponents]
    counts = [int(value) for value in exact]
    for index in sorted(range(len(exact)), key=lambda i: (-(exact[i] - counts[i]), i))[
        :battles - sum(counts)
    ]:
        counts[index] += 1
    return counts


async def collect(config: dict, output: Path, device: str) -> Path:
    if Path(config["teams"]).resolve() != Path("data/teams_w5/collect").resolve():
        raise ValueError("W5 collection must use only data/teams_w5/collect")
    if config["temperature"] != 1.0 or not 1 <= config["concurrent"] <= 8:
        raise ValueError("collection requires temperature 1.0 and concurrency <=8")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output / "collection_config.json").write_text(json.dumps(config, indent=2) + "\n")
    entries, records, summaries = [], [], []
    manifest = None
    for index, (opponent, count) in enumerate(zip(
        config["opponents"], allocate_battles(config["opponents"], config["battles"]),
        strict=True,
    )):
        if not count:
            continue
        summary = await run_battles(RunConfig(
            player="neural:" + config["learner"], opponent=opponent["policy"],
            opponent_type=opponent["type"], battles=count,
            concurrent=config["concurrent"], teams=config["teams"],
            sample=True, temperature=1.0, device=device, save_trajectories=True,
            team_preview_checkpoint=config["team_preview_checkpoint"],
            team_preview_argmax=True, schedule_seed=config["seed"] + index,
            alternate_challenger=True, out=str(output / "parts"),
        ))
        part = output / "parts" / summary["run_id"]
        part_manifest = json.loads((part / "manifest.json").read_text())
        if manifest is None:
            manifest = part_manifest.copy()
        records.extend(json.loads(line) for line in
                       (part / "battles.jsonl").read_text().splitlines())
        summaries.append(summary)
        for entry in part_manifest["trajectory_files"]:
            for key in ("file", "meta"):
                destination = output / entry[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.hardlink_to(part / entry[key])
            entries.append(entry)
        if summary["error_count"]:
            raise RuntimeError(f"pilot part {part} has battle errors; retained evidence")
    assert manifest is not None
    manifest.update(run_id=output.name, config=config, trajectory_files=entries,
                    expected_trajectories=2 * config["battles"],
                    saved_trajectories=len(entries), missing=[])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "battles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "run_id": output.name, "attempted": len(records),
        "completed": sum(r["error"] is None for r in records),
        "error_count": sum(r["error"] is not None for r in records),
        "wall_time_s": time.perf_counter() - started, "parts": summaries,
        "neural": {role: {"illegal_actions": sum(
            s.get("neural", {}).get(role, {}).get("illegal_actions", 0)
            for s in summaries)} for role in ("player", "opponent")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/w5_selfplay.yaml")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.out or (
        "runs/w5-selfplay-pilot-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    ))
    run = asyncio.run(collect(config, output, args.device))
    from pokeai.data.selfplay_validate import validate_selfplay

    report = validate_selfplay(run, device=args.device)
    print(json.dumps(report, indent=2))
    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
