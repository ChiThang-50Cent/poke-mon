"""Frozen four-arm W4.5 benchmark; no training or checkpoint selection."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_w4_team_preview import OPPONENTS, PREVIEW, arm_report, sha256

from pokeai.env.collector import git_commit
from pokeai.env.runner import RunConfig, run_battles
from pokeai.repro import source_fingerprint

MODELS = {"A": "models/pokedecision-v0", "B": "models/pokedecision-v1",
              "C": "models/pokedecision-v1-value",
              "D": "models/pokedecision-v1-joint-ablation"}
COMPARISONS = (("D", "C"), ("B", "A"), ("C", "B"), ("C", "A"))


def verdict(interval):
    if interval[0] <= 0 <= interval[1]:
        return "inconclusive"
    return "positive" if interval[0] > 0 else "negative"


def cluster_differences(control, treatment):
    """Validate exact paired coverage, team swaps, and completed battles."""
    def index(records):
        result = {r["matchup_index"]: r for r in records}
        if len(result) != len(records) or set(result) != set(range(len(records))):
            raise ValueError("incomplete or duplicate matchup indices")
        return result
    a, b = index(control), index(treatment)
    if len(a) != len(b) or not a or len(a) % 2:
        raise ValueError("complete even paired coverage required")
    differences = []
    keys = ("player_team", "opponent_team", "player_team_sha256",
            "opponent_team_sha256", "player_challenges")
    for i in range(len(a)):
        if any(a[i].get(k) != b[i].get(k) or k not in a[i] for k in keys):
            raise ValueError("team schedule mismatch")
        for records in (a, b):
            r = records[i]
            if r.get("error") is not None or r["winner"] not in (
                "player", "opponent", "tie"
            ):
                raise ValueError("invalid battle")
            if i % 2 and any(records[i][p] != records[i-1][q] for p, q in (
                ("player_team_sha256", "opponent_team_sha256"),
                ("opponent_team_sha256", "player_team_sha256"),
            )):
                raise ValueError("pair directions are not swapped")
        differences.append(int(b[i]["winner"] == "player")
                           - int(a[i]["winner"] == "player"))
    return np.asarray(differences).reshape(-1, 2).mean(axis=1)


def bootstrap(clusters, *, seed=20261016, resamples=5000):
    if resamples < 5000:
        raise ValueError("at least 5000 resamples required")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for start in range(0, resamples, 128):
        n = min(128, resamples-start)
        estimates[start:start+n] = clusters[
            rng.integers(len(clusters), size=(n, len(clusters)))
        ].mean(axis=1)
    ci = np.quantile(estimates, [.025, .975]).tolist()
    return {"win_rate_difference": float(clusters.mean()), "bootstrap_95": ci,
                "verdict": verdict(ci), "clusters": len(clusters), "seed": seed,
                "resamples": resamples}


async def evaluate(args):
    started = time.perf_counter()
    torch.set_num_threads(1)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Preserve existing report before rerun: {output}")
    report = {"valid": False, "arms": MODELS, "opponents": {}, "seed": args.seed,
                  "battles_per_arm_per_opponent": args.battles,
                  "fingerprints": {"git_commit": git_commit("."),
                      "source_sha256": source_fingerprint(),
                      "checkpoints_sha256": {k: sha256(Path(v)/"best.pt")
                                          for k, v in MODELS.items()},
                      "team_manifest_sha256": sha256(Path("data/teams/gen9ou/manifest.json")),
                      "preview_sha256": sha256(PREVIEW)},
                  "methodology": {"policy": "greedy argmax, CPU, one thread; value logging only",
                      "preview": "C/D pretrained gen9ou_high_elo_v4 argmax on CUDA",
                      "schedule": "seeded 250 pairs, each direction, alternate challenger",
                      "rng": "Showdown and heuristic RNG are not seedable",
                      "teams": "Never used for training or selection; W4.3 measured only",
                      "selection": "Probe remains production. Joint failed W4.4 gate: .0155 > .01"}}
    prior = Path("reports/w4_5_ablation_run1.json")
    if prior.exists():
        report["prior_invalid_run"] = {
            "path": str(prior), "sha256": sha256(prior),
            "reason": "One RandomBaseline Revival Blessing target rejection and timeout",
            "fix": "Baseline revival orders now use fainted party slots from the current request, not cached Pokemon flags/species; native defaults disabled during revival",
        }
    records = {}
    def save():
        report["wall_time_s"] = time.perf_counter()-started
        output.write_text(json.dumps(report, indent=2)+"\n")
    save()
    for oi, opponent in enumerate(OPPONENTS):
        result = {"schedule_seed": args.seed+oi, "arms": {}}
        report["opponents"][opponent] = result
        records[opponent] = {}
        for arm, model in MODELS.items():
            seed = args.seed+oi
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            summary = await run_battles(RunConfig(
                player=f"neural:{model}", opponent=opponent, battles=args.battles,
                concurrent=args.concurrent, device="cpu", sample=False,
                schedule_seed=seed, swap_team_pairs=True, alternate_challenger=True,
                team_preview_checkpoint=str(PREVIEW) if arm in "CD" else None,
                team_preview_argmax=True))
            log = Path(summary["config"]["out"])/summary["run_id"]/"battles.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            records[opponent][arm] = rows
            result["arms"][arm] = arm_report(summary, rows, args.battles)
            result["arms"][arm]["battles_path"] = str(log)
            save()
            print(json.dumps({"opponent": opponent, "arm": arm,
                "valid": result["arms"][arm]["valid"],
                "wins": result["arms"][arm]["wins"]}), flush=True)
    paired = {"method": "seeded percentile paired bootstrap clustered by team pair; both directions together; pooled stratified by opponent",
                  "comparisons": {}, "multiple_comparisons": "16 comparisons; no multiplicity adjustment; exploratory except primary D-C"}
    for treatment, control in COMPARISONS:
        name = treatment+"-"+control
        clusters = {opp: cluster_differences(records[opp][control], records[opp][treatment])
                    for opp in OPPONENTS}
        result = {opp: bootstrap(x, seed=args.seed+10000+i)
                  for i, (opp, x) in enumerate(clusters.items())}
        # Equal strata sizes: resampling within each opponent preserves design.
        rng = np.random.default_rng(args.seed+20000)
        means = np.zeros(5000)
        for x in clusters.values():
            for start in range(0, 5000, 128):
                n = min(128, 5000-start)
                means[start:start+n] += x[rng.integers(len(x), size=(n, len(x)))].mean(1)/3
        ci = np.quantile(means, [.025, .975]).tolist()
        result["pooled"] = {"win_rate_difference": float(np.concatenate(list(clusters.values())).mean()),
                                "bootstrap_95": ci, "verdict": verdict(ci), "resamples": 5000,
                                "seed": args.seed+20000, "clusters": args.battles//2*3}
        paired["comparisons"][name] = result
    Path(args.paired_out).write_text(json.dumps(paired, indent=2)+"\n")
    report["valid"] = all(r["valid"] for o in report["opponents"].values()
                          for r in o["arms"].values())
    save()
    if not report["valid"]:
        raise RuntimeError("Invalid benchmark; preserve report as _run1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--concurrent", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", default="reports/w4_5_ablation.json")
    parser.add_argument("--paired-out", default="reports/w4_5_paired_comparison.json")
    asyncio.run(evaluate(parser.parse_args()))
