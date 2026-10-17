"""Paired local A/B evaluation of frozen W3 turn policy with learned preview.

A uses the existing random lead; T changes only preview to the vendored model.
No training or checkpoint selection is performed by this script.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pokeai.env.collector import git_commit
from pokeai.env.runner import RunConfig, run_battles, wilson_interval
from pokeai.repro import source_fingerprint

POLICY = Path("models/pokedecision-v0")
PREVIEW = Path(
    "third_party/metamon/metamon/backend/team_preview/"
    "gen9ou_high_elo_v4/best_model.pt"
)
OPPONENTS = ("RandomBaseline", "Grunt", "GymLeader")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(
    a: list[dict[str, Any]], t: list[dict[str, Any]], *,
    battles: int, seed: int, resamples: int,
) -> dict[str, Any]:
    """Align on matchup_index, check actual team hashes, then bootstrap pairs.

    The estimand is T minus A win rate: a tie is a non-win, consistent with
    runner Wilson intervals. Failed/unmatched pairs never enter the estimate.
    """
    if resamples < 2000:
        raise ValueError("paired bootstrap requires at least 2000 resamples")
    problems: list[str] = []

    def indexed(records: list[dict[str, Any]], arm: str) -> dict[int, dict]:
        result = {}
        for record in records:
            index = record.get("matchup_index")
            if type(index) is not int or not 0 <= index < battles:
                problems.append(f"{arm}: invalid matchup_index {index!r}")
                continue
            if index in result:
                problems.append(f"{arm}: duplicate matchup_index {index}")
            result[index] = record
        if set(result) != set(range(battles)):
            problems.append(f"{arm}: incomplete matchup_index coverage")
        return result

    arms = (indexed(a, "A"), indexed(t, "T"))
    differences = []
    matched = 0
    for index in sorted(arms[0].keys() & arms[1].keys()):
        control, treatment = arms[0][index], arms[1][index]
        keys = (
            "player_team", "opponent_team",
            "player_team_sha256", "opponent_team_sha256",
        )
        if any(not control.get(key) or control[key] != treatment.get(key)
               for key in keys):
            problems.append(f"matchup_index {index}: team pair mismatch/missing hash")
            continue
        matched += 1
        if any(record.get("error") is not None
               or record.get("winner") not in ("player", "opponent", "tie")
               for record in (control, treatment)):
            continue
        differences.append(
            int(treatment["winner"] == "player")
            - int(control["winner"] == "player")
        )
    delta = np.asarray(differences, dtype=np.float64)
    interval = None
    if delta.size:
        rng = np.random.default_rng(seed)
        # Resample complete aligned pairs; bounded memory even for larger runs.
        estimates = np.empty(resamples)
        for start in range(0, resamples, 128):
            count = min(128, resamples - start)
            indices = rng.integers(delta.size, size=(count, delta.size))
            estimates[start:start + count] = delta[indices].mean(axis=1)
        interval = np.quantile(estimates, [0.025, 0.975]).tolist()
    return {
        "valid": not problems and matched == battles and len(delta) == battles,
        "team_pairs_equal": not problems and matched == battles,
        "matched_team_pairs": matched,
        "completed_pairs": len(delta),
        "problems": problems,
        "estimand": "T minus A win rate; ties count as non-wins",
        "win_rate_difference": float(delta.mean()) if delta.size else None,
        "bootstrap_95": interval,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "pairing_key": "matchup_index",
    }


def arm_report(summary: dict, records: list[dict], battles: int) -> dict:
    completed = [r for r in records if r.get("error") is None
                 and r.get("winner") in ("player", "opponent", "tie")]
    outcomes = Counter(r["winner"] for r in completed)
    errors = [{"matchup_index": r.get("matchup_index"), "error": r.get("error"),
               "winner": r.get("winner")}
              for r in records if r not in completed]
    neural = summary.get("neural", {}).get("player", {})
    illegal = neural.get("illegal_actions")
    retry_counts = summary.get(
        "protocol_retry_counts", {"player": 0, "opponent": 0}
    )
    leads = Counter(r.get("team_preview", {}).get("chosen_lead") for r in records)
    leads.pop(None, None)
    sources = Counter(r.get("team_preview", {}).get("source") for r in records)
    sources.pop(None, None)
    total = len(completed)
    valid = (
        len(records) == battles and total == battles and not errors
        and summary.get("attempted") == battles
        and summary.get("completed") == battles
        and summary.get("error_count") == 0 and illegal == 0
        and all(summary.get(field) == outcomes[outcome] for field, outcome in
                (("wins", "player"), ("losses", "opponent"), ("ties", "tie")))
    )
    return {
        "valid": valid,
        "attempted": len(records), "completed": total,
        "wins": outcomes["player"], "losses": outcomes["opponent"],
        "ties": outcomes["tie"], "error_count": len(errors), "errors": errors,
        "illegal_actions": illegal,
        "protocol_retries": retry_counts,
        "protocol_retry_evidence": summary.get(
            "protocol_retry_evidence", {"player": [], "opponent": []}
        ),
        "trapped_rejected_choices": sum(retry_counts.values()),
        "win_rate": outcomes["player"] / total if total else None,
        "win_rate_wilson_95": list(wilson_interval(outcomes["player"], total)),
        "lead_source_counts": dict(sorted(sources.items())),
        "top_lead_species": [
            {"species": species, "count": count}
            for species, count in sorted(leads.items(), key=lambda x: (-x[1], x[0]))[:20]
        ],
        "run_summary": summary,
    }


def prior_report_metadata(path: str | Path) -> dict[str, Any]:
    """Reference prior evidence without modifying or reclassifying its results."""
    path = Path(path)
    raw = path.read_bytes()
    prior = json.loads(raw)
    arms = {
        f"{opponent}/{arm}": {
            "valid": result.get("valid"),
            "errors": result.get("error_count"),
            "illegal_actions": result.get("illegal_actions"),
        }
        for opponent, matchup in prior.get("opponents", {}).items()
        for arm, result in matchup.get("arms", {}).items()
    }

    def total(field: str) -> int | None:
        values = [arm[field] for arm in arms.values()]
        return sum(values) if values and all(
            type(value) is int for value in values
        ) else None

    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "valid": prior.get("valid"),
        "errors": total("errors"),
        "illegal_actions": total("illegal_actions"),
        "wall_time_s": prior.get("wall_time_s"),
        "arms": arms,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--out", default="reports/w4_team_preview.json")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--teams", default="data/teams")
    parser.add_argument("--preview-checkpoint", default=str(PREVIEW))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--prior-report",
        help="Preserved prior-run report to fingerprint and summarize, never overwrite",
    )
    args = parser.parse_args(argv)
    if args.battles < 1 or not 1 <= args.concurrent <= 8:
        parser.error("battles must be positive and concurrent must be in [1, 8]")
    if args.bootstrap_resamples < 2000:
        parser.error("bootstrap-resamples must be at least 2000")
    if len(set(args.opponents)) != len(args.opponents):
        parser.error("opponents must be unique")
    return args


async def evaluate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(1)
    output = Path(args.out)
    prior_path = getattr(args, "prior_report", None)
    if prior_path and Path(prior_path).resolve() == output.resolve():
        raise ValueError("--prior-report must differ from --out")
    prior = prior_report_metadata(prior_path) if prior_path else None
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "valid": False, "passed": False,
        "policy": str(POLICY), "preview_checkpoint": args.preview_checkpoint,
        "fingerprints": {
            "git_commit": git_commit("."),
            "metamon_commit": git_commit("third_party/metamon"),
            "showdown_commit": git_commit("third_party/metamon/server/pokemon-showdown"),
            "policy_files_sha256": {name: sha256(POLICY / name) for name in
                                    ("best.pt", "config.yaml", "tokenizer.json")},
            "preview_checkpoint_sha256": sha256(Path(args.preview_checkpoint)),
            "team_manifest_sha256": sha256(Path(args.teams) / "gen9ou/manifest.json"),
            "source_sha256": source_fingerprint(),
        },
        "python": platform.python_version(), "torch": torch.__version__,
        "seed": args.seed, "battles_per_arm_per_opponent": args.battles,
        "concurrent": args.concurrent, "team_pool": args.teams,
        "prior_report": prior,
        "methodology": {
            "protocol_retries": "Expected trapped-choice rejection discovery is "
                                "reported separately as protocol retries and total "
                                "trapped rejected choices, not as illegal actions. "
                                "Unexpected server rejections remain errors.",
            "A": "frozen pokedecision-v0 greedy turn policy + random-default preview",
            "T": "same frozen greedy turn policy + learned argmax preview",
            "device": "cpu", "torch_threads": 1, "training": False,
            "no_leakage": "Preview inputs contain own species/moves/abilities/items "
                          "and public opponent species only; no hidden opponent sets, "
                          "future turns, or outcomes are inputs.",
            "pairing": "Exact player/opponent team identities and packed-team SHA256s "
                       "must match by matchup_index; both arms use the same schedule_seed.",
            "rng_caveat": "Showdown and heuristic RNG are uncontrolled. This pairs "
                          "team schedules, not simulator trajectories or opponent actions.",
            "intervals": "Two-sided 95% Wilson win-rate intervals and seeded paired "
                         "percentile bootstrap T-A; ties count as non-wins.",
            "validity": "Exact requested attempted/completed counts, zero errors and "
                        "illegal actions in both arms, and complete exact team pairing.",
        },
        "opponents": {},
    }

    def save() -> None:
        report["wall_time_s"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for opponent in args.opponents:
        seed = args.seed + OPPONENTS.index(opponent)
        bootstrap_seed = args.seed + 10000 + OPPONENTS.index(opponent)
        result: dict[str, Any] = {
            "schedule_seed": seed, "arm_rng_seed": seed,
            "bootstrap_seed": bootstrap_seed, "arms": {},
        }
        report["opponents"][opponent] = result
        records = {}
        for arm in ("A", "T"):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            try:
                summary = await run_battles(RunConfig(
                    player=f"neural:{POLICY}", opponent=opponent,
                    battles=args.battles, concurrent=args.concurrent,
                    teams=args.teams, timeout_s=args.timeout_s,
                    device="cpu", sample=False, schedule_seed=seed,
                    team_preview_checkpoint=(
                        args.preview_checkpoint if arm == "T" else None
                    ),
                    team_preview_argmax=True,
                ))
                log = Path(summary["config"]["out"]) / summary["run_id"] / "battles.jsonl"
                with log.open() as stream:
                    records[arm] = [json.loads(line) for line in stream if line.strip()]
                result["arms"][arm] = arm_report(summary, records[arm], args.battles)
            except Exception as exc:  # noqa: BLE001 - persist invalid evaluation evidence
                result["arms"][arm] = {"valid": False,
                                       "fatal_error": f"{type(exc).__name__}: {exc}"}
                records[arm] = []
            save()
        result["paired"] = paired_bootstrap(
            records["A"], records["T"], battles=args.battles,
            seed=bootstrap_seed, resamples=args.bootstrap_resamples,
        )
        result["valid"] = result["paired"]["valid"] and all(
            item["valid"] for item in result["arms"].values()
        )
        save()
        print(json.dumps({"opponent": opponent, "valid": result["valid"],
                          "paired": result["paired"]}), flush=True)
    report["valid"] = report["passed"] = all(
        item["valid"] for item in report["opponents"].values()
    )
    save()
    if not report["valid"]:
        raise RuntimeError(f"W4 team preview evaluation invalid; see {output}")
    return report


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
