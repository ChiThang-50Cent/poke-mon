"""Paired head-to-head: NeuralPlayer vs a pretrained Metamon agent, LOCAL server only.

The Metamon agent runs in a worker subprocess (``metamon_h2h_worker.py``) that accepts
challenges by username; the neural player runs in this process and challenges it once per
scheduled battle. Team pairs come from the runner's seeded, swapped schedule
(``(A, B)`` then ``(B, A)``), so results are comparable across neural arms and can be fed
to ``cluster_differences`` in ``scripts/evaluate_w4_5.py``. The neural player is always the
challenger (``player_challenges`` = True on every row).

Example:
    CUDA_VISIBLE_DEVICES="" uv run python scripts/run_metamon_h2h.py \
        --player neural:models/pokedecision-v2-pile --metamon Minikazam \
        --teams data/teams_w5/dev --battles 4 --seed 1 --out runs/h2h_smoke
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pokeai  # noqa: F401  (must precede metamon imports)
from pokeai.env.runner import RunConfig, _build_schedule, summarize

SERVER_WS = "ws://localhost:8000/showdown/websocket"
SERVER_HTTP = "http://localhost:8000/action.php?"
DEFAULT_PREVIEW = (
    "third_party/metamon/metamon/backend/team_preview/gen9ou_high_elo_v4/best_model.pt"
)


def build_h2h_schedule(teams: str, seed: int, battles: int):
    """Neural/Metamon team draws per battle: runner schedule with swapped pairs."""
    from pokeai.env.teams import load_teamset

    config = RunConfig(
        teams=teams, battles=battles, schedule_seed=seed, swap_team_pairs=True
    )
    schedule = _build_schedule(load_teamset(teams), config)
    assert schedule is not None
    return schedule


def validate_team_log(log_path: Path, schedule) -> None:
    """The Metamon side must have drawn exactly its scheduled team for every battle."""
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    expected = [
        {"yield": i, "name": opp[1],
         "sha256": hashlib.sha256(opp[0].encode()).hexdigest()}
        for i, (_, opp) in enumerate(schedule)
    ]
    if rows != expected:
        raise RuntimeError(
            f"Metamon team draws differ from schedule: got {rows}, expected {expected}"
        )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def play(
    args, schedule, username: str, opp_username: str, out: Path
) -> tuple[list[dict], Any]:

    from poke_env import AccountConfiguration
    from poke_env.concurrency import handle_threaded_coroutines
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from poke_env.teambuilder import Teambuilder

    from pokeai.agents.neural_player import NeuralPlayer

    class FixedTeam(Teambuilder):
        def __init__(self, packed: str) -> None:
            self.packed = packed

        def yield_team(self) -> str:
            return self.packed

    async def run() -> tuple[list[dict], Any]:
        preview = None
        if args.team_preview_checkpoint:
            from metamon.backend.team_preview.preview import TeamPreviewModel

            preview = TeamPreviewModel.load_from_checkpoint(
                args.team_preview_checkpoint, device=args.device, use_argmax=True
            )
        player_kwargs = {
            "battle_format": "gen9ou", "max_concurrent_battles": 1,
            "account_configuration": AccountConfiguration(username, None),
            "server_configuration": ServerConfiguration(SERVER_WS, SERVER_HTTP),
            "log_level": logging.ERROR,
        }
        if args.player.startswith("baseline:"):
            from pokeai.env.runner import _counted_baseline

            player = _counted_baseline(args.player.removeprefix("baseline:"))(**player_kwargs)
        else:
            player = NeuralPlayer(
                model_dir=args.player.removeprefix("neural:"), sample=args.sample,
                temperature=args.temperature, device=args.device,
                team_preview_model=preview, **player_kwargs,
            )
        records: list[dict] = []
        try:
            with (out / "battles.jsonl").open("x") as log:
                for index, ((team, team_name), (opp_team, opp_name)) in zip(
                        args.indices, schedule):
                    start = time.perf_counter()
                    previous = next(reversed(player.battles), None)
                    illegal_before = getattr(player, "illegal_actions", 0)
                    player.decisions = 0
                    player.last_server_error = None
                    record: dict[str, Any] = {
                        "battle_id": f"{out.name}-{index:06d}", "matchup_index": index,
                        "run_id": out.name, "format": "gen9ou", "arm": args.player,
                        "player": args.player, "opponent": f"metamon:{args.metamon}",
                        "player_team": team_name, "opponent_team": opp_name,
                        "player_team_sha256": hashlib.sha256(team.encode()).hexdigest(),
                        "opponent_team_sha256": hashlib.sha256(opp_team.encode()).hexdigest(),
                        "player_challenges": True, "team_preview": None,
                        "protocol_retries": {"player": [], "opponent": []},
                        "winner": None, "turns": 0, "decisions_player": 0,
                        "illegal_actions": 0, "duration_s": 0.0, "error": None,
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                    try:
                        player.update_team(FixedTeam(team))
                        await asyncio.wait_for(
                            player.send_challenges(opp_username, 1),
                            timeout=args.timeout_s,
                        )
                        tag = next(reversed(player.battles), None)
                        if tag is None or tag == previous:
                            raise RuntimeError("no battle was created")
                        battle = player.battles[tag]
                        record["battle_id"] = tag
                        if not battle.finished:
                            raise RuntimeError("challenge returned before battle finished")
                        if player.last_server_error is not None:
                            raise RuntimeError(player.last_server_error)
                        record["winner"] = (
                            "player" if battle.won else "opponent" if battle.lost else "tie"
                        )
                        record["turns"] = battle.turn
                        record["team_preview"] = getattr(
                            player, "team_preview_records", {}).get(tag)
                        record["protocol_retries"]["player"] = player.protocol_retries.get(
                            tag, []
                        )
                    except Exception as exc:  # noqa: BLE001 - recorded, run aborts below
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["decisions_player"] = player.decisions
                    record["illegal_actions"] = getattr(player, "illegal_actions", 0) - illegal_before
                    record["duration_s"] = time.perf_counter() - start
                    records.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if record["error"]:
                        break  # Metamon's team cursor is now out of sync with the schedule
        finally:
            with contextlib.suppress(Exception):  # best-effort cleanup
                for battle in player.battles.values():
                    if not battle.finished:
                        await player.ps_client.send_message("/forfeit", battle.battle_tag)
                await asyncio.wait_for(player.ps_client.stop_listening(), timeout=5)
        return records, player

    return await handle_threaded_coroutines(run())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", required=True, help="neural:<model_dir> | baseline:<Name>")
    parser.add_argument("--metamon", required=True, help="Pretrained model, e.g. Minikazam")
    parser.add_argument("--metamon-checkpoint", type=int)
    parser.add_argument("--metamon-temperature", type=float, default=1.0)
    parser.add_argument("--metamon-attention", choices=("native", "flex"), default="native")
    parser.add_argument("--teams", default="data/teams")
    parser.add_argument("--battles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--team-preview-checkpoint", default=DEFAULT_PREVIEW)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1,
                        help="play only swapped pairs p with p %% lanes == lane (own worker); "
                             "matchup_index stays global, merge lanes by concatenation")
    args = parser.parse_args()
    if not args.player.startswith(("neural:", "baseline:")):
        parser.error("--player must be neural:<model_dir> or baseline:<Name>")
    if args.battles < 2 or args.battles % 2:
        parser.error("--battles must be a positive even number (swapped team pairs)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("METAMON_CACHE_DIR", str(Path(".cache/metamon").resolve()))

    full = build_h2h_schedule(args.teams, args.seed, args.battles)
    if not 0 <= args.lane < args.lanes:
        parser.error("--lane must be in [0, --lanes)")
    args.indices = [i for i in range(len(full)) if (i // 2) % args.lanes == args.lane]
    schedule = [full[i] for i in args.indices]
    (out / "schedule.json").write_text(json.dumps(
        [{"packed": opp[0], "name": opp[1]} for _, opp in schedule]
    ))
    username = "w" + uuid.uuid4().hex[:17]
    opp_username = "mm" + uuid.uuid4().hex[:12]
    ready = out / "worker.ready"
    info_path = out / "metamon_info.json"
    team_log = out / "metamon_teams.jsonl"
    started = time.perf_counter()
    with (out / "metamon_worker.log").open("w") as worker_log:
        worker = subprocess.Popen(
            [sys.executable, "scripts/metamon_h2h_worker.py", "--model", args.metamon,
             "--username", opp_username, "--opponent-username", username,
             "--schedule", str(out / "schedule.json"), "--info-out", str(info_path),
             "--team-log", str(team_log), "--ready-file", str(ready),
             "--temperature", str(args.metamon_temperature),
             "--attention", args.metamon_attention,
             *(["--checkpoint", str(args.metamon_checkpoint)]
               if args.metamon_checkpoint is not None else [])],
            stdout=worker_log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout_s
            while not ready.exists():
                if worker.poll() is not None:
                    raise RuntimeError(
                        f"Metamon worker exited early (code {worker.returncode}); "
                        f"see {out / 'metamon_worker.log'}"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError("Metamon worker startup timed out")
                time.sleep(1)
            time.sleep(5)  # let the acceptor finish logging in
            records, player = asyncio.run(play(args, schedule, username, opp_username, out))
            try:
                worker.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()

    summary = summarize(records, time.perf_counter() - started, {
        k: v for k, v in vars(args).items()
    })
    metamon = json.loads(info_path.read_text())
    summary["metamon"] = metamon
    summary["backend"] = metamon["backend"]
    summary["neural"] = {
        "player": args.player,
        "checkpoint_sha256": getattr(player, "checkpoint_sha256", None),
        "backend": "metamon",
        "illegal_actions": getattr(player, "illegal_actions", 0),
        "fallbacks": dict(getattr(player, "fallbacks", {})),
    }
    summary["illegal_actions"] = sum(r["illegal_actions"] for r in records)
    summary["errors"] = [r["error"] for r in records if r["error"]]
    summary["server"] = SERVER_WS
    summary["schedule"] = {"seed": args.seed, "teams": args.teams, "swap_team_pairs": True,
                           "player_challenges": True}
    results = [
        json.loads(line.removeprefix("METAMON_RESULTS "))
        for line in (out / "metamon_worker.log").read_text().splitlines()
        if line.startswith("METAMON_RESULTS ")
    ]
    summary["metamon_results"] = results[-1] if results else None
    problems = list(summary["errors"])
    if summary["metamon_results"] is not None and not problems:
        win_rates = [
            v for k, v in summary["metamon_results"].items() if k.startswith("Average Win Rate")
        ]
        if win_rates != [summary["losses"] / summary["completed"]]:
            problems.append(
                f"Metamon win rate {win_rates} disagrees with neural-side "
                f"losses {summary['losses']}/{summary['completed']}"
            )
    if len(records) != len(schedule):
        problems.append(f"only {len(records)}/{len(schedule)} battles ran")
    elif worker.returncode != 0:
        problems.append(f"Metamon worker exit code {worker.returncode}")
    if not problems:
        try:
            validate_team_log(team_log, schedule)
        except RuntimeError as exc:
            problems.append(str(exc))
    if summary["illegal_actions"]:
        problems.append(f"{summary['illegal_actions']} illegal actions")
    summary["valid"] = not problems
    summary["problems"] = problems
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if problems:
        print("FAIL: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
