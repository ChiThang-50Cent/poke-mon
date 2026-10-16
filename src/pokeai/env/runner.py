"""Run bounded local baseline battles and write reproducible JSONL metrics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    player: str = "GymLeader"
    opponent: str = "RandomBaseline"
    battles: int = 100
    concurrent: int = 8
    teams: str = "data/teams"
    timeout_s: float = 120.0
    out: str = "runs"
    save_trajectories: bool = False
    sample: bool = False
    temperature: float = 1.0
    device: str = "cpu"
    team_preview_checkpoint: str | None = None
    team_preview_argmax: bool = False
    schedule_seed: int | None = None
    swap_team_pairs: bool = False
    alternate_challenger: bool = False

    def __post_init__(self) -> None:
        if self.battles < 1 or self.concurrent < 1:
            raise ValueError("battles and concurrent must be positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if self.swap_team_pairs and (
            self.schedule_seed is None or self.battles % 2
        ):
            raise ValueError("swapped pairs require a seed and an even battle count")


def wilson_interval(wins: int, total: int) -> tuple[float, float]:
    """Return the two-sided Wilson 95% interval, or [0, 1] without trials."""
    if total < 0 or not 0 <= wins <= total:
        raise ValueError("expected 0 <= wins <= total")
    if total == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize(
    records: list[dict[str, Any]], wall_time_s: float, config: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate outcomes; errored attempts are excluded from outcome rates."""
    completed = [record for record in records if record["error"] is None]
    counts = {
        outcome: sum(record["winner"] == outcome for record in completed)
        for outcome in ("player", "opponent", "tie")
    }
    total = len(completed)
    decisions = sum(record["decisions_player"] for record in records)
    lead_sources: dict[str, int] = {}
    for record in records:
        preview = record.get("team_preview")
        if preview is not None:
            source = preview["source"]
            lead_sources[source] = lead_sources.get(source, 0) + 1
    retry_evidence = {
        role: [
            {
                **evidence,
                "battle_id": record["battle_id"],
                "matchup_index": record["matchup_index"],
            }
            for record in records
            for evidence in record.get("protocol_retries", {}).get(role, [])
        ]
        for role in ("player", "opponent")
    }
    return {
        "attempted": len(records),
        "completed": total,
        "wins": counts["player"],
        "losses": counts["opponent"],
        "ties": counts["tie"],
        "error_count": len(records) - total,
        "win_rate": counts["player"] / total if total else 0.0,
        "lose_rate": counts["opponent"] / total if total else 0.0,
        "tie_rate": counts["tie"] / total if total else 0.0,
        "win_rate_wilson_95": list(wilson_interval(counts["player"], total)),
        "wall_time_s": wall_time_s,
        "games_per_s": total / wall_time_s if wall_time_s > 0 else 0.0,
        "decisions_player": decisions,
        "decisions_per_s": decisions / wall_time_s if wall_time_s > 0 else 0.0,
        "lead_source_counts": lead_sources,
        "protocol_retry_counts": {
            role: len(evidence) for role, evidence in retry_evidence.items()
        },
        "protocol_retry_evidence": retry_evidence,
        "config": config,
    }


def _sample_team(teamset: Any) -> tuple[str, str]:
    packed = teamset.yield_team()
    source = teamset.most_recent_team_file
    identity = Path(source).name if source else hashlib.sha256(packed.encode()).hexdigest()
    return packed, identity


def _build_schedule(
    teamset: Any, config: RunConfig
) -> list[tuple[tuple[str, str], tuple[str, str]]] | None:
    """Freeze paired team draws by matchup index without touching global RNG state.

    Without a schedule seed, leave the original per-battle TeamSet sampling intact.
    A seeded schedule uses a stable file ordering and the same packing path as TeamSet.
    """
    if config.schedule_seed is None:
        return None
    rng = random.Random(config.schedule_seed)
    files = sorted(teamset.team_files)
    packed_teams: dict[str, tuple[str, str]] = {}

    def draw() -> tuple[str, str]:
        filename = rng.choice(files)
        if filename not in packed_teams:
            packed = teamset.join_team(
                teamset.parse_showdown_team(Path(filename).read_text())
            )
            packed_teams[filename] = (packed, Path(filename).name)
        return packed_teams[filename]

    if config.swap_team_pairs:
        schedule = []
        for _ in range(config.battles // 2):
            first, second = draw(), draw()
            schedule.extend(((first, second), (second, first)))
        return schedule
    return [(draw(), draw()) for _ in range(config.battles)]


def _prepare_revival_request(battle: Any) -> None:
    """Restrict Revival Blessing selections to the server's legal fainted targets."""
    if battle.reviving:
        # The poke-env fork includes every non-active reserve here, even healthy ones.
        # A revival selection uses switch syntax, but is NOT an ordinary switch.
        battle.available_switches[:] = [
            pokemon for pokemon in battle.available_switches if pokemon.fainted
        ]


def _protocol_choice_evidence(battle: Any, order: Any) -> dict[str, Any]:
    """Snapshot only the acting player's request and public view before sending."""
    request = (
        getattr(battle, "last_request", None)
        or getattr(battle, "_last_request", {})
    )
    active = request.get("active") or [{}]
    pokemon = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    return {
        "order": order if isinstance(order, str) else order.message,
        "trapped": bool(active[0].get("trapped", False)),
        "maybeTrapped": bool(active[0].get("maybeTrapped", False)),
        "rqid": request.get("rqid"),
        "turn": battle.turn,
        "active_species": getattr(pokemon, "species", None),
        "active_ability": getattr(pokemon, "ability", None),
        "opponent_species": getattr(opponent, "species", None),
        "opponent_ability": getattr(opponent, "ability", None),
    }


def _is_expected_trapped_retry(
    error_message: str, evidence: dict[str, Any] | None
) -> bool:
    """Recognize hidden-trap discovery, never an already-known illegal switch.

    poke-env owns the retry handshake and awaits the server's revised request.
    These rejected choices are protocol retries, not policy legality failures.
    """
    return bool(
        evidence
        and evidence["maybeTrapped"]
        and not evidence["trapped"]
        and evidence["order"].startswith("/choose switch ")
        and error_message.startswith((
            "[Unavailable choice] Can't switch: The active Pokémon is trapped",
            "[Invalid choice] Can't switch: The active Pokémon is trapped",
        ))
    )


def _counted_baseline(name: str) -> type:
    from metamon.baselines import get_baseline

    base = get_baseline(name)

    class CountedBaseline(base):
        decisions = 0
        last_server_error: str | None = None
        collector: Any = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.protocol_retries: dict[str, list[dict[str, Any]]] = {}
            self._protocol_choices: dict[str, dict[str, Any]] = {}
            super().__init__(*args, **kwargs)

        async def _handle_battle_message(
            self, split_messages: list[list[str]]
        ) -> None:
            tag = split_messages[0][0].removeprefix(">")
            # Process in order even without collection: a request and error can
            # share a batch, and recognition must use that request's sent order.
            for message in split_messages[1:]:
                if (
                    self.collector is not None
                    and len(message) > 1 and message[1] != "request"
                ):
                    self.collector.events.append(message)
                if len(message) > 1 and message[1] == "error":
                    evidence = self._protocol_choices.get(tag)
                    if len(message) > 2 and _is_expected_trapped_retry(
                        message[2], evidence
                    ):
                        self.protocol_retries.setdefault(tag, []).append({
                            **evidence, "error": "|".join(message),
                        })
                        # One rejected choice can account for only one retry.
                        self._protocol_choices.pop(tag, None)
                    else:
                        self.last_server_error = "|".join(message)
                await super()._handle_battle_message([split_messages[0], message])

        async def _handle_battle_request(
            self, battle: Any, maybe_default_order: bool = False
        ) -> None:
            self._protocol_choices.pop(battle.battle_tag, None)
            _prepare_revival_request(battle)
            if battle.reviving:
                # Native default orders and cached Pokemon status can select a
                # previously revived target. The request is authoritative.
                maybe_default_order = False
            await super()._handle_battle_request(battle, maybe_default_order)

        def choose_move(self, battle: Any) -> Any:
            self.decisions += 1
            if getattr(battle, "reviving", False):
                from pokeai.agents.neural_player import RevivalOrder

                request = getattr(battle, "last_request", None) or battle._last_request
                targets = [
                    index for index, pokemon in
                    enumerate(request["side"]["pokemon"], 1)
                    if pokemon["condition"].endswith(" fnt")
                ]
                if not targets:
                    raise RuntimeError("Revival request has no fainted target")
                order = RevivalOrder(random.choice(targets))
            else:
                order = super().choose_move(battle)
            self._protocol_choices[battle.battle_tag] = _protocol_choice_evidence(
                battle, order
            )
            if self.collector is not None:
                try:
                    self.collector.record(battle, order)
                except Exception as exc:  # noqa: BLE001 - preserve battle, reject data
                    self.collector.error = f"{type(exc).__name__}: {exc}"
            return order

        def teampreview(self, battle: Any) -> str:
            order = super().teampreview(battle)
            if self.collector is not None:
                selection = order.removeprefix("/team ").strip()
                self.collector.team_preview = {
                    "chosen_lead": int(selection[0]) if selection else None,
                    "order": selection,
                }
            return order

    return CountedBaseline


def _player_type(
    name: str, config: RunConfig, team_preview_model: Any = None
) -> Any:
    if not name.startswith("neural:"):
        return _counted_baseline(name)
    from functools import partial

    from pokeai.agents.neural_player import NeuralPlayer

    return partial(
        NeuralPlayer, model_dir=name.removeprefix("neural:"),
        sample=config.sample, temperature=config.temperature, device=config.device,
        team_preview_model=team_preview_model,
    )


async def _close_players(players: list[Any]) -> list[str]:
    async def close(player: Any) -> None:
        for battle in player.battles.values():
            if not battle.finished:
                await player.ps_client.send_message("/forfeit", battle.battle_tag)
        await player.ps_client.stop_listening()

    results = await asyncio.gather(
        *(asyncio.wait_for(close(player), timeout=5) for player in players),
        return_exceptions=True,
    )
    return [
        f"cleanup: {type(item).__name__}: {item}"
        for item in results
        if isinstance(item, BaseException)
    ]


async def _run_on_poke_loop(
    config: RunConfig, run_id: str, output: Path
) -> dict[str, Any]:
    from poke_env import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from poke_env.teambuilder import Teambuilder

    from pokeai.env.teams import load_teamset

    class FixedTeam(Teambuilder):
        # Freeze each TeamSet sample so both its identity and battle team agree.
        # Passing packed strings directly hits the fork's ConstantTeambuilder bug:
        # its battle initializer expects _mons, absent for packed-string input.
        def __init__(self, packed: str) -> None:
            self.packed = packed

        def yield_team(self) -> str:
            return self.packed

    # Explicitly local: neither battle traffic nor authentication uses the public service.
    server = ServerConfiguration(
        "ws://localhost:8000/showdown/websocket", "http://localhost:8000/action.php?"
    )
    teamset = load_teamset(config.teams)
    schedule = _build_schedule(teamset, config)
    team_preview_model = None
    if config.team_preview_checkpoint is not None:
        import torch
        from metamon.backend.team_preview.preview import TeamPreviewModel

        team_preview_model = TeamPreviewModel.load_from_checkpoint(
            config.team_preview_checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_argmax=config.team_preview_argmax,
        )
    player_type = _player_type(config.player, config, team_preview_model)
    opponent_type = _player_type(config.opponent, config, team_preview_model)
    records: list[dict[str, Any]] = []
    next_index = iter(range(config.battles))
    started = time.perf_counter()
    neural_players: dict[str, list[Any]] = {"player": [], "opponent": []}
    manifest: dict[str, Any] | None = None
    if config.save_trajectories:
        from pokeai.env.collector import atomic_write, git_commit

        manifest = {
            "run_id": run_id,
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "metamon_commit": git_commit("third_party/metamon"),
            "showdown_commit": git_commit("third_party/metamon/server/pokemon-showdown"),
            "trajectory_files": [],
            "expected_trajectories": 2 * config.battles,
            "saved_trajectories": 0,
            "missing": [],
        }

        def write_manifest() -> None:
            assert manifest is not None
            atomic_write(output / "manifest.json", json.dumps(manifest, indent=2).encode())

        write_manifest()

    with (output / "battles.jsonl").open("x") as log:
        async def lane() -> None:
            players: list[Any] = []
            try:
                for index in next_index:
                    start = time.perf_counter()
                    previous_tag = None
                    if players:
                        previous_tag = next(reversed(players[0].battles), None)
                        players[0].decisions = 0
                    record: dict[str, Any] = {
                        "battle_id": f"{run_id}-{index:06d}",
                        "matchup_index": index,
                        "run_id": run_id,
                        "format": "gen9ou",
                        "player": config.player,
                        "opponent": config.opponent,
                        "player_team": None,
                        "opponent_team": None,
                        "player_team_sha256": None,
                        "opponent_team_sha256": None,
                        "team_preview": None,
                        "protocol_retries": {"player": [], "opponent": []},
                        "values": {"player": [], "opponent": []},
                        "winner": None,
                        "turns": 0,
                        "decisions_player": 0,
                        "duration_s": 0.0,
                        "error": None,
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                    try:
                        if schedule is None:
                            player_draw = _sample_team(teamset)
                            opponent_draw = _sample_team(teamset)
                        else:
                            player_draw, opponent_draw = schedule[index]
                        player_team, record["player_team"] = player_draw
                        opponent_team, record["opponent_team"] = opponent_draw
                        record["player_team_sha256"] = hashlib.sha256(
                            player_team.encode()
                        ).hexdigest()
                        record["opponent_team_sha256"] = hashlib.sha256(
                            opponent_team.encode()
                        ).hexdigest()
                        if not players:
                            for cls in (player_type, opponent_type):
                                players.append(
                                    cls(
                                        battle_format="gen9ou",
                                        max_concurrent_battles=1,
                                        account_configuration=AccountConfiguration(
                                            "w" + uuid.uuid4().hex[:17], None
                                        ),
                                        server_configuration=server,
                                        log_level=logging.ERROR,
                                    )
                                )
                                if hasattr(players[-1], "neural_stats"):
                                    role = "player" if len(players) == 1 else "opponent"
                                    neural_players[role].append(players[-1])
                        player, opponent = players
                        player.decisions = 0
                        for participant in players:
                            participant.last_server_error = None
                            if config.save_trajectories:
                                from pokeai.env.collector import TrajectoryCollector

                                participant.collector = TrajectoryCollector()
                        player.update_team(FixedTeam(player_team))
                        opponent.update_team(FixedTeam(opponent_team))
                        player.randomize()
                        opponent.randomize()
                        challenger, challenged = (
                            (opponent, player)
                            if config.alternate_challenger and index % 2
                            else (player, opponent)
                        )
                        record["player_challenges"] = challenger is player
                        await asyncio.wait_for(
                            challenger.battle_against(challenged, n_battles=1),
                            timeout=config.timeout_s,
                        )
                        battle = player.battles[next(reversed(player.battles))]
                        if not battle.finished:
                            raise RuntimeError(
                                "battle task returned before terminal result"
                            )
                        for participant in players:
                            if participant.last_server_error is not None:
                                raise RuntimeError(participant.last_server_error)
                        record["winner"] = (
                            "player" if battle.won else "opponent" if battle.lost else "tie"
                        )
                        if manifest is not None:
                            from pokeai.env.collector import save_battle

                            record["battle_id"] = battle.battle_tag
                            entries = save_battle(
                                output, run_id, players, record,
                                manifest["metamon_commit"],
                            )
                            manifest["trajectory_files"].extend(entries)
                            manifest["saved_trajectories"] += len(entries)
                    except Exception as exc:  # noqa: BLE001 - log each failed attempt
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        if isinstance(exc, TimeoutError):
                            record["error"] = f"TimeoutError: exceeded {config.timeout_s}s"
                            for participant in players:
                                server_error = getattr(
                                    participant, "last_server_error", None
                                )
                                if server_error:
                                    record["error"] += "; " + server_error
                    finally:
                        if players:
                            player = players[0]
                            record["decisions_player"] = player.decisions
                            latest_tag = next(reversed(player.battles), None)
                            if latest_tag is not None and latest_tag != previous_tag:
                                battle = player.battles[latest_tag]
                                record["battle_id"] = battle.battle_tag
                                record["turns"] = battle.turn
                                record["team_preview"] = getattr(
                                    player, "team_preview_records", {}
                                ).get(latest_tag)
                                for role, participant in zip(
                                    ("player", "opponent"), players
                                ):
                                    record["protocol_retries"][role] = getattr(
                                        participant, "protocol_retries", {}
                                    ).get(latest_tag, [])
                                    record["values"][role] = getattr(
                                        participant, "decision_values", {}
                                    ).pop(latest_tag, [])
                            if record["error"]:
                                cleanup_errors = await _close_players(players)
                                if cleanup_errors:
                                    record["error"] += "; " + "; ".join(cleanup_errors)
                                players = []
                            # Keep finished battles until disconnect: trailing protocol
                            # messages still need their battle objects in this fork.
                        record["duration_s"] = time.perf_counter() - start
                        if manifest is not None:
                            if record["error"]:
                                manifest["missing"].extend(
                                    {
                                        "battle_id": record["battle_id"],
                                        "player_side": side,
                                        "reason": record["error"],
                                    }
                                    for side in ("p1", "p2")
                                )
                            write_manifest()
                        records.append(record)
                        log.write(json.dumps(record) + "\n")
                        log.flush()
            finally:
                cleanup_errors = await _close_players(players)
                if cleanup_errors:
                    raise RuntimeError("; ".join(cleanup_errors))

        await asyncio.gather(
            *(lane() for _ in range(min(config.concurrent, config.battles)))
        )
    summary_config = asdict(config)
    if not config.save_trajectories:
        summary_config.pop("save_trajectories")
    summary = summarize(records, time.perf_counter() - started, summary_config)
    summary["run_id"] = run_id
    from collections import Counter

    import numpy as np

    summary["neural"] = {}
    for role, participants in neural_players.items():
        if not participants:
            continue
        stats = [participant.neural_stats() for participant in participants]
        latency = [ms for item in stats for ms in item["inference_ms"]]
        fallbacks: Counter[str] = Counter()
        lead_sources: Counter[str] = Counter()
        lead_species: Counter[str] = Counter()
        for item in stats:
            fallbacks.update(item["fallbacks"])
            lead_sources.update(item.get("lead_sources", {}))
        for participant in participants:
            for preview in getattr(participant, "team_preview_records", {}).values():
                lead_species.update([preview["chosen_lead"]])
        tokens = sum(item["total_tokens"] for item in stats)
        unknown = sum(item["unknown_tokens"] for item in stats)
        summary["neural"][role] = {
            "decisions": sum(item["model_decisions"] for item in stats),
            "fallbacks": dict(fallbacks),
            "lead_source_counts": dict(lead_sources),
            "lead_species_counts": dict(lead_species),
            "illegal_actions": sum(item["illegal_actions"] for item in stats),
            "protocol_retries": sum(
                item.get("protocol_retries", 0) for item in stats
            ),
            "inference_ms_p50": float(np.percentile(latency, 50)) if latency else None,
            "inference_ms_p95": float(np.percentile(latency, 95)) if latency else None,
            "unknown_tokens": unknown,
            "total_tokens": tokens,
            "unknown_token_rate": unknown / tokens if tokens else 0.0,
            "backend": "metamon",
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def run_battles(config: RunConfig) -> dict[str, Any]:
    """Execute a run on poke-env's event loop with independent bounded battle lanes."""
    from poke_env.concurrency import handle_threaded_coroutines

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_id = timestamp + "-" + uuid.uuid4().hex[:8]
    output = Path(config.out) / run_id
    output.mkdir(parents=True, exist_ok=False)
    return await handle_threaded_coroutines(_run_on_poke_loop(config, run_id, output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", default="GymLeader")
    parser.add_argument("--opponent", default="RandomBaseline")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--teams", default="data/teams")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--team-preview-checkpoint",
        help="Optional team-preview checkpoint shared by neural players.",
    )
    parser.add_argument("--team-preview-argmax", action="store_true")
    parser.add_argument(
        "--schedule-seed", type=int,
        help="Precompute paired team draws by matchup index for reproducible arms.",
    )
    parser.add_argument("--swap-team-pairs", action="store_true")
    parser.add_argument("--alternate-challenger", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run_battles(RunConfig(**vars(args))))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
