"""Native Metamon trajectories from the existing poke-env player-view backend.

Keep W1's poke-env Battle backend: changing parsers would change baseline behavior.
Only UniversalState.from_Battle(actor_battle) is used; the opposing client's private
team is never accessed. Sidecars live outside the native dataset root because
Metamon also indexes ordinary .json files (including same-stem metadata).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lz4.frame
from metamon.interface import UniversalAction, UniversalState


def git_commit(path: str) -> str:
    """Record the actual checkout, failing rather than inventing provenance."""
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def order_to_action(battle: Any, order: Any) -> tuple[int, str | None]:
    """Invert Metamon's canonical sorted moves/switches, including slots 9–12.

    Special requests are explicitly missing, never aliased to a regular move.
    Comparing complete wire orders also preserves Tera and rejects other mechanics.
    """
    if battle.reviving:
        return -1, "revival_target"
    move_ids = {move.id for move in battle.available_moves}
    selected = getattr(order, "order", None)
    if move_ids == {"struggle"} and getattr(selected, "id", None) == "struggle":
        return -1, "struggle"
    if move_ids == {"recharge"}:
        return -1, "recharge"
    candidates = []
    for index in range(13):
        # Metamon helpfully drops unavailable Tera; do not duplicate base labels.
        if index >= 9 and battle.can_tera is None:
            continue
        candidate = UniversalAction.action_idx_to_BattleOrder(battle, index)
        if candidate is not None and candidate.message == order.message:
            candidates.append(index)
    if len(candidates) == 1:
        return candidates[0], None
    return -1, "ambiguous_order" if candidates else "unsupported_order"


@dataclass
class TrajectoryCollector:
    states: list[dict[str, Any]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    masks: list[list[int]] = field(default_factory=list)
    reasons: list[str | None] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    events: list[list[str]] = field(default_factory=list)
    event_counts: list[int] = field(default_factory=list)
    team_preview: dict[str, Any] | None = None
    error: str | None = None

    def record(self, battle: Any, order: Any) -> None:
        state = UniversalState.from_Battle(battle)
        legal = UniversalAction.definitely_valid_actions(state, battle)
        mask = [int(UniversalAction(index) in legal) for index in range(13)]
        action, reason = order_to_action(battle, order)
        if action >= 0 and not mask[action]:
            raise ValueError(f"chosen action {action} outside mask: {mask}")
        self.states.append(state.to_dict())
        self.actions.append(action)
        self.masks.append(mask)
        self.reasons.append(reason)
        self.turns.append(battle.turn)
        self.event_counts.append(len(self.events))

    def payloads(
        self,
        battle: Any,
        run_id: str,
        policy: str,
        team_file: str,
        commit: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.error:
            raise ValueError(self.error)
        if not battle.finished or not self.states:
            raise ValueError("cannot save unfinished or empty trajectory")
        terminal = UniversalState.from_Battle(battle).to_dict()
        result = "win" if battle.won else "loss" if battle.lost else "tie"
        native = {
            "states": [*self.states, terminal],
            "actions": [*self.actions, -1],
        }
        metadata = {
            "schema_version": "1.0",
            "battle_id": battle.battle_tag,
            "run_id": run_id,
            "format": "gen9ou",
            "player_side": battle.player_role,
            "policy_version": f"{policy}@metamon-{commit[:7]}",
            "observation_space_version": f"UniversalState@metamon-{commit[:7]}",
            "battle_backend": "poke-env",
            "action_space": "DefaultActionSpace",
            "reward_function": "DefaultShapedReward",
            "final_result": result,
            "n_steps": len(self.actions),
            "action_masks": self.masks,
            "chosen_actions": self.actions,
            "unsupported_reasons": self.reasons,
            "team_file": team_file,
            "team_preview": self.team_preview,
            "decision_turns": self.turns,
            "public_events": self.events,
            "decision_event_counts": self.event_counts,
        }
        return native, metadata


def save_battle(
    output: Path,
    run_id: str,
    players: list[Any],
    record: dict[str, Any],
    commit: str,
) -> list[dict[str, str]]:
    """Prepare both POVs before publishing files; manifest is the commit point."""
    prepared = []
    timestamp = datetime.now(UTC).strftime("%m-%d-%Y-%H:%M:%S")
    for participant, policy_key, team_key in zip(
        players, ("player", "opponent"), ("player_team", "opponent_team"), strict=True
    ):
        if participant.last_server_error:
            raise ValueError(f"server rejected an order: {participant.last_server_error}")
        battle = participant.battles[record["battle_id"]]
        native, meta = participant.collector.payloads(
            battle, run_id, record[policy_key], record[team_key], commit
        )
        expected = (
            "tie" if record["winner"] == "tie"
            else "win" if record["winner"] == policy_key else "loss"
        )
        if meta["final_result"] != expected:
            raise ValueError("terminal POV result differs from battles.jsonl winner")
        result = meta["final_result"].upper()
        stem = (
            f"pokeai-gen9ou-{battle.battle_tag.rsplit('-', 1)[-1]}"
            f"_Unrated_{participant.username}_vs_{battle.opponent_username}"
            f"_{timestamp}_{result}"
        )
        file = Path("trajectories/gen9ou") / (stem + ".json.lz4")
        sidecar = Path("trajectory_meta/gen9ou") / (stem + ".meta.json")
        prepared.append((file, sidecar, native, meta))
    entries = []
    published: list[Path] = []
    try:
        for file, sidecar, native, meta in prepared:
            atomic_write(output / sidecar, json.dumps(meta).encode())
            published.append(output / sidecar)
            atomic_write(output / file, lz4.frame.compress(json.dumps(native).encode()))
            published.append(output / file)
            entries.append({
                "file": str(file), "meta": str(sidecar),
                "battle_id": record["battle_id"], "player_side": meta["player_side"],
            })
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    return entries
