"""Offline Showdown match harness: poke-env/Metamon players without a websocket.

The Node sidecar (``sim.py``) hosts ``sim/Battle``; this module feeds the players the
same per-player frames ``server/room-battle.ts`` would (see ``FRAMING_NOTES``), captures
the orders they would have sent to the server, and supports snapshot/restore of the
whole match (simulator + per-tag Python player state) for counterfactual branching.

Hidden information: a player only ever receives its own side's channel
(``|split|`` extraction, per-side ``sideupdate``). The omniscient channel is recorded
for analysis but never delivered.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pokeai.cf.sim import Sim, gen5_seed

SIDES = ("p1", "p2")
PLAYER_NAMES = {"p1": "CFp1", "p2": "CFp2"}
ROOM_PREFIX = "battle-gen9ou-"
# Metamon baselines parse the room as exactly `battle-<format>-<n>` (3 dash parts).
_TAG_COUNTER = [0]


def new_tag() -> str:
    """Fresh server-shaped battle tag (digits only after the format)."""
    _TAG_COUNTER[0] += 1
    return f"{ROOM_PREFIX}{_TAG_COUNTER[0]}"

FRAMING_NOTES = [
    (
        "Init frames (`|init|battle`, `|title|`, `|j|`) copy what a live local server "
        "sent (p1: init+own join, then p2's `|j|`; p2: both joins); real timing relative "
        "to the first sim update chunk is not reproduced, only the order."
    ),
    (
        "Each sim `update` is delivered as one frame per player containing that player's "
        "channel (|split| extraction identical to sim/battle.ts); the real server sends "
        "room.update() broadcast buffers, which may batch several sim updates or `|inactive|` "
        "timer lines; no timer lines are produced offline."
    ),
    (
        "`sideupdate |request|` is delivered as its own frame AFTER the update frame with an "
        "incrementing room-wide `rqid` appended as the last JSON key (room-battle.ts receive)."
    ),
    (
        "Both players receive every frame of a sim output batch before either player's "
        "reply is submitted (p1 then p2); in production the arrival order of the two "
        "replies is racy."
    ),
    (
        "RoomBattlePlayer.request.isWait bookkeeping is mirrored: 'There's nothing to choose' "
        "and 'too late' errors are produced; the `|rqid` suffix check and the battle timer "
        "are not (our players never send an rqid suffix)."
    ),
    (
        "No `|:|` scrollback timestamp, no `|inactive|` lines, no `|uhtml|` open-team-sheet "
        "prompts, no `|deinit|`/`|l|` or lobby `|pm|`/`|updatesearch|` frames. Frame "
        "kinds/order/request key order were checked against a live capture "
        "(reports/w5_cf0_1_frame_parity.json)."
    ),
]

_T_LINE = re.compile(r"^\|t:\|\d+$")


def normalise_line(line: str) -> str:
    """Wall-clock `|t:|` protocol lines are the only content normalisation."""
    return "|t:|" if _T_LINE.match(line) else line


def normalise_frame(raw: str, tag: str) -> str:
    """Normalise the battle id in the frame header and the `|t:|` lines."""
    head, _, body = raw.partition("\n")
    head = head.replace(tag, "<TAG>")
    return head + "\n" + "\n".join(normalise_line(x) for x in body.split("\n"))


_HP_FIELD = re.compile(r"^(\d+)/(\d+)((?: [a-z]+)?)$")
_HP_LINES = ("switch", "drag", "replace", "-damage", "-heal", "-sethp")


def public_pct(hp: int, maxhp: int) -> int:
    if hp == 0:
        return 0
    pct = -(-100 * hp // maxhp)
    return 99 if pct == 100 and hp < maxhp else pct


def hp_for_pct(pct: int, maxhp: int) -> int:
    if pct == 0:
        return 0
    return maxhp if pct >= 100 else max(1, pct * maxhp // 100)


def rewrite_p2_hp(line: str, by_name: dict[str, dict[str, Any]]) -> str:
    """Re-express exact HP fields of an opponent-channel line for a new max HP."""
    parts = line.split("|")
    if len(parts) < 4 or parts[1] not in _HP_LINES:
        return line
    ident = re.match(r"^p2[a-z]?: (.*)$", parts[2])
    if not ident or ident.group(1) not in by_name:
        return line
    info = by_name[ident.group(1)]
    for i in range(3, len(parts)):
        match = _HP_FIELD.match(parts[i])
        if match and int(match.group(2)) == info["old_max"]:
            hp = int(match.group(1))
            new = hp_for_pct(public_pct(hp, info["old_max"]), info["new_max"])
            parts[i] = f"{new}/{info['new_max']}{match.group(3)}"
            break
    return "|".join(parts)


def digest(array: Any) -> str:
    data = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else array
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run(coro: Any) -> Any:
    """Run a coroutine on poke-env's loop (players' asyncio primitives live there)."""
    from poke_env.concurrency import POKE_LOOP

    return asyncio.run_coroutine_threadsafe(coro, POKE_LOOP).result()


# ---------------------------------------------------------------- player construction


@dataclass
class PlayerSpec:
    """Policy description: ``neural:<model_dir>`` or a Metamon baseline name."""

    policy: str
    sample: bool = True
    temperature: float = 1.0
    preview: str | None = None  # team preview checkpoint (argmax) or None


_MODEL_CACHE: dict[str, Any] = {}
_PREVIEW_CACHE: dict[str, Any] = {}


def make_player(spec: PlayerSpec, side: str, device: str = "cpu") -> Any:
    from poke_env import AccountConfiguration

    common = {
        "battle_format": "gen9ou",
        # restored battles add queue slots; never block
        "max_concurrent_battles": 10**6,
        "account_configuration": AccountConfiguration(PLAYER_NAMES[side], None),
        "start_listening": False,
        "log_level": logging.ERROR,
    }
    preview = None
    if spec.preview:
        if spec.preview not in _PREVIEW_CACHE:
            from metamon.backend.team_preview.preview import TeamPreviewModel

            _PREVIEW_CACHE[spec.preview] = TeamPreviewModel.load_from_checkpoint(
                spec.preview, device=device, use_argmax=True
            )
        preview = _PREVIEW_CACHE[spec.preview]
    if spec.policy.startswith("neural:"):
        from pokeai.agents.neural_player import NeuralPlayer
        from pokeai.models.pokedecision import load_pokedecision

        model_dir = spec.policy.removeprefix("neural:")
        if model_dir not in _MODEL_CACHE:
            _MODEL_CACHE[model_dir] = load_pokedecision(model_dir, device)
        model, encoder, _ = _MODEL_CACHE[model_dir]
        player = NeuralPlayer(
            model_dir=model_dir, model=model, encoder=encoder, device=device,
            sample=spec.sample, temperature=spec.temperature,
            team_preview_model=preview, **common,
        )
    else:
        from pokeai.env.runner import _counted_baseline

        player = _counted_baseline(spec.policy)(**common)
        if preview is not None:
            player.team_preview_model = preview
    return player


# ---------------------------------------------------------------- python-side snapshots

_SKIP_ATTRS = {
    "ps_client", "model", "encoder", "team_preview_model", "collector", "_battles",
    "_team", "sampler", "_logger",
}
_SCALARS = (bool, int, float, str, type(None))


def snapshot_player(player: Any, tag: str) -> dict[str, Any]:
    """Deep copy of everything the player holds for `tag` (+ instance scalars)."""
    snap: dict[str, Any] = {
        "battle": copy.deepcopy(player._battles[tag]),
        "tagged": {},
        "scalars": {},
        "containers": {},
    }
    for name, value in vars(player).items():
        if name in _SKIP_ATTRS:
            continue
        if isinstance(value, dict):
            snap["tagged"][name] = (tag in value, copy.deepcopy(value.get(tag)))
        elif isinstance(value, _SCALARS):
            snap["scalars"][name] = value
        elif isinstance(value, (Counter, list)):
            snap["containers"][name] = copy.deepcopy(value)
    return snap


def restore_player(player: Any, snap: dict[str, Any], old_tag: str, new_tag: str) -> None:
    """Re-register a fresh copy of the snapshot under `new_tag` (snapshot untouched)."""
    battle = copy.deepcopy(snap["battle"])
    battle._battle_tag = new_tag
    if hasattr(battle, "_mm_battle"):
        battle._mm_battle.gameid = new_tag
    player._battles[new_tag] = battle
    for name, (present, value) in snap["tagged"].items():
        target = getattr(player, name)
        if present:
            target[new_tag] = copy.deepcopy(value)
        else:
            target.pop(new_tag, None)
    for name, value in snap["scalars"].items():
        setattr(player, name, value)
    for name, value in snap["containers"].items():
        setattr(player, name, copy.deepcopy(value))


def discard_player_tag(player: Any, tag: str) -> None:
    player._battles.pop(tag, None)
    for value in vars(player).values():
        if isinstance(value, dict):
            value.pop(tag, None)


# ---------------------------------------------------------------- match


@dataclass
class ModelDecision:
    tokens_sha: str
    numbers_sha: str
    mask: list[int]
    action: int
    forced: bool
    hidden_sha: str
    value: float | None
    probs: list[float] | None = None
    sampled_action: int | None = None
    features: list[float] | None = None


@dataclass
class RoundRecord:
    index: int
    delivered: dict[str, list[str]]
    omni: list[list[str]]
    decisions: dict[str, list[dict[str, Any]]]
    submitted: list[tuple[str, str]]
    errors: list[str]
    events_out: list[dict[str, Any]]
    state_after: dict[str, Any] | None = None


@dataclass
class Checkpoint:
    """Whole-match snapshot taken BEFORE the pending events are delivered."""

    sim_key: str
    tag: str
    players: dict[str, dict[str, Any]]
    pending: list[dict[str, Any]]
    rqid: int
    wait: dict[str, Any]
    ended: bool
    end_data: dict[str, Any] | None
    round_index: int
    py_rng: dict[str, Any]
    sampler: dict[str, Any]
    turn: int
    info: dict[str, Any] = field(default_factory=dict)


class OfflineMatch:
    """One live battle (sim id == tag suffix) between two offline players."""

    def __init__(
        self, sim: Sim, players: dict[str, Any], tag: str,
        record_probs: bool = False,
    ) -> None:
        self.sim = sim
        self.players = players
        self.tag = tag
        self.pending: list[dict[str, Any]] = []
        self.rqid = 1
        self.wait: dict[str, dict[str, Any]] = {
            s: {"rqid": 0, "is_wait": "cantUndo"} for s in SIDES
        }
        self.ended = False
        self.end_data: dict[str, Any] | None = None
        self.round_index = 0
        self.rounds: list[RoundRecord] = []
        self.state_rounds: set[int] = set()
        self.py_rng: dict[str, Any] = {}
        self.samplers: dict[str, torch.Generator] = {}
        self.forced: dict[str, deque[int | str]] = {s: deque() for s in SIDES}
        # Scripted outgoing orders (restore replay): substituted for the player's own
        # order; every disagreement with what the player computed is recorded.
        self.forced_events: list[tuple[int, str, int | str]] = []
        self.script: dict[str, deque[str]] = {s: deque() for s in SIDES}
        self.script_mismatches: list[tuple[int, str, str, str]] = []
        self.record_probs = record_probs
        self.turn = 0
        self.rejected: list[str] = []
        self._outbox: dict[str, list[tuple[str, str]]] = {s: [] for s in SIDES}
        self._model_log: dict[str, list[ModelDecision]] = {s: [] for s in SIDES}
        for side in SIDES:
            self._instrument(side)

    # -- player instrumentation ------------------------------------------------------
    def _instrument(self, side: str) -> None:
        player = self.players[side]
        match = self

        async def send_message(message: str, room: str = "", message_2: Any = None) -> None:
            if message_2:
                message = f"{message}|{message_2}"
            match._outbox[side].append((message, room))

        player.ps_client.send_message = send_message
        if not hasattr(player, "select_action") or not hasattr(player, "sampler"):
            return
        if getattr(player, "_cf_wrapped", False):
            return
        original = player.select_action

        def select_action(tag: str, encoded: dict, mask: np.ndarray) -> Any:
            owner = player._cf_owner  # the match currently driving this player
            action = original(tag, encoded, mask)
            if action is None:
                return None
            sampled = action
            probs = None
            features = None
            if owner.record_probs:
                before = player._hidden_before.get(tag)
                with torch.inference_mode():
                    tokens = torch.as_tensor(encoded["tokens"]).long()[None, None]
                    numbers = torch.as_tensor(encoded["numbers"]).float()[None, None]
                    # features = the memory output feeding the policy/value heads
                    feats = player.model._forward_features(tokens, numbers, before)[0]
                    out = player.model.policy(feats)
                    features = feats[0, 0].float().tolist()
                    scores = out[0, 0].masked_fill(
                        ~torch.as_tensor(mask), -torch.inf
                    )
                    probs = torch.softmax(
                        scores.float() / player.temperature, dim=-1
                    ).tolist()
            forced = False
            if owner.forced[side]:
                wanted = owner.forced[side].popleft()
                if wanted == "tera":  # first legal move+Tera slot, if any
                    wanted = next((i for i in range(9, 13) if mask[i]), None)
                if wanted is not None:
                    if not mask[wanted]:
                        raise RuntimeError(f"forced action {wanted} outside mask")
                    action, forced = wanted, True
            hidden = player.hidden.get(tag)
            owner._model_log[side].append(ModelDecision(
                tokens_sha=digest(np.asarray(encoded["tokens"])),
                numbers_sha=digest(np.asarray(encoded["numbers"])),
                mask=[int(x) for x in mask],
                action=int(action),
                forced=forced,
                hidden_sha=digest(hidden) if hidden is not None else "none",
                value=(player.decision_values.get(tag) or [None])[-1],
                probs=probs,
                sampled_action=int(sampled),
                features=features,
            ))
            return action

        player.select_action = select_action
        player._cf_wrapped = True

    def _bind(self) -> None:
        for side in SIDES:
            self.players[side]._cf_owner = self
            if hasattr(self.players[side], "sampler"):
                self.players[side].sampler = self.samplers.get(side)

    # -- seeding -----------------------------------------------------------------------
    def seed_side(self, side: str, seed: int) -> None:
        """Independent python-`random` and torch sampling streams for one side."""
        self.py_rng[side] = random.Random(f"cf-py-{seed}-{side}").getstate()
        generator = torch.Generator()
        generator.manual_seed(int(seed) & 0x7FFFFFFF)
        self.samplers[side] = generator

    # -- lifecycle ---------------------------------------------------------------------
    def start(self, sim_seed: int, teams: dict[str, str], side_seeds: dict[str, int]) -> None:
        for side in SIDES:
            self.seed_side(side, side_seeds[side])
        self._bind()
        events = self.sim.create(
            self.tag, gen5_seed(sim_seed),
            {"name": PLAYER_NAMES["p1"], "team": teams["p1"]},
            {"name": PLAYER_NAMES["p2"], "team": teams["p2"]},
        )
        title = f"{PLAYER_NAMES['p1']} vs. {PLAYER_NAMES['p2']}"
        join1, join2 = (f"|j|\u2606{PLAYER_NAMES[s]}" for s in SIDES)
        # Captured live (local server): p1 joins first, so p1 gets its own `|j|` in the
        # init frame (with a trailing newline) and p2's `|j|` as a separate frame; p2's
        # init frame already lists both.
        self.pending = [
            {"t": "init", "p1": [f"|init|battle\n|title|{title}\n{join1}\n", join2],
             "p2": [f"|init|battle\n|title|{title}\n{join1}\n{join2}\n"]},
            *events,
        ]

        async def begin() -> None:
            for side in SIDES:
                player = self.players[side]
                random.setstate(self.py_rng[side])
                player.randomize()
                self.py_rng[side] = random.getstate()
                player.last_server_error = None

        _run(begin())

    def checkpoint(self) -> Checkpoint:
        key = self.sim.snapshot(self.tag)
        return Checkpoint(
            sim_key=key, tag=self.tag,
            players={s: snapshot_player(self.players[s], self.tag) for s in SIDES},
            pending=copy.deepcopy(self.pending), rqid=self.rqid,
            wait=copy.deepcopy(self.wait), ended=self.ended,
            end_data=copy.deepcopy(self.end_data), round_index=self.round_index,
            py_rng=dict(self.py_rng),
            sampler={s: g.get_state().clone() for s, g in self.samplers.items()},
            turn=self.turn,
        )

    @classmethod
    def restore(
        cls, sim: Sim, players: dict[str, Any], cp: Checkpoint, new_tag: str,
        record_probs: bool = False,
    ) -> OfflineMatch:
        match = cls(sim, players, new_tag, record_probs=record_probs)
        sim.restore(cp.sim_key, new_tag)
        for side in SIDES:
            restore_player(players[side], cp.players[side], cp.tag, new_tag)
        match.pending = copy.deepcopy(cp.pending)
        match.rqid = cp.rqid
        match.wait = copy.deepcopy(cp.wait)
        match.ended = cp.ended
        match.end_data = copy.deepcopy(cp.end_data)
        match.round_index = cp.round_index
        match.turn = cp.turn
        match.py_rng = dict(cp.py_rng)
        for side, state in cp.sampler.items():
            generator = torch.Generator()
            generator.set_state(state.clone())
            match.samplers[side] = generator
        match._bind()

        async def register() -> None:
            for side in SIDES:
                players[side]._battle_count_queue.put_nowait(None)

        _run(register())
        return match

    def apply_world(self, world: dict[str, Any]) -> dict[str, Any]:
        """Transplant a sampled opponent world (p2 hidden fields) into this root.

        The opponent's own pending request is regenerated by the sim from the new sets;
        exact HP numbers in the opponent's own-channel lines are re-expressed against the
        new max HP with the same public percentage. The learner's frames are untouched.
        """
        response = self.sim.apply_world(self.tag, world)
        maps = {m["index"]: m for m in response["hp_maps"]}
        roster = {m["index"]: m["name"] for m in self.sim.roster(self.tag)}
        by_name = {roster[i]: maps[i] for i in maps}
        for ev in self.pending:
            if ev["t"] == "sideupdate" and ev["side"] == "p2" and ev["data"].startswith(
                "|request|"
            ):
                ev["data"] = "|request|" + response["request"]
            elif ev["t"] == "update":
                ev["p2"] = [rewrite_p2_hp(x, by_name) for x in ev["p2"]]
        return response

    def close(self) -> None:
        """Release sim + python state of this tag."""
        try:
            self.sim.destroy(self.tag)
        finally:
            for side in SIDES:
                discard_player_tag(self.players[side], self.tag)

    # -- server-identical framing --------------------------------------------------------
    def _plan(self, events: list[dict[str, Any]]) -> list[tuple[str, str]]:
        frames: list[tuple[str, str]] = []
        head = f">{self.tag}\n"
        for ev in events:
            kind = ev["t"]
            if kind == "init":
                for side in SIDES:
                    frames.extend((side, head + text) for text in ev[side])
            elif kind == "update":
                for side in SIDES:
                    lines = ev[side]
                    if lines:
                        frames.append((side, head + "\n".join(lines)))
                    for line in lines:
                        if line.startswith("|turn|"):
                            self.turn = int(line[6:])
            elif kind == "sideupdate":
                side, data = ev["side"], ev["data"]
                if data.startswith("|error|[Invalid choice] Can't do anything"):
                    pass
                elif data.startswith("|error|[Invalid choice]"):
                    self.wait[side]["is_wait"] = (
                        "cantUndo" if "Can't undo" in data else False
                    )
                    self.wait[side]["choice"] = ""
                elif data.startswith("|request|"):
                    self.rqid += 1
                    request = json.loads(data[len("|request|"):])
                    request["rqid"] = self.rqid
                    data = "|request|" + json.dumps(
                        request, separators=(",", ":"), ensure_ascii=False
                    )
                    self.wait[side] = {
                        "rqid": self.rqid,
                        "is_wait": "cantUndo" if request.get("wait") else False,
                    }
                frames.append((side, head + data))
            elif kind == "end":
                self.ended = True
                self.end_data = ev["data"]
        return frames

    async def _deliver(self, side: str, raw: str) -> None:
        player = self.players[side]
        random.setstate(self.py_rng[side])
        try:
            await player.ps_client._handle_message(raw)
        finally:
            self.py_rng[side] = random.getstate()

    def _submit(self, side: str, message: str, errors: list[str]) -> list[dict[str, Any]]:
        if message.startswith("/choose "):
            choice = message[len("/choose "):]
        elif message.startswith("/team "):
            choice = "team " + message[len("/team "):]
        else:
            raise RuntimeError(f"unsupported outgoing message {message!r}")
        wait = self.wait[side]
        other = self.wait["p2" if side == "p1" else "p1"]
        if wait["is_wait"] is not False and wait["is_wait"] is not True:
            return [{
                "t": "sideupdate", "side": side,
                "data": "|error|[Invalid choice] There's nothing to choose",
            }]
        if bool(wait["is_wait"]) and bool(other["is_wait"]):
            return [{
                "t": "sideupdate", "side": side,
                "data": "|error|[Invalid choice] Sorry, too late to make a different "
                "move; the next turn has already started",
            }]
        wait["is_wait"] = True
        wait["choice"] = choice
        response = self.sim.choose(self.tag, side, choice)
        return response["events"]

    def step(self) -> RoundRecord:
        """One delivery round: deliver pending frames, collect + submit replies."""
        events, self.pending = self.pending, []
        frames = self._plan(events)
        delivered: dict[str, list[str]] = {s: [] for s in SIDES}
        before_illegal = {s: getattr(self.players[s], "illegal_actions", 0) for s in SIDES}
        for side in SIDES:
            self._model_log[side].clear()
        omni = [ev["omni"] for ev in events if ev["t"] == "update"]

        async def deliver_all() -> None:
            for side, raw in frames:
                delivered[side].append(raw)
                await self._deliver(side, raw)

        _run(deliver_all())
        errors = [
            line for side in SIDES for raw in delivered[side]
            for line in raw.split("\n") if line.startswith("|error|")
        ]
        submitted: list[tuple[str, str]] = []
        decisions: dict[str, list[dict[str, Any]]] = {s: [] for s in SIDES}
        new_events: list[dict[str, Any]] = []
        for side in SIDES:
            outbox, self._outbox[side] = self._outbox[side], []
            models = list(self._model_log[side])
            for message, room in outbox:
                if room != self.tag:
                    raise RuntimeError(f"{side} wrote to unexpected room {room!r}")
                if self.script[side]:
                    scripted = self.script[side].popleft()
                    if scripted != message:
                        self.script_mismatches.append(
                            (self.round_index, side, message, scripted)
                        )
                    message = scripted
                submitted.append((side, message))
                decisions[side].append({
                    "message": message,
                    "model": models.pop(0).__dict__ if models else None,
                })
                new_events.extend(self._submit(side, message, errors))
            for leftover in models:  # model decision without an outgoing order
                decisions[side].append({"message": None, "model": leftover.__dict__})
        self.pending = new_events
        for side in SIDES:
            if getattr(self.players[side], "illegal_actions", 0) != before_illegal[side]:
                errors.append(f"policy-illegal:{side}")
        record = RoundRecord(
            index=self.round_index, delivered=delivered, omni=omni,
            decisions=decisions, submitted=submitted, errors=errors,
            events_out=copy.deepcopy(new_events),
        )
        if self.round_index in self.state_rounds:
            record.state_after = self.sim.state(self.tag)
        self.round_index += 1
        self.rounds.append(record)
        return record

    @property
    def finished(self) -> bool:
        return self.ended and not any(e["t"] in ("update", "sideupdate") for e in self.pending)

    def play(self, max_rounds: int = 2000, hook: Any = None) -> None:
        while not self.finished:
            if self.round_index >= max_rounds:
                raise RuntimeError("battle exceeded max_rounds")
            if hook is not None:
                hook(self)
            self.step()
        # deliver the trailing end frames (|win|) if still pending
        if self.pending:
            self.step()

    def winner(self) -> str | None:
        if not self.end_data:
            return None
        name = self.end_data.get("winner")
        for side, value in PLAYER_NAMES.items():
            if name == value:
                return side
        return "tie"
