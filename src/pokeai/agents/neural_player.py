"""Masked behavior-cloning policy using Metamon's online replay parser."""

from __future__ import annotations

import hashlib
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.config import format_for_agent
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalPokemon, UniversalState
from poke_env.data import GenData, to_id_str
from poke_env.player.battle_order import BattleOrder


class PartySlotOrder(BattleOrder):
    """Explicit party index for a request the vendor cannot map to Pokemon."""

    def __init__(self, party_index: int) -> None:
        super().__init__(None)
        self.party_index = party_index

    @property
    def message(self) -> str:
        return f"/choose switch {self.party_index}"


def switch_by_party_slot(battle: Any, order: Any) -> Any:
    """Address switches by request slot, not mutable battle-form species names."""
    if not order.message.startswith("/choose switch "):
        return order
    request = getattr(battle, "last_request", None) or battle._last_request
    selected = order.order

    dex = GenData.from_gen(9).pokedex

    def identity(species: str) -> str:
        key = to_id_str(species)
        return to_id_str(dex[key]["baseSpecies"])

    species = identity(selected.species)
    slots = [
        index for index, pokemon in enumerate(request["side"]["pokemon"], 1)
        if identity(pokemon["details"].split(",")[0]) == species
    ]
    if len(slots) != 1:
        raise RuntimeError(f"Switch species {species} does not identify one request slot")
    return PartySlotOrder(slots[0])


class NeuralPlayer(MetamonPlayer):
    """One recurrent state per battle; no hidden information or unmasked choices."""

    # Optional torch.Generator for sampling (counterfactual branches seed it per
    # rollout); None keeps the process-global generator used by live play.
    sampler: torch.Generator | None = None
    # When True, `choose_move` stores the public ActionDescriptor of the legal slots
    # (pokeai.cf.descriptors; built from the battle object only) in `last_descriptor`.
    describe: bool = False
    last_descriptor: Any = None
    # BS1.1: optional live belief-search layer (pokeai.agents.live_search.LiveSearch). None is a
    # strict no-op: no extra state, snapshots, softmax or timestamps are taken.
    search: Any = None
    forward_count: int = 0

    def __init__(
        self,
        *args: Any,
        model_dir: str | Path | None = None,
        model: Any = None,
        encoder: Any = None,
        device: str = "cpu",
        sample: bool = False,
        temperature: float = 1.0,
        team_preview_model: Any = None,
        search: Any = None,
        **kwargs: Any,
    ) -> None:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if model is None:
            from pokeai.models.pokedecision import load_pokedecision

            model, encoder, _ = load_pokedecision(model_dir, device)
        if encoder is None:
            raise ValueError("encoder is required with an injected model")
        self.model = model.eval()
        self.encoder = encoder
        self.device = next(model.parameters()).device
        self.sample = sample
        self.temperature = temperature
        self.hidden: dict[str, Any] = {}
        self.model_dir = str(model_dir) if model_dir is not None else None
        self.policy_version = Path(model_dir).name if model_dir is not None else "injected"
        self.checkpoint_sha256 = (
            hashlib.sha256((Path(model_dir) / "best.pt").read_bytes()).hexdigest()
            if model_dir is not None else None
        )
        self._behavior: dict[str, dict[str, Any]] = {}
        # P(win) for represented decisions only; fallbacks/preview have no value.
        self.decision_values: dict[str, list[float]] = {}
        self._model_choice: dict[str, bool] = {}
        self._hidden_before: dict[str, Any] = {}
        self._pending_choice: dict[str, dict[str, Any]] = {}
        self._public_log: dict[str, deque[str]] = {}
        # BS0: the learner's complete public protocol history per battle (never truncated)
        self.history: dict[str, list[str]] = {}
        self.protocol_retries: dict[str, list[dict[str, Any]]] = {}
        self.decisions = 0
        self.last_server_error: str | None = None
        self.collector: Any = None
        self.fallbacks: Counter[str] = Counter()
        self.team_preview_records: dict[str, dict[str, Any]] = {}
        self.lead_sources: Counter[str] = Counter()
        self.illegal_actions = 0
        self.latencies_ms: list[float] = []
        self.unknown_tokens = 0
        self.total_tokens = 0
        self.search = search
        if search is not None:
            self._frame_snap: dict[str, Any] = {}  # pre-frame python snapshot per battle
            self._frame_start: dict[str, int] = {}  # history length at that snapshot
            self._frame_decided: dict[str, bool] = {}
            self._t_request: dict[str, float] = {}
            self.search_rows: dict[str, list[dict[str, Any]]] = {}
            self._search_open: dict[str, dict[str, Any]] = {}
            self._last_probs: list[float] | None = None
        super().__init__(*args, team_preview_model=team_preview_model, **kwargs)

    def randomize(self) -> None:
        """Greedy evaluation deliberately has no policy randomization."""

    def _battle_finished_callback(self, battle: Any) -> None:
        self.hidden.pop(battle.battle_tag, None)
        self._model_choice.pop(battle.battle_tag, None)
        self._hidden_before.pop(battle.battle_tag, None)
        self._pending_choice.pop(battle.battle_tag, None)
        self._public_log.pop(battle.battle_tag, None)
        self.history.pop(battle.battle_tag, None)
        if self.search is not None:
            for table in (self._frame_snap, self._frame_start, self._frame_decided,
                          self._t_request, self._search_open):
                table.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        from pokeai.env.runner import expected_retry_reason

        tag = split_messages[0][0].removeprefix(">")
        if self.search is not None:
            self._search_frame_hook(tag, split_messages)
        public_log = self._public_log.setdefault(tag, deque(maxlen=40))
        history = self.history.setdefault(tag, [])
        for message in split_messages[1:]:
            if len(message) > 1 and message[1] != "request":
                public_log.append("|".join(message))
                history.append("|".join(message))
                if self.collector is not None:
                    self.collector.events.append(message)
            if len(message) > 2 and message[1] == "error":
                error = "|".join(message)
                previous = self._pending_choice.get(tag)
                reason = expected_retry_reason(message[2], previous)
                if reason is not None:
                    self.protocol_retries.setdefault(tag, []).append({
                        **previous, "error": error, "public_log": list(public_log),
                        "reason": reason,
                    })
                    # Showdown sends a corrected request; the vendor owns retry
                    # scheduling. Restore memory so that request is not a new turn.
                    if self._model_choice.get(tag, False):
                        hidden = self._hidden_before.pop(tag, None)
                        if hidden is None:
                            self.hidden.pop(tag, None)
                        else:
                            self.hidden[tag] = hidden
                        values = self.decision_values.get(tag)
                        if self.collector is not None and self.collector.behavior:
                            self.collector.behavior[-1]["protocol_retry"] = True
                        if values:
                            values.pop()
                    self._pending_choice.pop(tag, None)
                else:
                    self.last_server_error = error
                    if self._model_choice.get(tag, False):
                        self.illegal_actions += 1
        await super()._handle_battle_message(split_messages)

    def _search_frame_hook(self, tag: str, split_messages: list[list[str]]) -> None:
        """BS1.1 bookkeeping, run before the vendor processes a message batch: the request
        arrival clock, and the python snapshot of the state BEFORE the lines of the frame that
        precedes the next decision (what the offline reconstruction expects as its root)."""
        lines = [m for m in split_messages[1:] if len(m) > 1]
        if any(m[1] == "request" for m in lines):
            self._t_request[tag] = time.time()
        fresh = [m for m in lines if m[1] not in ("request", "error")]
        new_frame = tag not in self._frame_snap or self._frame_decided.get(tag, True)
        if fresh and new_frame and tag in self._battles and not getattr(
            self._battles[tag], "finished", False
        ):
            from pokeai.cf.offline import snapshot_player

            self._frame_snap[tag] = snapshot_player(self, tag)
            self._frame_start[tag] = len(self.history.get(tag, []))
            self._frame_decided[tag] = False

    async def _handle_battle_request(
        self, battle: Any, maybe_default_order: bool = False
    ) -> None:
        # Native default orders bypass choose_move. Never attribute their errors
        # to the preceding policy order or reuse its hidden-trap evidence.
        self._pending_choice.pop(battle.battle_tag, None)
        self._model_choice[battle.battle_tag] = False
        if self.search is not None:
            self._t_request.setdefault(battle.battle_tag, time.time())
        await super()._handle_battle_request(battle, maybe_default_order)
        if self.search is not None:
            row = self._search_open.pop(battle.battle_tag, None)
            if row is not None:
                row["t_send"] = time.time()
            self._t_request.pop(battle.battle_tag, None)

    def teampreview(self, battle: Any) -> str:
        """Record source, actual lead, order, and maximum predicted lead probability.

        Match the vendor adapter's inputs and ordering without its per-battle
        warning. Random previews retain the team_preview fallback count; model
        fallbacks additionally retain the exact reason in lead_sources and the
        per-battle record. Probability is None when no prediction was made.
        """
        self.hidden.pop(battle.battle_tag, None)
        self._model_choice[battle.battle_tag] = False
        top_probability = None
        preview_probabilities = None
        if self.team_preview_model is None:
            source = "random_default"
            # Preserve the vendor's default random order and RNG consumption.
            order = super().teampreview(battle)
        else:
            agent_format = format_for_agent(self._format.replace("-", "").lower())
            team = list(battle.team.values())
            opponent = list(battle.opponent_team.values())
            reason = None
            if agent_format not in self.team_preview_model.trained_formats:
                reason = "format_not_trained"
            elif len(team) != 6 or len(opponent) != 6:
                reason = "team_size_not_six"
            else:
                predicted_lead, probabilities, _ = self.team_preview_model.predict_lead(
                    our_team=[pokemon_name(p.species) for p in team],
                    our_team_moves=[
                        [move_name(m.id) for m in p.moves.values()] if p.moves else []
                        for p in team
                    ],
                    our_team_abilities=[
                        UniversalPokemon.universal_abilities(p.ability) for p in team
                    ],
                    our_team_items=[
                        UniversalPokemon.universal_items(p.item) for p in team
                    ],
                    opponent_team=[pokemon_name(p.species) for p in opponent],
                    battle_format=agent_format,
                )
                top_probability = float(probabilities.max().item())
                preview_probabilities = probabilities.detach().float().cpu().tolist()
                lead_position = next(
                    (i for i, p in enumerate(team, 1)
                     if pokemon_name(p.species) == predicted_lead),
                    None,
                )
                if lead_position is None:
                    reason = "predicted_lead_not_found"
                else:
                    members = [lead_position] + [
                        i for i in range(1, len(team) + 1) if i != lead_position
                    ]
                    order = "/team " + "".join(str(i) for i in members)
            if reason is not None:
                source = f"model_fallback:{reason}"
                order = self.random_teampreview(battle)
            else:
                source = "model"

        if source != "model":
            self.fallbacks["team_preview"] += 1
        self.lead_sources[source] += 1
        if self.search is not None:
            self._frame_decided[battle.battle_tag] = True
        lead_index = int(order.removeprefix("/team ")[0]) - 1
        self.team_preview_records[battle.battle_tag] = {
            "source": source,
            "chosen_lead": list(battle.team.values())[lead_index].species,
            "top_probability": top_probability,
            "order": order,
            "probabilities": preview_probabilities,
            "chosen_index": -1,
            "fallback_reason": "team_preview",
        }
        return order

    def _fallback(self, battle: Any, reason: str) -> Any:
        self.fallbacks[reason] += 1
        self._model_choice[battle.battle_tag] = False
        self._behavior[battle.battle_tag] = {
            "probabilities": None, "chosen_index": -1, "log_probability": None,
            "mode": "sample" if self.sample else "argmax",
            "temperature": self.temperature, "mask": [0] * 13,
            "policy_version": self.policy_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "value": None, "fallback_reason": reason,
        }
        if battle.reviving:
            # Metamon's online request parser skips revival side updates. Read the
            # request's explicit party slots, not its empty available_switches.
            side = battle._last_request["side"]["pokemon"]
            for index, pokemon in enumerate(side, 1):
                if pokemon["condition"].startswith("0 fnt"):
                    return PartySlotOrder(index)
            raise RuntimeError("Revival request has no fainted target")
        return self.choose_random_move(battle)

    def select_action(self, tag: str, encoded: dict, mask: np.ndarray) -> int | None:
        """Advance memory once for a represented decision, then select in-mask."""
        if not mask.any():
            return None
        tokens = torch.as_tensor(encoded["tokens"], device=self.device).long()[None, None]
        numbers = torch.as_tensor(encoded["numbers"], device=self.device).float()[
            None, None
        ]
        start = time.perf_counter()
        with torch.inference_mode():
            self._hidden_before[tag] = self.hidden.get(tag)
            if self.model.config.value_head:
                logits, value_logits, hidden = self.model.forward_with_value(
                    tokens, numbers, self.hidden.get(tag)
                )
                self.decision_values.setdefault(tag, []).append(
                    float(value_logits[0, 0].sigmoid().item())
                )
            else:
                logits, hidden = self.model(tokens, numbers, self.hidden.get(tag))
            scores = logits[0, 0].masked_fill(
                ~torch.as_tensor(mask, device=self.device), -torch.inf
            )
            self.forward_count += 1
            if self.sample or self.collector is not None or self.search is not None:
                probabilities = torch.softmax(scores.float() / self.temperature, dim=-1)
            if self.sample:
                if self.sampler is None:
                    draw = torch.multinomial(probabilities, 1)
                else:
                    draw = torch.multinomial(probabilities, 1, generator=self.sampler)
                action = int(draw.item())
            else:
                action = int(scores.argmax().item())
            if self.collector is not None:
                self._behavior[tag] = {
                    "probabilities": probabilities.cpu().tolist(),
                    "chosen_index": action,
                    "log_probability": float(probabilities[action].log().item()),
                    "mode": "sample" if self.sample else "argmax",
                    "temperature": self.temperature, "mask": mask.astype(int).tolist(),
                    "policy_version": self.policy_version,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "value": (
                        self.decision_values[tag][-1]
                        if self.model.config.value_head else None
                    ),
                    "fallback_reason": None,
                }
            if self.search is not None:
                self._last_probs = probabilities.cpu().tolist()
            self.hidden[tag] = hidden
        self.latencies_ms.append((time.perf_counter() - start) * 1000)
        if not mask[action]:
            raise RuntimeError("Policy selected an action outside the legality mask")
        self.total_tokens += int(np.size(encoded["tokens"]))
        unknown_id = getattr(self.encoder, "unknown_id", None)
        if unknown_id is not None:
            self.unknown_tokens += int(np.count_nonzero(encoded["tokens"] == unknown_id))
        return action

    def choose_move(self, battle: Any) -> Any:
        from pokeai.env.runner import _protocol_choice_evidence

        self.decisions += 1
        self.last_descriptor = None
        if self.search is not None:
            tag = battle.battle_tag
            t_choose = time.time()
            forwards_before = self.forward_count
            search_row: dict[str, Any] | None = None
        if battle.reviving:
            order = self._fallback(battle, "revival_target")
        elif {move.id for move in battle.available_moves} in (
            {"struggle"}, {"recharge"}, {"fight"}
        ):
            reason = battle.available_moves[0].id
            order = self._fallback(battle, reason)
        else:
            state = UniversalState.from_Battle(battle)
            legal = UniversalAction.definitely_valid_actions(state, battle)
            mask = np.array([UniversalAction(i) in legal for i in range(13)])
            if battle.trapped and not battle.force_switch:
                mask[4:9] = False
            if self.describe:
                from pokeai.cf.descriptors import describe_battle

                self.last_descriptor = describe_battle(battle, mask)
            action = self.select_action(
                battle.battle_tag, self.encoder.encode(state), mask
            )
            if action is None:
                order = self._fallback(battle, "all_masked")
            else:
                if self.search is not None:
                    action, search_row = self._search_decide(battle, action, mask)
                order = UniversalAction(action).to_BattleOrder(battle)
                if order is None:
                    raise RuntimeError("Masked-legal action failed conversion")
                if 4 <= action <= 8:
                    order = switch_by_party_slot(battle, order)
                self._model_choice[battle.battle_tag] = True
        self._pending_choice[battle.battle_tag] = _protocol_choice_evidence(battle, order)
        if self.search is not None:
            if search_row is None:
                search_row = {"category": "not_represented", "action": None, "c_action": None}
            search_row.update(
                t_request=self._t_request.get(tag, t_choose), t_choose=t_choose,
                t_choice_returned=time.time(),
                c_forwards=self.forward_count - forwards_before,
                model_choice=bool(self._model_choice.get(tag, False)),
                hidden_keys=len(self.hidden), turn=battle.turn)
            rows = self.search_rows.setdefault(tag, [])
            if self.collector is not None:
                # W6.0: the native trajectory action must be what was sent. C's own choice
                # stays in chosen_index/probabilities (replay parity); the search row is
                # joined by index, so every recorded step has one.
                self._behavior[tag]["executed_index"] = (
                    action if self._model_choice.get(tag) else -1
                )
                self._behavior[tag]["search_row"] = len(rows)
            rows.append(search_row)
            self._search_open[tag] = search_row
            self._frame_decided[tag] = True
        if self.collector is not None:
            self.collector.record(battle, order, self._behavior.pop(battle.battle_tag))
        return order

    def _search_decide(
        self, battle: Any, action: int, mask: np.ndarray
    ) -> tuple[int, dict[str, Any]]:
        """C's decision is made (one forward, memory advanced). Let the search layer propose a
        different legal action; it never touches `self.hidden`."""
        tag = battle.battle_tag
        t_c_done = time.time()
        if int(np.count_nonzero(mask)) < 2:
            return action, {"category": "not_searched_single_legal", "action": action,
                            "c_action": action, "t_c_done": t_c_done}
        final, row = self.search.run(
            self, battle, tag, action, mask, self._last_probs,
            self._t_request.get(tag, t_c_done), self._frame_snap.get(tag),
            self._frame_start.get(tag, 0), len(self.search_rows.get(tag, [])))
        row["t_c_done"] = t_c_done
        return final, row

    def neural_stats(self) -> dict[str, Any]:
        return {
            "model_decisions": len(self.latencies_ms),
            "fallbacks": dict(self.fallbacks),
            "lead_sources": dict(self.lead_sources),
            "protocol_retries": sum(map(len, self.protocol_retries.values())),
            "protocol_retry_reasons": dict(Counter(
                r["reason"] for v in self.protocol_retries.values() for r in v
            )),
            "illegal_actions": self.illegal_actions,
            "inference_ms": self.latencies_ms.copy(),
            "unknown_tokens": self.unknown_tokens,
            "total_tokens": self.total_tokens,
            "backend": "metamon",
        }
