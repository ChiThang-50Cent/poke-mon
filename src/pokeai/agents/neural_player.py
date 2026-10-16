"""Masked behavior-cloning policy using Metamon's online replay parser."""

from __future__ import annotations

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
from poke_env.player.battle_order import BattleOrder


class RevivalOrder(BattleOrder):
    """Explicit party index for a request the vendor cannot map to Pokemon."""

    def __init__(self, party_index: int) -> None:
        super().__init__(None)
        self.party_index = party_index

    @property
    def message(self) -> str:
        return f"/choose switch {self.party_index}"


class NeuralPlayer(MetamonPlayer):
    """One recurrent state per battle; no hidden information or unmasked choices."""

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
        # P(win) for represented decisions only; fallbacks/preview have no value.
        self.decision_values: dict[str, list[float]] = {}
        self._model_choice: dict[str, bool] = {}
        self._hidden_before: dict[str, Any] = {}
        self._pending_choice: dict[str, dict[str, Any]] = {}
        self._public_log: dict[str, deque[str]] = {}
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
        super().__init__(*args, team_preview_model=team_preview_model, **kwargs)

    def randomize(self) -> None:
        """Greedy evaluation deliberately has no policy randomization."""

    def _battle_finished_callback(self, battle: Any) -> None:
        self.hidden.pop(battle.battle_tag, None)
        self._model_choice.pop(battle.battle_tag, None)
        self._hidden_before.pop(battle.battle_tag, None)
        self._pending_choice.pop(battle.battle_tag, None)
        self._public_log.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        from pokeai.env.runner import _is_expected_trapped_retry

        tag = split_messages[0][0].removeprefix(">")
        public_log = self._public_log.setdefault(tag, deque(maxlen=40))
        for message in split_messages[1:]:
            if len(message) > 1 and message[1] != "request":
                public_log.append("|".join(message))
            if len(message) > 2 and message[1] == "error":
                error = "|".join(message)
                previous = self._pending_choice.get(tag)
                if _is_expected_trapped_retry(message[2], previous):
                    self.protocol_retries.setdefault(tag, []).append({
                        **previous, "error": error, "public_log": list(public_log),
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
                        if values:
                            values.pop()
                    self._pending_choice.pop(tag, None)
                else:
                    self.last_server_error = error
                    if self._model_choice.get(tag, False):
                        self.illegal_actions += 1
        await super()._handle_battle_message(split_messages)

    async def _handle_battle_request(
        self, battle: Any, maybe_default_order: bool = False
    ) -> None:
        # Native default orders bypass choose_move. Never attribute their errors
        # to the preceding policy order or reuse its hidden-trap evidence.
        self._pending_choice.pop(battle.battle_tag, None)
        self._model_choice[battle.battle_tag] = False
        await super()._handle_battle_request(battle, maybe_default_order)

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
        lead_index = int(order.removeprefix("/team ")[0]) - 1
        self.team_preview_records[battle.battle_tag] = {
            "source": source,
            "chosen_lead": list(battle.team.values())[lead_index].species,
            "top_probability": top_probability,
            "order": order,
        }
        return order

    def _fallback(self, battle: Any, reason: str) -> Any:
        self.fallbacks[reason] += 1
        self._model_choice[battle.battle_tag] = False
        if battle.reviving:
            # Metamon's online request parser skips revival side updates. Read the
            # request's explicit party slots, not its empty available_switches.
            side = battle._last_request["side"]["pokemon"]
            for index, pokemon in enumerate(side, 1):
                if pokemon["condition"].startswith("0 fnt"):
                    return RevivalOrder(index)
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
            if self.sample:
                action = int(torch.multinomial(
                    torch.softmax(scores / self.temperature, dim=-1), 1
                ).item())
            else:
                action = int(scores.argmax().item())
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
            action = self.select_action(
                battle.battle_tag, self.encoder.encode(state), mask
            )
            if action is None:
                order = self._fallback(battle, "all_masked")
            else:
                order = UniversalAction(action).to_BattleOrder(battle)
                if order is None:
                    raise RuntimeError("Masked-legal action failed conversion")
                self._model_choice[battle.battle_tag] = True
        self._pending_choice[battle.battle_tag] = _protocol_choice_evidence(battle, order)
        if self.collector is not None:
            self.collector.record(battle, order)
        return order

    def neural_stats(self) -> dict[str, Any]:
        return {
            "model_decisions": len(self.latencies_ms),
            "fallbacks": dict(self.fallbacks),
            "lead_sources": dict(self.lead_sources),
            "protocol_retries": sum(map(len, self.protocol_retries.values())),
            "illegal_actions": self.illegal_actions,
            "inference_ms": self.latencies_ms.copy(),
            "unknown_tokens": self.unknown_tokens,
            "total_tokens": self.total_tokens,
            "backend": "metamon",
        }
