"""Masked behavior-cloning policy using Metamon's online replay parser."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalState
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
        self._model_choice: dict[str, bool] = {}
        self.decisions = 0
        self.last_server_error: str | None = None
        self.collector: Any = None
        self.fallbacks: Counter[str] = Counter()
        self.illegal_actions = 0
        self.latencies_ms: list[float] = []
        self.unknown_tokens = 0
        self.total_tokens = 0
        super().__init__(*args, **kwargs)

    def randomize(self) -> None:
        """Greedy evaluation deliberately has no policy randomization."""

    def _battle_finished_callback(self, battle: Any) -> None:
        self.hidden.pop(battle.battle_tag, None)
        self._model_choice.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        tag = split_messages[0][0].removeprefix(">")
        for message in split_messages[1:]:
            if len(message) > 1 and message[1] == "error":
                self.last_server_error = "|".join(message)
                if self._model_choice.get(tag, False):
                    self.illegal_actions += 1
        await super()._handle_battle_message(split_messages)

    def teampreview(self, battle: Any) -> str:
        self.hidden.pop(battle.battle_tag, None)
        self._model_choice[battle.battle_tag] = False
        self.fallbacks["team_preview"] += 1
        return super().teampreview(battle)

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
        if self.collector is not None:
            self.collector.record(battle, order)
        return order

    def neural_stats(self) -> dict[str, Any]:
        return {
            "model_decisions": len(self.latencies_ms),
            "fallbacks": dict(self.fallbacks),
            "illegal_actions": self.illegal_actions,
            "inference_ms": self.latencies_ms.copy(),
            "unknown_tokens": self.unknown_tokens,
            "total_tokens": self.total_tokens,
            "backend": "metamon",
        }
