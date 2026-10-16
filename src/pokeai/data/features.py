"""One frozen Metamon encoder for replay preprocessing and online decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from metamon.interface import (
    DefaultObservationSpace,
    TokenizedObservationSpace,
    UniversalState,
)
from metamon.tokenizer import UNKNOWN_TOKEN, get_tokenizer

from pokeai.data.dataset import SPACE_VERSIONS


class ObservationEncoder:
    """Preserve premade IDs; map native -1 to one appended embedding-safe UNK.

    No vocabulary is learned from validation, test, or online states. Numbers stay
    raw: their train-only normalization belongs to the model's registered buffers.
    """

    def __init__(self, tokenizer_name: str = "DefaultObservationSpace-v1") -> None:
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.tokenizer.freeze()
        self.unknown_id = len(self.tokenizer)
        self.vocab_size = self.unknown_id + 1
        self.token_len = DefaultObservationSpace().tokenizable["text"]
        self.space = TokenizedObservationSpace(DefaultObservationSpace(), self.tokenizer)

    def encode(self, state: UniversalState) -> dict[str, np.ndarray]:
        obs = self.space.state_to_obs(state)
        tokens = np.asarray(obs["text_tokens"], dtype=np.int64)
        tokens = np.where(tokens == UNKNOWN_TOKEN, self.unknown_id, tokens)
        numbers = np.asarray(obs["numbers"], dtype=np.float32)
        if tokens.shape != (self.token_len,) or numbers.shape != (48,):
            raise ValueError(
                f"Unexpected observation shapes: {tokens.shape}, {numbers.shape}"
            )
        return {"tokens": tokens, "numbers": numbers}

    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "token_len": self.token_len,
            "unknown_id": self.unknown_id,
            "native_unknown_id": UNKNOWN_TOKEN,
            "unknown_handling": "native -1 maps to appended unknown_id; known IDs unchanged",
            "observation_space": "DefaultObservationSpace",
            "observation_space_version": SPACE_VERSIONS["observation_space_version"],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.metadata(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ObservationEncoder:
        metadata = json.loads(Path(path).read_text())
        encoder = cls(metadata["tokenizer_name"])
        if metadata != encoder.metadata():
            raise ValueError(
                "Encoder metadata does not match installed Metamon vocabulary"
            )
        return encoder
