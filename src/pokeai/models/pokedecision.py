"""Shared replay/live policy with an optional win-value head.

Policy-only configs and checkpoints omit value parameters. Value-enabled models
share the entire encoder/memory and expose raw value logits via forward_with_value;
ordinary forward retains the policy-only return contract.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn


@dataclass
class PokeDecisionConfig:
    arch: str = "gru"
    vocab_size: int = 4096
    token_len: int = 1
    d_model: int = 96
    hidden_size: int = 384
    numeric_size: int = 128
    value_head: bool = False
    # W6-A: opponent-action classes (+1 Tera logit); 0 keeps C's exact parameter set.
    opponent_classes: int = 0


class PokeDecision(nn.Module):
    def __init__(
        self, config: PokeDecisionConfig, feature_mean: Any, feature_std: Any
    ) -> None:
        super().__init__()
        if config.arch not in {"gru", "mlp"}:
            raise ValueError(f"Unknown architecture: {config.arch}")
        self.config = config
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float())
        self.register_buffer(
            "feature_std", torch.as_tensor(feature_std).float().clamp_min(1e-6)
        )
        self.tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.fields = nn.Embedding(config.token_len, config.d_model)
        self.token_projection = nn.Linear(config.d_model, config.d_model)
        self.numeric = nn.Sequential(
            nn.Linear(48, config.numeric_size), nn.GELU(),
            nn.Linear(config.numeric_size, config.numeric_size), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.d_model + config.numeric_size, config.hidden_size),
            nn.GELU(),
        )
        if config.arch == "gru":
            self.memory = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        else:
            self.memory = nn.Sequential(
                nn.Linear(config.hidden_size, 2 * config.hidden_size), nn.GELU(),
                nn.Linear(2 * config.hidden_size, config.hidden_size), nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            )
        self.policy = nn.Linear(config.hidden_size, 13)
        # Construct last so enabling value prediction preserves seeded policy weights.
        self.value_head = (
            nn.Linear(config.hidden_size, 1) if config.value_head else None
        )
        # Constructed after value_head so enabling it preserves every C parameter.
        self.opponent_head = (
            nn.Linear(config.hidden_size, config.opponent_classes + 1)
            if config.opponent_classes else None
        )

    def _forward_features(
        self, tokens: Tensor, numbers: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        positions = torch.arange(self.config.token_len, device=tokens.device)
        embedded = self.tokens(tokens) + self.fields(positions)
        embedded = torch.nn.functional.gelu(self.token_projection(embedded))
        categorical = torch.cat((embedded.mean(-2), embedded.amax(-2)), dim=-1)
        numerical = self.numeric((numbers - self.feature_mean) / self.feature_std)
        fused = self.fusion(torch.cat((categorical, numerical), dim=-1))
        if self.config.arch == "gru":
            features, hidden = self.memory(fused, hidden)
        else:
            features, hidden = self.memory(fused), None
        return features, hidden

    def forward(
        self, tokens: Tensor, numbers: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return policy logits and memory without evaluating the optional value head."""
        features, hidden = self._forward_features(tokens, numbers, hidden)
        return self.policy(features), hidden

    def forward_with_value(
        self, tokens: Tensor, numbers: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Return policy logits, win-value logits [B,T], and shared memory."""
        if self.value_head is None:
            raise ValueError("Value prediction requires config.value_head=True")
        features, hidden = self._forward_features(tokens, numbers, hidden)
        return self.policy(features), self.value_head(features).squeeze(-1), hidden

    def forward_with_opponent(
        self, tokens: Tensor, numbers: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
        """Policy logits, value logits (None without a value head), opponent logits
        [B,T,classes+1] (last column = Tera logit), and memory."""
        if self.opponent_head is None:
            raise ValueError("Model has no opponent head")
        features, hidden = self._forward_features(tokens, numbers, hidden)
        value = (
            self.value_head(features).squeeze(-1) if self.value_head is not None else None
        )
        return self.policy(features), value, self.opponent_head(features), hidden


def masked_logits(logits: Tensor, masks: Tensor) -> Tensor:
    """All-false rows remain all -inf; callers must skip/fallback for those."""
    return logits.masked_fill(~masks.bool(), -torch.inf)


def save_checkpoint(model: PokeDecision, path: str | Path) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "feature_mean": model.feature_mean.cpu(),
            "feature_std": model.feature_std.cpu(),
        },
        path,
    )


def load_pokedecision(
    model_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[PokeDecision, Any, PokeDecisionConfig]:
    from pokeai.data.features import ObservationEncoder

    model_dir = Path(model_dir)
    if (model_dir / "model.py").is_file():
        return load_research_model(model_dir, device)
    checkpoint = torch.load(
        model_dir / "best.pt", map_location="cpu", weights_only=True
    )
    recorded = yaml.safe_load((model_dir / "config.yaml").read_text())
    config = PokeDecisionConfig(**recorded["model"])
    model = PokeDecision(
        config, checkpoint["feature_mean"], checkpoint["feature_std"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    encoder = ObservationEncoder.load(model_dir / "tokenizer.json")
    return model, encoder, config


def load_research_model(
    model_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, Any, Any]:
    """Autoresearch export: `model.py` (a snapshot of autoresearch/train.py) defines
    `build_model()`; `best.pt` is its state_dict. Same call contract as PokeDecision."""
    import importlib.util

    from pokeai.data.features import ObservationEncoder

    model_dir = Path(model_dir)
    spec = importlib.util.spec_from_file_location(
        f"research_model_{abs(hash(str(model_dir.resolve())))}", model_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {model_dir / 'model.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model()
    model.load_state_dict(
        torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    from pokeai.data.matchup import EXTRAS_DIM, EXTRAS_VERSION

    extras = getattr(model.config, "extras", 0)
    if extras not in (0, EXTRAS_DIM):
        raise ValueError(f"model expects {extras} extras; encoder provides {EXTRAS_DIM}")
    encoder = ObservationEncoder.load(
        model_dir / "tokenizer.json", EXTRAS_VERSION if extras else None
    )
    return model, encoder, model.config
