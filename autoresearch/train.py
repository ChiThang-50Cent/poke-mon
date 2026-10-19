"""Autoresearch: the ONE file the agent edits (model + optimizer + training loop).

Baseline = the production architecture (C: token embeddings mean/max-pooled, numeric MLP,
GRU 384, 13-way policy head, value head), trained from scratch for TIME_BUDGET seconds.

Rules (see program.md): the model section above `# ---- training ----` may import only torch,
and must define `build_model()`; training code lives under `main()`.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab_size: int = 2542
    token_len: int = 87
    d_model: int = 96
    hidden_size: int = 384
    numeric_size: int = 128
    flatten_dim: int = 24
    value_head: bool = True


class Policy(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(48))
        self.register_buffer("feature_std", torch.ones(48))
        self.tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.fields = nn.Embedding(config.token_len, config.d_model)
        self.token_projection = nn.Linear(config.d_model, config.d_model)
        self.numeric = nn.Sequential(
            nn.Linear(48, config.numeric_size), nn.GELU(),
            nn.Linear(config.numeric_size, config.numeric_size), nn.GELU())
        self.token_flatten = nn.Linear(config.d_model, config.flatten_dim)
        self.fusion = nn.Sequential(
            nn.Linear(config.token_len * config.flatten_dim + config.numeric_size,
                      config.hidden_size), nn.GELU())
        self.memory = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        self.policy = nn.Linear(config.hidden_size, 13)
        self.value = nn.Linear(config.hidden_size, 1)

    def features(self, tokens, numbers, hidden):
        positions = torch.arange(self.config.token_len, device=tokens.device)
        embedded = F.gelu(self.token_projection(self.tokens(tokens) + self.fields(positions)))
        categorical = self.token_flatten(embedded).flatten(-2)  # keeps each field's slot
        numerical = self.numeric((numbers - self.feature_mean) / self.feature_std)
        fused = self.fusion(torch.cat((categorical, numerical), -1))
        return self.memory(fused, hidden)

    def forward(self, tokens, numbers, hidden=None):
        features, hidden = self.features(tokens, numbers, hidden)
        return self.policy(features), hidden

    def forward_with_value(self, tokens, numbers, hidden=None):
        features, hidden = self.features(tokens, numbers, hidden)
        return self.policy(features), self.value(features).squeeze(-1), hidden


def build_model() -> Policy:
    return Policy(Config())


# ---- training ----

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 16
VALUE_WEIGHT = 0.5
GRAD_CLIP = 1.0
SEED = 20261019


def main() -> None:
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare as H

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda")
    started = time.time()
    model = build_model()
    model.feature_mean.copy_(H.FEATURE_MEAN)
    model.feature_std.copy_(H.FEATURE_STD)
    model.to(device)
    data = H.load_train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trained, steps, order, cursor = 0.0, 0, rng.permutation(data.indices), 0
    H.check_contract(model, device)  # fail fast, before spending the budget
    model.train()
    while trained < H.TIME_BUDGET:
        tick = time.time()
        if cursor + BATCH > len(order):
            order, cursor = rng.permutation(data.indices), 0
        b = data.batch(order[cursor:cursor + BATCH], device)
        cursor += BATCH
        logits, values, _ = model.forward_with_value(b["tokens"], b["numbers"])
        v = b["valid"]
        policy = F.cross_entropy(logits[v].masked_fill(~b["masks"][v], -torch.inf),
                                 b["actions"][v])
        value = F.binary_cross_entropy_with_logits(values[b["real"]], b["outcomes"][b["real"]])
        loss = policy + VALUE_WEIGHT * value
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        torch.cuda.synchronize()
        trained += time.time() - tick
        steps += 1
    contract = H.check_contract(model, device)
    metrics = H.evaluate(model, device)
    summary = {**metrics, **contract, "training_seconds": trained, "num_steps": steps,
               "trajectories_seen": steps * BATCH,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
               "total_seconds": time.time() - started}
    H.save(model, Path("checkpoints/autoresearch/last"), Path(__file__), summary)
    print("---")
    for key, value in summary.items():
        print(f"{key + ':':20s}{value:.6f}" if isinstance(value, float) else f"{key + ':':20s}{value}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
