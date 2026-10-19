"""Autoresearch: the ONE file the agent edits (model + optimizer + training loop).

Model = entity-pointer network (EPN). The observation is a fixed set of entities (our active,
4 moves, 5 bench Pokémon, the opponent's active, global context). Each entity type has ONE
small shared encoder; the GRU carries the battle state; each action is scored by a shared
additive-attention scorer over the entity it refers to (move i -> actions i and 9+i, bench j ->
action 4+j). Slot weights are shared, so the policy is permutation-equivariant over moves and
bench slots and stays small.

Rules (see program.md): the model section above `# ---- training ----` may import only torch,
and must define `build_model()`; training code lives under `main()`.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Token / number layout of DefaultObservationSpace (see program.md).
ACTIVE_TOKENS = (list(range(3, 10)), list(range(72, 79)))       # ours, opponent
ACTIVE_NUMBERS = (list(range(1, 16)), list(range(33, 48)))
MOVE_TOKENS = [[11 + 4 * i, 12 + 4 * i, 13 + 4 * i] for i in range(4)]
MOVE_NUMBERS = [[16 + 3 * i, 17 + 3 * i, 18 + 3 * i] for i in range(4)]
BENCH_TOKENS = [list(range(27 + 9 * j, 35 + 9 * j)) for j in range(5)]
BENCH_NUMBERS = [[28 + j] for j in range(5)]
GLOBAL_TOKENS = [0, 1, 80, 81, 82, 84, 86]
GLOBAL_NUMBERS = [0]


@dataclass
class Config:
    vocab_size: int = 2542
    d_token: int = 48
    d_entity: int = 96
    hidden_size: int = 256
    d_score: int = 128
    value_head: bool = True


class EntityEncoder(nn.Module):
    def __init__(self, n_tokens: int, n_numbers: int, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tokens * config.d_token + n_numbers, config.d_entity), nn.GELU(),
            nn.Linear(config.d_entity, config.d_entity))

    def forward(self, token_emb, numbers):
        return self.net(torch.cat((token_emb.flatten(-2), numbers), -1))


class Scorer(nn.Module):
    """score(h, e) = w . gelu(A h + B e): one scorer shared by all slots of an action kind."""

    def __init__(self, config: Config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.d_score)
        self.key = nn.Linear(config.d_entity, config.d_score)
        self.out = nn.Linear(config.d_score, 1)

    def forward(self, h, entities):  # h [B,T,H], entities [B,T,K,E] -> [B,T,K]
        return self.out(F.gelu(self.query(h).unsqueeze(-2) + self.key(entities))).squeeze(-1)


class Policy(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(48))
        self.register_buffer("feature_std", torch.ones(48))
        for name, rows in (("active_tokens", ACTIVE_TOKENS), ("move_tokens", MOVE_TOKENS),
                           ("bench_tokens", BENCH_TOKENS), ("active_numbers", ACTIVE_NUMBERS),
                           ("move_numbers", MOVE_NUMBERS), ("bench_numbers", BENCH_NUMBERS)):
            self.register_buffer(name, torch.tensor(rows), persistent=False)
        self.register_buffer("global_tokens", torch.tensor(GLOBAL_TOKENS), persistent=False)
        self.register_buffer("global_numbers", torch.tensor(GLOBAL_NUMBERS), persistent=False)
        self.tokens = nn.Embedding(config.vocab_size, config.d_token)
        self.side = nn.Parameter(torch.zeros(2, config.d_entity))
        self.active = EntityEncoder(7, 15, config)
        self.move = EntityEncoder(3, 3, config)
        self.bench = EntityEncoder(8, 1, config)
        self.context = EntityEncoder(len(GLOBAL_TOKENS), len(GLOBAL_NUMBERS), config)
        # kinds: context, ours, opponent, move x4, bench x5 (shared kind ids keep slot symmetry)
        self.register_buffer("kind_ids", torch.tensor([0, 1, 2] + [3] * 4 + [4] * 5),
                             persistent=False)
        self.kinds = nn.Embedding(5, config.d_entity)
        self.mix = nn.TransformerEncoderLayer(config.d_entity, 4, 2 * config.d_entity,
                                              dropout=0.0, batch_first=True, norm_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(5 * config.d_entity, config.hidden_size), nn.GELU())
        self.memory = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        self.move_score = Scorer(config)
        self.tera_score = Scorer(config)
        self.switch_score = Scorer(config)
        self.value = nn.Linear(config.hidden_size, 1)

    def encode(self, tokens, numbers):
        emb = self.tokens(tokens)                                   # [B,T,87,D]
        num = (numbers - self.feature_mean) / self.feature_std      # [B,T,48]
        active = self.active(emb[..., self.active_tokens, :], num[..., self.active_numbers])
        active = active + self.side                                 # [B,T,2,E]
        moves = self.move(emb[..., self.move_tokens, :], num[..., self.move_numbers])
        bench = self.bench(emb[..., self.bench_tokens, :], num[..., self.bench_numbers])
        context = self.context(emb[..., self.global_tokens, :], num[..., self.global_numbers])
        entities = torch.cat((context.unsqueeze(-2), active, moves, bench), -2)  # [B,T,12,E]
        lead = entities.shape[:-2]
        entities = self.mix((entities + self.kinds(self.kind_ids)).reshape(-1, 12, entities.shape[-1]))
        entities = entities.reshape(*lead, 12, -1)
        context, active, moves, bench = (entities[..., 0, :], entities[..., 1:3, :],
                                         entities[..., 3:7, :], entities[..., 7:, :])
        state = torch.cat((context, active.flatten(-2), moves.mean(-2), bench.mean(-2)), -1)
        return self.fusion(state), moves, bench

    def run(self, tokens, numbers, hidden):
        state, moves, bench = self.encode(tokens, numbers)
        h, hidden = self.memory(state, hidden)
        logits = torch.cat((self.move_score(h, moves), self.switch_score(h, bench),
                            self.tera_score(h, moves)), -1)
        return logits, h, hidden

    def forward(self, tokens, numbers, hidden=None):
        logits, _, hidden = self.run(tokens, numbers, hidden)
        return logits, hidden

    def forward_with_value(self, tokens, numbers, hidden=None):
        logits, h, hidden = self.run(tokens, numbers, hidden)
        return logits, self.value(h).squeeze(-1), hidden


def build_model() -> Policy:
    return Policy(Config())


# ---- training ----

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 16
VALUE_WEIGHT = 0.5
GRAD_CLIP = 1.0
WARMDOWN = 0.5   # fraction of the budget spent decaying LR linearly to 0
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
        progress = trained / H.TIME_BUDGET
        for group in optimizer.param_groups:  # constant, then linear warmdown to 0
            group["lr"] = LR * min(1.0, (1.0 - progress) / WARMDOWN)
        if cursor + BATCH > len(order):
            order, cursor = rng.permutation(data.indices), 0
        b = data.batch(order[cursor:cursor + BATCH], device)
        cursor += BATCH
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, values, _ = model.forward_with_value(b["tokens"], b["numbers"])
        logits, values = logits.float(), values.float()
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
