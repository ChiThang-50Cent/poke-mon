"""CF1.5 action-conditioned head over C's frozen root features.

f(h, a) is scored for each candidate a; the prediction is dQ_head(a) = f(h, a) - f(h, a_C)
(a_C is candidate 0 and so scores exactly 0). The ONLY inputs are C's frozen root feature
vector h, the one-hot 13-slot action identity and C's masked log-probability of a: no
sampled-world or true-state variable can reach the head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

N_SLOTS = 13
TAUS = (0.0, 0.01, 0.02, 0.05)


class LinearHead(nn.Module):
    """Per-slot linear scorer over h plus a shared log-prob coefficient."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(N_SLOTS, dim))
        self.bias = nn.Parameter(torch.zeros(N_SLOTS))
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor, slots: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        idx = slots.clamp(min=0)
        score = (h[:, None, :] * self.weight[idx]).sum(-1) + self.bias[idx]
        return score + self.beta * logp


class MLPHead(nn.Module):
    """2-layer MLP (hidden 128) on [h, one-hot slot, log-prob]."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + N_SLOTS + 1, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, h: torch.Tensor, slots: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(slots.clamp(min=0), N_SLOTS).float()
        hh = h[:, None, :].expand(-1, slots.shape[1], -1)
        return self.net(torch.cat((hh, onehot, logp[..., None]), dim=-1)).squeeze(-1)


def make_head(arch: str, dim: int) -> nn.Module:
    return {"linear": LinearHead, "mlp": MLPHead}[arch](dim)


def delta_scores(head: nn.Module, h, slots, logp) -> torch.Tensor:
    """dQ_head per candidate; column 0 (a_C) is exactly 0."""
    f = head(h, slots, logp)
    return f - f[:, :1]


def choose(delta: np.ndarray, valid: np.ndarray, tau: float) -> np.ndarray:
    """Candidate index per root: argmax dQ_head over valid non-greedy candidates if it
    exceeds tau, else a_C (index 0). This is the live and offline decision rule."""
    masked = np.where(valid, delta, -np.inf)
    masked[:, 0] = -np.inf
    best = masked.argmax(axis=1)
    top = masked[np.arange(len(best)), best]
    return np.where(top > tau, best, 0)


@dataclass
class Arrays:
    h: np.ndarray        # [N, D] float32
    logp: np.ndarray     # [N, 4] log-prob of each candidate slot
    slots: np.ndarray    # [N, 4] int, -1 pad
    valid: np.ndarray    # [N, 4] bool
    q: np.ndarray        # [N, 4] mean outcome per candidate (nan pad)
    meta: dict[str, np.ndarray]

    @property
    def d(self) -> np.ndarray:
        """Rollout dQ-hat per candidate (a_C = 0, pad = 0)."""
        out = np.nan_to_num(self.q - self.q[:, :1])
        out[~self.valid] = 0.0
        return out

    def subset(self, sel: np.ndarray) -> Arrays:
        return Arrays(self.h[sel], self.logp[sel], self.slots[sel], self.valid[sel],
                      self.q[sel], {k: v[sel] for k, v in self.meta.items()})


def load_arrays(path: str) -> Arrays:
    z = np.load(path, allow_pickle=False)
    slots = z["slots"].astype(np.int64)
    valid = slots >= 0
    lp = np.take_along_axis(z["logp"], np.clip(slots, 0, None), axis=1)
    q = np.nanmean(z["y"], axis=2)
    meta = {k: z[k] for k in ("root", "group", "source", "split", "opponent", "phase",
                              "forced_switch", "pair", "reasons")}
    return Arrays(z["h"], np.where(valid, lp, 0.0).astype(np.float32), slots, valid, q, meta)


def save_head(path: Path, head: nn.Module, config: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path / "head.pt")
    (path / "config.json").write_text(json.dumps(config, indent=1))


class LiveHead:
    """Loads a saved head and applies the decision rule to C's root features."""

    def __init__(self, path: str, device: str = "cpu") -> None:
        root = Path(path)
        self.config = json.loads((root / "config.json").read_text())
        self.mean = torch.tensor(self.config["h_mean"], dtype=torch.float32, device=device)
        self.std = torch.tensor(self.config["h_std"], dtype=torch.float32, device=device)
        self.head = make_head(self.config["arch"], len(self.config["h_mean"])).to(device)
        self.head.load_state_dict(torch.load(root / "head.pt", map_location=device,
                                             weights_only=True))
        self.head.eval()
        self.tau = float(self.config["tau"])
        self.device = device

    @torch.inference_mode()
    def decide(self, features: list[float], logp: list[float], candidates: list[int]) -> int:
        """Slot to play: a head-chosen candidate if its dQ_head > tau, else candidates[0]."""
        h = (torch.tensor(features, dtype=torch.float32, device=self.device)
             - self.mean) / self.std
        slots = torch.tensor([candidates], dtype=torch.long, device=self.device)
        lp = torch.tensor([[logp[c] for c in candidates]], dtype=torch.float32,
                          device=self.device)
        delta = delta_scores(self.head, h[None], slots, lp).cpu().numpy()
        pick = choose(delta, np.ones_like(delta, dtype=bool), self.tau)[0]
        return int(candidates[int(pick)])
