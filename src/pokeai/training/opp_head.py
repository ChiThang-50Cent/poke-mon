"""W6-A opponent-action head: batching, learner-view support, loss, metrics, baselines.

Support at a row (public at decision time): every move class, the opponent's team-preview
species except its current active, and the two <unk> classes. Labels never enter inputs.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

STEP_KEYS = ("tokens", "numbers", "masks", "actions", "valid")
LABEL_KEYS = ("opp_class", "opp_tera", "opp_preview", "opp_active")
PAD = {"actions": -1, "opp_class": -1, "opp_tera": -1, "opp_preview": -1, "opp_active": -1}


def val_half(battle_id: str) -> str:
    """Deterministic battle-level split of W4 val: 'A' selects, 'B' reports."""
    return "A" if int(hashlib.sha256(battle_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


class OpponentSequences(Dataset):
    """Full trajectories (one uninterrupted hidden state each), with W6-A labels."""

    def __init__(self, arrays: dict, labels: dict, outcomes: np.ndarray,
                 trajectories: list[int] | None = None) -> None:
        self.arrays, self.labels, self.outcomes = arrays, labels, outcomes
        offsets = arrays["offsets"]
        self.indices = list(range(len(offsets) - 1)) if trajectories is None else trajectories

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        t = self.indices[index]
        start, end = self.arrays["offsets"][t], self.arrays["offsets"][t + 1]
        item = {k: torch.from_numpy(self.arrays[k][start:end]) for k in STEP_KEYS}
        item |= {k: torch.from_numpy(self.labels[k][start:end]) for k in LABEL_KEYS}
        item["outcomes"] = torch.full((end - start,), float(self.outcomes[t]))
        item["real"] = torch.ones(end - start, dtype=torch.bool)
        return item


def collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    lengths = torch.tensor([len(item["actions"]) for item in batch])
    result = {
        key: torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=PAD.get(key, 0))
        for key in batch[0]
    }
    result["valid"] &= torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
    return result


def move_class_mask(vocab: list[str]) -> Tensor:
    """Classes always in support: every move class and both <unk> classes."""
    return torch.tensor(
        [c.startswith("move:") or c.endswith("<unk>") for c in vocab], dtype=torch.bool)


def support(preview: Tensor, active: Tensor, always: Tensor) -> Tensor:
    """[...,K] learner-view support: moves/unk always; preview species except active."""
    classes = torch.arange(len(always), device=preview.device)
    switch = (preview.unsqueeze(-1) == classes).any(-2)  # padding -1 matches nothing
    current = active.unsqueeze(-1) == classes
    return always | (switch & ~current)


def opponent_loss(opp_logits: Tensor, batch: dict[str, Tensor], always: Tensor
                  ) -> tuple[Tensor | None, Tensor | None, int, int]:
    """Masked class CE over labelled rows and Tera BCE over labelled move rows."""
    labelled = (batch["opp_class"] >= 0) & batch["real"]
    tera = (batch["opp_tera"] >= 0) & batch["real"]
    ce = bce = None
    if labelled.any():
        logits = opp_logits[..., :-1][labelled]
        mask = support(batch["opp_preview"][labelled], batch["opp_active"][labelled], always)
        ce = F.cross_entropy(logits.masked_fill(~mask, -torch.inf),
                             batch["opp_class"][labelled].long())
    if tera.any():
        bce = F.binary_cross_entropy_with_logits(
            opp_logits[..., -1][tera], batch["opp_tera"][tera].float())
    return ce, bce, int(labelled.sum()), int(tera.sum())


class PredictionMetrics:
    """NLL, top-k, P(switch) Brier, Tera Brier/BCE and top-1 ECE over labelled rows."""

    def __init__(self, switch_classes: Tensor, bins: int = 15) -> None:
        self.switch_classes = switch_classes
        self.bins = bins
        self.nll = self.top1 = self.top5 = self.n = 0.0
        self.switch_brier = 0.0
        self.tera_brier = self.tera_bce = self.n_tera = 0.0
        self.conf = np.zeros(bins)
        self.acc = np.zeros(bins)
        self.count = np.zeros(bins)

    def update(self, probs: Tensor, labels: Tensor, tera_p: Tensor | None = None,
               tera_y: Tensor | None = None) -> None:
        """probs [N,K] already restricted to support and normalised."""
        labels = labels.long()
        picked = probs.gather(-1, labels[:, None]).squeeze(-1).clamp_min(1e-12)
        self.nll += float(-picked.log().sum())
        top = probs.topk(min(5, probs.shape[-1]), dim=-1).indices
        hit1 = top[:, 0] == labels
        self.top1 += float(hit1.sum())
        self.top5 += float((top == labels[:, None]).any(-1).sum())
        p_switch = probs[:, self.switch_classes].sum(-1)
        y_switch = self.switch_classes[labels].float()
        self.switch_brier += float(((p_switch - y_switch) ** 2).sum())
        confidence = probs.max(-1).values.cpu().numpy()
        index = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
        np.add.at(self.conf, index, confidence)
        np.add.at(self.acc, index, hit1.cpu().numpy())
        np.add.at(self.count, index, 1)
        self.n += len(labels)
        if tera_p is not None and tera_y is not None and len(tera_p):
            y = tera_y.float()
            self.tera_brier += float(((tera_p - y) ** 2).sum())
            self.tera_bce += float(F.binary_cross_entropy(
                tera_p.clamp(1e-7, 1 - 1e-7), y, reduction="sum"))
            self.n_tera += len(tera_p)

    def result(self) -> dict[str, float]:
        filled = self.count > 0
        ece = float(np.abs(self.acc[filled] - self.conf[filled]).sum() / self.n)
        return {
            "rows": int(self.n), "nll": self.nll / self.n, "top1": self.top1 / self.n,
            "top5": self.top5 / self.n, "switch_brier": self.switch_brier / self.n,
            "top1_ece": ece, "tera_rows": int(self.n_tera),
            "tera_brier": self.tera_brier / max(self.n_tera, 1),
            "tera_bce": self.tera_bce / max(self.n_tera, 1),
        }
