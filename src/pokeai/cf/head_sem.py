"""CF1.5c semantic action head: MLP(hidden 128) on [h, slot one-hot, logπ, x(s,a)].

x(s,a) is the public ActionDescriptor (F1) optionally extended with the frozen C token
embedding of the candidate's move / incoming species (F2). Loss, optimiser and early-stopping
protocol are those of CF1.5 (original pairwise loss); only the input is wider.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from pokeai.cf.head import N_SLOTS, Arrays
from pokeai.cf.head_train import _loss, permute_labels
from pokeai.cf.signal_stats import spearman


class SemHead(nn.Module):
    def __init__(self, dim_h: int, dim_x: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_h + N_SLOTS + 1 + dim_x, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, h, slots, logp, x):
        onehot = torch.nn.functional.one_hot(slots.clamp(min=0), N_SLOTS).float()
        hh = h[:, None, :].expand(-1, slots.shape[1], -1)
        return self.net(torch.cat((hh, onehot, logp[..., None], x), dim=-1)).squeeze(-1)


def delta_sem(head, h, slots, logp, x) -> torch.Tensor:
    f = head(h, slots, logp, x)
    return f - f[:, :1]


@dataclass
class SemData:
    arr: Arrays
    x: np.ndarray  # [N, 4, E] float32

    def subset(self, sel: np.ndarray) -> SemData:
        return SemData(self.arr.subset(sel), self.x[sel])


@torch.inference_mode()
def predict_sem(head, data: SemData, mean: np.ndarray, std: np.ndarray, device: str):
    head.eval()
    a = data.arr
    out = delta_sem(
        head, torch.as_tensor((a.h - mean) / std, dtype=torch.float32, device=device),
        torch.as_tensor(a.slots, device=device), torch.as_tensor(a.logp, device=device),
        torch.as_tensor(data.x, dtype=torch.float32, device=device),
    )
    return out.cpu().numpy()


def root_spearman(delta: np.ndarray, arr: Arrays) -> np.ndarray:
    """Per-root Spearman(score, rollout Q) over valid candidates (nan when < 3 candidates)."""
    out = np.full(len(arr.h), np.nan)
    for i in range(len(arr.h)):
        n = int(arr.valid[i].sum())
        if n >= 3:
            out[i] = spearman(delta[i, :n], arr.q[i, :n])
    return out


def fit_sem(
    seed: int, train: SemData, val: SemData | None, mean: np.ndarray, std: np.ndarray,
    device: str, shuffle_labels: bool = False, max_epochs: int = 60, patience: int = 10,
    lr: float = 1e-3, weight_decay: float = 1e-2, batch: int = 128,
    fixed_epochs: int | None = None, use_x: bool = True,
) -> dict[str, Any]:
    """Train; with `val`, keep the epoch with the best mean per-root val Spearman;
    with `fixed_epochs` (and no val) train exactly that many epochs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a = train.arr
    dim_x = train.x.shape[2] if use_x else 0
    head = SemHead(a.h.shape[1], dim_x).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    labels = permute_labels(a.d, a.valid, seed + 10_000) if shuffle_labels else a.d
    h = torch.as_tensor((a.h - mean) / std, dtype=torch.float32, device=device)
    slots = torch.as_tensor(a.slots, device=device)
    logp = torch.as_tensor(a.logp, device=device)
    valid = torch.as_tensor(a.valid, device=device)
    d = torch.as_tensor(labels, dtype=torch.float32, device=device)
    x = torch.as_tensor(train.x[:, :, :dim_x], dtype=torch.float32, device=device)
    epochs = fixed_epochs or max_epochs
    best, best_state, bad, curve = -np.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.as_tensor(rng.permutation(len(h)), device=device)
        total = 0.0
        for start in range(0, len(h), batch):
            idx = order[start:start + batch]
            delta = delta_sem(head, h[idx], slots[idx], logp[idx], x[idx])
            value = _loss("pairwise", delta, d[idx], valid[idx])
            opt.zero_grad()
            value.backward()
            opt.step()
            total += float(value.detach()) * len(idx)
        row = {"epoch": epoch, "train_loss": total / len(h)}
        if val is not None:
            vd = predict_sem(
                head, SemData(val.arr, val.x[:, :, :dim_x]), mean, std, device
            )
            metric = float(np.nanmean(root_spearman(vd, val.arr)))
            row["val_spearman"] = metric
            curve.append(row)
            if metric > best + 1e-12:
                best, bad, best_state = metric, 0, copy.deepcopy(head.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            curve.append(row)
    if best_state is not None:
        head.load_state_dict(best_state)
    best_epoch = (max(curve, key=lambda c: c.get("val_spearman", -1))["epoch"]
                  if val is not None else epochs)
    return {"head": head, "best_val_spearman": best, "best_epoch": best_epoch,
            "curve": curve, "dim_x": dim_x}
