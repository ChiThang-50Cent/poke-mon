"""CF1.5 training + offline evaluation of the candidate head (frozen C features only)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from pokeai.cf.cf1_metrics import cluster_bootstrap
from pokeai.cf.head import Arrays, choose, delta_scores, make_head
from pokeai.cf.signal_stats import spearman

HUBER_DELTA = 0.1


def permute_labels(d: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle the non-greedy candidate labels within each root (a_C stays 0)."""
    rng = np.random.default_rng(seed)
    out = d.copy()
    for i in range(len(d)):
        idx = np.flatnonzero(valid[i])[1:]
        out[i, idx] = d[i, rng.permutation(idx)]
    return out


def root_gains(delta: np.ndarray, arr: Arrays, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(Q(a_head) - Q(a_C) per root, chosen candidate index)."""
    pick = choose(delta, arr.valid, tau)
    q = np.nan_to_num(arr.q)
    gain = q[np.arange(len(pick)), pick] - q[:, 0]
    return gain, pick


@torch.inference_mode()
def predict(head: torch.nn.Module, arr: Arrays, mean: np.ndarray, std: np.ndarray,
            device: str) -> np.ndarray:
    head.eval()
    h = torch.as_tensor((arr.h - mean) / std, dtype=torch.float32, device=device)
    out = delta_scores(
        head, h, torch.as_tensor(arr.slots, device=device),
        torch.as_tensor(arr.logp, device=device),
    )
    return out.cpu().numpy()


def _loss(kind: str, delta: torch.Tensor, d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if kind == "huber":
        ng = valid.clone()
        ng[:, 0] = False
        w = ng.float() / ng.float().sum(1, keepdim=True).clamp(min=1)
        per = torch.nn.functional.huber_loss(delta, d, delta=HUBER_DELTA, reduction="none")
        return (per * w).sum(1).mean()
    pair = valid[:, :, None] & valid[:, None, :] & (d[:, :, None] > d[:, None, :])
    w = pair.float() / pair.float().sum((1, 2), keepdim=True).clamp(min=1)
    margin = delta[:, :, None] - delta[:, None, :]
    return (torch.nn.functional.softplus(-margin) * w).sum((1, 2)).mean()


def fit(
    arch: str, loss: str, seed: int, train: Arrays, val: Arrays, mean: np.ndarray,
    std: np.ndarray, device: str, shuffle_labels: bool = False, max_epochs: int = 60,
    patience: int = 10, lr: float = 1e-3, weight_decay: float = 1e-2, batch: int = 128,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = make_head(arch, train.h.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    labels = permute_labels(train.d, train.valid, seed + 10_000) if shuffle_labels else train.d

    def tens(a: Arrays, d: np.ndarray) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor((a.h - mean) / std, dtype=torch.float32, device=device),
            torch.as_tensor(a.slots, device=device),
            torch.as_tensor(a.logp, device=device),
            torch.as_tensor(a.valid, device=device),
            torch.as_tensor(d, dtype=torch.float32, device=device),
        )

    h, slots, logp, valid, d = tens(train, labels)
    best, best_state, bad, curve = -np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.as_tensor(rng.permutation(len(h)), device=device)
        total = 0.0
        for start in range(0, len(h), batch):
            idx = order[start:start + batch]
            delta = delta_scores(head, h[idx], slots[idx], logp[idx])
            value = _loss(loss, delta, d[idx], valid[idx])
            opt.zero_grad()
            value.backward()
            opt.step()
            total += float(value.detach()) * len(idx)
        val_delta = predict(head, val, mean, std, device)
        gain, pick = root_gains(val_delta, val, 0.0)
        metric = float(gain.mean())
        curve.append({"epoch": epoch, "train_loss": total / len(h), "val_delta_tau0": metric,
                      "val_switch_rate": float((pick != 0).mean())})
        if metric > best + 1e-12:
            best, bad, best_state = metric, 0, copy.deepcopy(head.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return {"head": head, "best_val_delta_tau0": best,
            "best_epoch": max(curve, key=lambda c: c["val_delta_tau0"])["epoch"],
            "curve": curve}


def per_root_h1(delta: np.ndarray, arr: Arrays) -> tuple[np.ndarray, np.ndarray]:
    """Spearman(dQ_head, rollout Q) for >=3 candidates (nan undefined); sign agreement for
    2-candidate roots (nan when the rollout dQ is exactly 0)."""
    sp = np.full(len(arr.h), np.nan)
    sign = np.full(len(arr.h), np.nan)
    for i in range(len(arr.h)):
        n = int(arr.valid[i].sum())
        if n >= 3:
            sp[i] = spearman(delta[i, :n], arr.q[i, :n])
        elif n == 2:
            truth = arr.q[i, 1] - arr.q[i, 0]
            if truth != 0:
                sign[i] = float(np.sign(delta[i, 1]) == np.sign(truth))
    return sp, sign


def pooled_ci(values: np.ndarray, arr: Arrays, seed: int = 0) -> dict[str, Any]:
    lo, hi = cluster_bootstrap(values, list(arr.meta["pair"]), 5000, seed)
    return {"mean": float(values.mean()), "ci95": [lo, hi], "n": len(values)}
