"""Fixed harness for autonomous policy-architecture research (karpathy/autoresearch pattern).

DO NOT MODIFY during a research run. `train.py` is the only file the agent edits.

  uv run python autoresearch/prepare.py      one-time: pile npz -> memory-mappable .npy

Data: Metamon self-play pile (pac-base gen9ou, Alakazam family) encoded with the production
ObservationEncoder (`datasets/w6p`, `datasets/w6p_b`). Train = both train splits. VAL (the agent's
metric) = a fixed 2,000-trajectory sample of `datasets/w6p_b/val`. `datasets/w6p/val` is NOT exposed
here: it is the sealed check for promotion. No human data, no spent data, no battles.

Model contract (what live play needs, see `pokeai.models.pokedecision.load_research_model`):
  model(tokens[B,T,87] long, numbers[B,T,48] float, hidden) -> (logits[B,T,13], hidden)
  model.forward_with_value(tokens, numbers, hidden) -> (logits, value_logits[B,T], hidden)
  model.config.value_head == True; numbers are raw (normalize inside the model with
  FEATURE_MEAN/FEATURE_STD); `hidden=None` starts a battle.
The metric is computed STEPWISE (T=1, hidden carried), exactly like live play, so a model can
never see future turns.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/autoresearch"
SOURCES = [ROOT / "datasets/w6p", ROOT / "datasets/w6p_b"]
C_DIR = ROOT / "models/pokedecision-v1-value"

TIME_BUDGET = 300           # seconds of training (wall clock, excluding startup and eval)
MAX_LEN = 500               # trajectories longer than this are dropped (train and val)
VAL_TRAJECTORIES = 2000
VAL_SEED = 20261020
MAX_PARAMS = 4_700_000     # human-set: below Metamon's smallest agent (Minikazam 4.7M); smaller wins ties
MAX_STEP_MS = 10.0          # one live decision (batch 1, T=1) on the GPU
KEYS = ("tokens", "numbers", "masks", "actions", "valid")
N_ACTIONS, TOKEN_LEN, N_NUMBERS = 13, 87, 48


def _c_stats() -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(C_DIR / "best.pt", map_location="cpu", weights_only=True)
    return checkpoint["feature_mean"].float(), checkpoint["feature_std"].float().clamp_min(1e-6)


FEATURE_MEAN, FEATURE_STD = _c_stats()


def prepare() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    encoder = (ROOT / "datasets/w4/encoder.json").read_bytes()
    for source in SOURCES:
        if (source / "encoder.json").read_bytes() != encoder:
            raise ValueError(f"{source}: encoder differs from production")
    for split, parts in (("train", [s / "train.npz" for s in SOURCES]),
                         ("val", [SOURCES[1] / "val.npz"])):
        loaded = [np.load(p) for p in parts]
        offsets, base = [np.zeros(1, np.int64)], 0
        for z in loaded:
            offsets.append(z["offsets"][1:] + base)
            base += int(z["offsets"][-1])
        np.save(CACHE / f"{split}_offsets.npy", np.concatenate(offsets))
        np.save(CACHE / f"{split}_outcomes.npy", np.concatenate([z["outcomes"] for z in loaded]))
        for key in KEYS:
            np.save(CACHE / f"{split}_{key}.npy", np.concatenate([z[key] for z in loaded]))
        print(split, "trajectories", len(np.load(CACHE / f"{split}_offsets.npy")) - 1, flush=True)
    (CACHE / "manifest.json").write_text(json.dumps(
        {"sources": [str(s.relative_to(ROOT)) for s in SOURCES], "val": "w6p_b/val"}, indent=1))


class Split:
    """Memory-mapped trajectories. `indices` = usable trajectories (length <= MAX_LEN)."""

    def __init__(self, split: str):
        self.offsets = np.load(CACHE / f"{split}_offsets.npy")
        self.outcomes = np.load(CACHE / f"{split}_outcomes.npy")
        self.arrays = {k: np.load(CACHE / f"{split}_{k}.npy", mmap_mode="r") for k in KEYS}
        lengths = np.diff(self.offsets)
        self.lengths = lengths
        self.indices = np.flatnonzero(lengths <= MAX_LEN)

    def batch(self, trajectories, device) -> dict[str, torch.Tensor]:
        """Pad full trajectories to [B, T]. `valid` marks steps with a usable BC label;
        `real` marks non-padding steps; `outcomes` is the POV's result per step."""
        lengths = [int(self.lengths[t]) for t in trajectories]
        n = max(lengths)
        out = {}
        for key in KEYS:
            ref = self.arrays[key]
            pad = np.zeros((len(trajectories), n, *ref.shape[1:]), dtype=ref.dtype)
            for i, t in enumerate(trajectories):
                pad[i, :lengths[i]] = ref[self.offsets[t]:self.offsets[t + 1]]
            out[key] = torch.from_numpy(pad)
        real = torch.arange(n)[None, :] < torch.tensor(lengths)[:, None]
        out["tokens"] = out["tokens"].long()
        out["real"] = real
        out["valid"] = out["valid"] & real & eligible(out["actions"], out["masks"])
        out["outcomes"] = torch.from_numpy(
            np.repeat(self.outcomes[list(trajectories)][:, None], n, 1).astype(np.float32))
        return {k: v.to(device, non_blocking=True) for k, v in out.items()}


def eligible(actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    ok = (actions >= 0) & (actions < N_ACTIONS) & masks.any(-1)
    return ok & masks.gather(-1, actions.clamp(0, N_ACTIONS - 1).unsqueeze(-1)).squeeze(-1)


def load_train() -> Split:
    return Split("train")


def _val_trajectories(split: Split) -> list[int]:
    rng = np.random.default_rng(VAL_SEED)
    return sorted(rng.choice(split.indices, VAL_TRAJECTORIES, replace=False).tolist())


@torch.inference_mode()
def evaluate(model: torch.nn.Module, device, batch_size: int = 128) -> dict:
    """Stepwise (live-play) evaluation on the fixed VAL sample. Lower val_nll is better."""
    model.eval()
    split = Split("val")
    chosen = _val_trajectories(split)
    nll = top1 = top2 = steps = 0.0
    for s in range(0, len(chosen), batch_size):
        b = split.batch(chosen[s:s + batch_size], device)
        hidden, logits = None, []
        for t in range(b["tokens"].shape[1]):
            out, hidden = model(b["tokens"][:, t:t + 1], b["numbers"][:, t:t + 1], hidden)
            logits.append(out)
        logits = torch.cat(logits, 1)
        if logits.shape != (*b["actions"].shape, N_ACTIONS):
            raise ValueError(f"bad logits shape {tuple(logits.shape)}")
        v = b["valid"]
        lg = logits[v].float().masked_fill(~b["masks"][v], -torch.inf)
        act = b["actions"][v]
        nll += float(F.cross_entropy(lg, act, reduction="sum"))
        top = lg.topk(2, -1).indices
        top1 += float((top[:, 0] == act).sum())
        top2 += float(((top[:, 0] == act) | (top[:, 1] == act)).sum())
        steps += int(v.sum())
    if not math.isfinite(nll):
        raise FloatingPointError("non-finite val_nll")
    return {"val_nll": nll / steps, "val_top1": top1 / steps, "val_top2": top2 / steps,
            "val_steps": int(steps)}


@torch.inference_mode()
def check_contract(model: torch.nn.Module, device) -> dict:
    """Params cap, value head, live-step latency, and stepwise == full-sequence parity."""
    params = sum(p.numel() for p in model.parameters())
    if params > MAX_PARAMS:
        raise ValueError(f"{params} parameters > {MAX_PARAMS}")
    if not getattr(getattr(model, "config", None), "value_head", False):
        raise ValueError("model.config.value_head must be True")
    model.eval()
    split = Split("val")
    b = split.batch(_val_trajectories(split)[:4], device)
    full, _ = model(b["tokens"], b["numbers"], None)
    hidden, steps = None, []
    for t in range(b["tokens"].shape[1]):
        out, hidden = model(b["tokens"][:, t:t + 1], b["numbers"][:, t:t + 1], hidden)
        steps.append(out)
    gap = float((torch.cat(steps, 1) - full)[b["real"]].abs().max())
    if gap > 1e-2:  # float/cuDNN noise is ~1e-3; any use of future turns is O(1)
        raise ValueError(f"stepwise != full-sequence logits (max gap {gap:.2e})")
    _, values, _ = model.forward_with_value(b["tokens"], b["numbers"], None)
    if values.shape != b["actions"].shape:
        raise ValueError(f"bad value shape {tuple(values.shape)}")
    one = {k: b[k][:1, :1] for k in ("tokens", "numbers")}
    for _ in range(5):
        model(one["tokens"], one["numbers"], None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    hidden = None
    for _ in range(50):
        _, hidden = model(one["tokens"], one["numbers"], hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / 50 * 1000
    if step_ms > MAX_STEP_MS:
        raise ValueError(f"live step {step_ms:.1f} ms > {MAX_STEP_MS} ms")
    return {"num_params": params, "step_ms": step_ms, "parity_gap": gap}


def save(model: torch.nn.Module, out_dir: Path, train_file: Path, summary: dict) -> None:
    """Export a candidate that `load_research_model` (live play) can load."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best.pt")
    shutil.copy(train_file, out_dir / "model.py")
    shutil.copy(C_DIR / "tokenizer.json", out_dir / "tokenizer.json")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
