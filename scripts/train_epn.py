"""Train the autoresearch policy (`autoresearch/train.py:build_model`) for longer than the 5-minute
loop budget, with a periodic stepwise val curve.

Same data, loss, optimizer and LR schedule as the loop (constant LR, then linear warmdown to 0
over the last WARMDOWN fraction of the budget); only the budget and the logging differ.
Every --eval-every minutes of training it records `prepare.evaluate` (fixed 2,000-trajectory
w6p_b/val sample) and the epochs seen, so overfitting shows up as val_nll turning upward.
The final model is exported with `prepare.save` (loadable by `load_pokedecision`).

    uv run python scripts/train_epn.py --minutes 60 --eval-every 10 --out checkpoints/epn/long60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))

import prepare as H
import train as T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--eval-every", type=float, default=10.0, help="minutes of training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=T.SEED)
    ap.add_argument("--cache", default=None,
                    help="prepare.Split-layout cache (default: the autoresearch cache); val "
                         "files must be the autoresearch val sample (w7_build_cache.py)")
    ap.add_argument("--kakuna-weight", type=float, default=0.0,
                    help="policy loss = (1-w)*CE(data action) + w*CE(Kakuna probs) on rows the "
                         "cache's train_kakuna.npy labels; data CE only elsewhere")
    ap.add_argument("--config", nargs="*", default=[], metavar="FIELD=INT",
                    help="override train.Config fields (e.g. d_entity=192); baked into the "
                         "exported model.py so load_pokedecision rebuilds the same shape")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="W8 E2: add w * (CE(opponent next action) + CE(opponent revealed-move "
                         "set)) from train.opp_aux_labels; builds the model with aux=1")
    args = ap.parse_args()
    overrides = {k: int(v) for k, v in (kv.split("=", 1) for kv in args.config)}
    if args.aux_weight:
        overrides["aux"] = 1
    if args.cache:
        H.CACHE = Path(args.cache).resolve()
    kakuna = np.load(H.CACHE / "train_kakuna.npy", mmap_mode="r") if args.kakuna_weight else None
    budget = args.minutes * 60
    out = Path(args.out)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    model = T.Policy(T.Config(**overrides))
    model.feature_mean.copy_(H.FEATURE_MEAN)
    model.feature_std.copy_(H.FEATURE_STD)
    model.to(device)
    data = H.load_train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    H.check_contract(model, device)
    trained, steps, cursor, epochs = 0.0, 0, 0, 0
    order = rng.permutation(data.indices)
    next_eval, curve = args.eval_every * 60, []
    model.train()
    while trained < budget:
        tick = time.time()
        for group in optimizer.param_groups:
            group["lr"] = T.LR * min(1.0, (1.0 - trained / budget) / T.WARMDOWN)
        if cursor + T.BATCH > len(order):
            order, cursor, epochs = rng.permutation(data.indices), 0, epochs + 1
        idx = order[cursor:cursor + T.BATCH]
        b = data.batch(idx, device)
        cursor += T.BATCH
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.aux_weight:
                logits, values, opp_next, opp_set = model.forward_aux(b["tokens"], b["numbers"])
            else:
                logits, values, _ = model.forward_with_value(b["tokens"], b["numbers"])
        logits, values = logits.float(), values.float()
        v = b["valid"]
        legal = b["masks"] | ~b["masks"].any(-1, keepdim=True)  # padding rows: no -inf-only rows
        logp = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), -1)
        data_nll = -logp[v].gather(-1, b["actions"][v].unsqueeze(-1)).squeeze(-1)
        if kakuna is None:
            policy = data_nll.mean()
        else:
            target = pad_labels(kakuna, data, idx, b["real"].shape[1]).to(device)
            labeled = b["real"] & b["masks"].any(-1) & torch.isfinite(target).all(-1)
            soft = -(target[labeled] * torch.where(
                target[labeled] > 0, logp[labeled], torch.zeros_like(logp[labeled]))).sum(-1)
            w_data = torch.where(labeled[v], 1.0 - args.kakuna_weight, 1.0)
            policy = ((w_data * data_nll).sum() + args.kakuna_weight * soft.sum()) / (
                w_data.sum() + args.kakuna_weight * labeled.sum())
        value = F.binary_cross_entropy_with_logits(values[b["real"]], b["outcomes"][b["real"]])
        loss = policy + T.VALUE_WEIGHT * value
        if args.aux_weight:
            nxt, target = T.opp_aux_labels(b["tokens"], b["real"], model.config.vocab_size)
            aux = F.cross_entropy(opp_next.float()[nxt >= 0], nxt[nxt >= 0])
            has = target.sum(-1) > 0
            aux = aux - (target[has] * torch.log_softmax(opp_set.float()[has], -1)).sum(-1).mean()
            loss = loss + args.aux_weight * aux
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), T.GRAD_CLIP)
        optimizer.step()
        torch.cuda.synchronize()
        trained += time.time() - tick
        steps += 1
        if trained >= next_eval or trained >= budget:
            point = {"train_minutes": trained / 60, "steps": steps,
                     "epochs": epochs + cursor / len(order), **H.evaluate(model, device)}
            curve.append(point)
            print(json.dumps(point), flush=True)
            model.train()
            next_eval += args.eval_every * 60
    summary = {**curve[-1], **H.check_contract(model, device), "curve": curve,
               "minutes": args.minutes, "seed": args.seed, "batch": T.BATCH, "lr": T.LR,
               "cache": str(H.CACHE), "kakuna_weight": args.kakuna_weight,
               "aux_weight": args.aux_weight, "config": overrides}
    H.save(model, out, ROOT / "autoresearch/train.py", summary)
    if overrides:
        with open(out / "model.py", "a") as f:
            f.write(f"\n\n# train_epn.py --config overrides\ndef build_model() -> Policy:\n"
                    f"    return Policy(Config(**{overrides!r}))\n")
    print("saved", out)


def pad_labels(labels: np.ndarray, data, trajectories, n: int) -> torch.Tensor:
    """Kakuna probs padded to the batch's [B, T, 13] layout (NaN = unlabeled or padding)."""
    out = np.full((len(trajectories), n, labels.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(trajectories):
        out[i, :int(data.lengths[t])] = labels[data.offsets[t]:data.offsets[t + 1]]
    return torch.from_numpy(out)


if __name__ == "__main__":
    main()
