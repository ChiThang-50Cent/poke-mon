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
    args = ap.parse_args()
    budget = args.minutes * 60
    out = Path(args.out)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    model = T.build_model()
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
        b = data.batch(order[cursor:cursor + T.BATCH], device)
        cursor += T.BATCH
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, values, _ = model.forward_with_value(b["tokens"], b["numbers"])
        logits, values = logits.float(), values.float()
        v = b["valid"]
        policy = F.cross_entropy(logits[v].masked_fill(~b["masks"][v], -torch.inf),
                                 b["actions"][v])
        value = F.binary_cross_entropy_with_logits(values[b["real"]], b["outcomes"][b["real"]])
        loss = policy + T.VALUE_WEIGHT * value
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
               "minutes": args.minutes, "seed": args.seed, "batch": T.BATCH, "lr": T.LR}
    H.save(model, out, ROOT / "autoresearch/train.py", summary)
    print("saved", out)


if __name__ == "__main__":
    main()
