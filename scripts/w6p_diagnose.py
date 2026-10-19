"""W6-P diagnosis: why does 3x pile data barely move win rate? Offline, no battles.

For each model (C, P, pile300) and split (pile train sample, pile val, pile-b val, W4 val A) report
policy NLL, top-1/top-2 accuracy, and an error breakdown by decision kind:
- group: move (0-3) / switch (4-8) / tera-move (9-12); `group_acc` = right group,
  `within_acc` = right slot given the right group;
- forced (mask allows one group only) vs free choice;
- turn bucket (decision index in the battle).
Train-vs-val gap on the same model says bias (underfitting) vs variance (overfitting).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pokeai.data.w3 import load_split
from pokeai.models.pokedecision import load_pokedecision, masked_logits
from pokeai.training.opp_head import val_half
from pokeai.training.train_bc import eligible_steps

KEYS = ("tokens", "numbers", "masks", "actions", "valid")
GROUP = np.array([0] * 4 + [1] * 5 + [2] * 4)
BUCKETS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 10**9)]


def batches(arrays, indices, size=16):
    off = arrays["offsets"]
    for s in range(0, len(indices), size):
        chunk = indices[s:s + size]
        lengths = [off[t + 1] - off[t] for t in chunk]
        n = max(lengths)
        out = {}
        for k in KEYS:
            parts = [arrays[k][off[t]:off[t + 1]] for t in chunk]
            pad = np.zeros((len(chunk), n, *parts[0].shape[1:]), dtype=parts[0].dtype)
            for i, p in enumerate(parts):
                pad[i, :len(p)] = p
            out[k] = torch.from_numpy(pad)
        step = np.zeros((len(chunk), n), dtype=np.int64)
        real = np.zeros((len(chunk), n), dtype=bool)
        for i, length in enumerate(lengths):
            step[i, :length] = np.arange(length)
            real[i, :length] = True
        out["valid"] &= torch.from_numpy(real)
        out["step"] = torch.from_numpy(step)
        yield out


def collect(model, arrays, indices, device):
    rows = {k: [] for k in ("nll", "pred", "second", "act", "step", "groups_allowed")}
    model.eval()
    with torch.inference_mode():
        for batch in batches(arrays, indices):
            b = {k: v.to(device) for k, v in batch.items()}
            logits, _ = model(b["tokens"].long(), b["numbers"])
            valid = eligible_steps(b["actions"], b["masks"], b["valid"])
            lg = masked_logits(logits[valid], b["masks"][valid])
            logp = torch.log_softmax(lg, -1)
            act = b["actions"][valid].long()
            top2 = lg.topk(2, -1).indices
            mask = b["masks"][valid].bool().cpu().numpy()
            rows["nll"].append((-logp.gather(-1, act[:, None])[:, 0]).cpu().numpy())
            rows["pred"].append(top2[:, 0].cpu().numpy())
            rows["second"].append(top2[:, 1].cpu().numpy())
            rows["act"].append(act.cpu().numpy())
            rows["step"].append(b["step"][valid].cpu().numpy())
            rows["groups_allowed"].append(np.stack(
                [mask[:, GROUP == g].any(-1) for g in range(3)], -1).sum(-1))
    return {k: np.concatenate(v) for k, v in rows.items()}


def summarize(r):
    def block(sel):
        if not sel.any():
            return None
        act, pred, second = r["act"][sel], r["pred"][sel], r["second"][sel]
        right_group = GROUP[act] == GROUP[pred]
        return {"n": int(sel.sum()), "nll": float(r["nll"][sel].mean()),
                "acc": float((pred == act).mean()),
                "top2": float(((pred == act) | (second == act)).mean()),
                "group_acc": float(right_group.mean()),
                "within_acc": float((pred[right_group] == act[right_group]).mean())}
    allsel = np.ones(len(r["act"]), dtype=bool)
    free = r["groups_allowed"] > 1
    out = {"all": block(allsel), "forced_one_group": block(~free), "free": block(free)}
    for g, name in enumerate(("move", "switch", "tera")):
        out[f"teacher_{name}"] = block(free & (GROUP[r["act"]] == g))
    for lo, hi in BUCKETS:
        out[f"step_{lo}_{hi}"] = block(free & (r["step"] >= lo) & (r["step"] < hi))
    return out


def short(arrays, limit=500):
    return np.flatnonzero(np.diff(arrays["offsets"]) <= limit).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[
        "C=models/pokedecision-v1-value", "P=models/pokedecision-v2-pile",
        "pile300=checkpoints/w6p/pile300"])
    parser.add_argument("--train-sample", type=int, default=5000)
    parser.add_argument("--out", default="reports/w6p_diagnosis.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    pile = dict(np.load("datasets/w6p/train.npz"))
    rng = np.random.default_rng(0)
    train_idx = sorted(rng.choice(short(pile), args.train_sample, replace=False).tolist())
    val = dict(np.load("datasets/w6p/val.npz"))
    val_b = dict(np.load("datasets/w6p_b/val.npz"))
    w4 = load_split("val", "datasets/w4")
    w4_a = [i for i, b in enumerate(w4["battle_ids"]) if val_half(str(b)) == "A"]
    splits = {"pile_train_sample": (pile, train_idx), "pile_val": (val, short(val)),
              "pile_b_val": (val_b, short(val_b)), "w4_val_A": (w4, w4_a)}
    report = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, _, _ = load_pokedecision(path, device)
        report[name] = {s: summarize(collect(model, a, i, device)) for s, (a, i) in splits.items()}
        print(name, json.dumps({s: report[name][s]["all"] for s in splits}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
