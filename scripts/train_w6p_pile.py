"""W6-P: fine-tune C on Metamon self-play (Alakazam-family POVs) instead of human replays.

One changed component vs the W6-A `control` arm (checkpoints/w6a/control): the training data.
Everything else matches control: init C, BC + 0.5·value BCE, AdamW lr 1e-4 wd 1e-4, batch 16
full trajectories, grad clip 1.0, fixed 3 epochs, last epoch kept. Reported: policy loss on
W4 val A (human-policy drift) and on the pile val split. Selection happens in battles only.

Run under the AGENTS.md resource wrapper (needs CUBLAS_WORKSPACE_CONFIG for determinism).
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from pokeai.data.w3 import load_split
from pokeai.data.w4_outcomes import load_outcomes
from pokeai.models.pokedecision import load_pokedecision, masked_logits, save_checkpoint
from pokeai.training.opp_head import val_half
from pokeai.training.train_bc import eligible_steps, masked_bc_loss, seed_everything

C_DIR = Path("models/pokedecision-v1-value")
OUT = Path("checkpoints/w6p/pile")
SEED = 20261019
VALUE_WEIGHT = 0.5
KEYS = ("tokens", "numbers", "masks", "actions", "valid")
# W4 human trajectories are at most 500 decisions; the pile has a few stall battles up to
# ~8k decisions that do not fit a padded batch on 12 GB. Drop them (train and val alike).
MAX_LEN = 500


class Trajectories(Dataset):
    """Full trajectories with a per-trajectory outcome (one uninterrupted hidden state)."""

    def __init__(self, arrays: dict, outcomes: np.ndarray, indices: list[int] | None = None):
        self.arrays, self.outcomes = arrays, outcomes
        self.indices = list(range(len(arrays["offsets"]) - 1)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        t = self.indices[index]
        start, end = self.arrays["offsets"][t], self.arrays["offsets"][t + 1]
        item = {k: torch.from_numpy(self.arrays[k][start:end]) for k in KEYS}
        item["outcomes"] = torch.full((end - start,), float(self.outcomes[t]))
        item["real"] = torch.ones(end - start, dtype=torch.bool)
        return item


def collate(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(item["actions"]) for item in batch])
    out = {k: torch.nn.utils.rnn.pad_sequence([item[k] for item in batch], batch_first=True)
           for k in batch[0]}
    out["valid"] &= torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
    return out


def loader(arrays, outcomes, indices=None, shuffle=False, seed=SEED):
    return DataLoader(Trajectories(arrays, outcomes, indices), batch_size=16, shuffle=shuffle,
                      collate_fn=collate, generator=torch.Generator().manual_seed(seed))


def short(arrays: dict) -> list[int]:
    return np.flatnonzero(np.diff(arrays["offsets"]) <= MAX_LEN).tolist()


def evaluate(model, data, device) -> dict:
    model.eval()
    loss = correct = steps = bce = value_steps = 0.0
    with torch.inference_mode():
        for batch in data:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, values, _ = model.forward_with_value(batch["tokens"].long(), batch["numbers"])
            valid = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
            n = int(valid.sum())
            if n:
                loss += float(masked_bc_loss(logits, batch["actions"], batch["masks"], valid)) * n
                predicted = masked_logits(logits[valid], batch["masks"][valid]).argmax(-1)
                correct += float((predicted == batch["actions"][valid]).sum())
                steps += n
            real = batch["real"]
            bce += float(F.binary_cross_entropy_with_logits(
                values[real], batch["outcomes"][real], reduction="sum"))
            value_steps += int(real.sum())
    return {"policy_loss": loss / steps, "policy_accuracy": correct / steps,
            "policy_steps": int(steps), "value_bce": bce / value_steps}


def train_epoch(model, data, device, optimizer) -> float:
    model.train()
    total = batches = 0
    for batch in data:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, values, _ = model.forward_with_value(batch["tokens"].long(), batch["numbers"])
        objective = VALUE_WEIGHT * F.binary_cross_entropy_with_logits(
            values[batch["real"]], batch["outcomes"][batch["real"]])
        valid = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
        bc = masked_bc_loss(logits, batch["actions"], batch["masks"], valid)
        if bc is not None:
            objective = objective + bc
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite W6-P objective")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(objective.detach())
        batches += 1
    return total / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--data", nargs="+", default=["datasets/w6p"],
                        help="pile batches; train splits are concatenated, val is the first's")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", default="reports/w6p_pile_train.json")
    args = parser.parse_args()
    started = time.monotonic()
    seed_everything(SEED)
    device = torch.device("cuda")
    model, _, _ = load_pokedecision(C_DIR, device)
    trains = [dict(np.load(Path(d) / "train.npz")) for d in args.data]
    pile_val = dict(np.load(Path(args.data[0]) / "val.npz"))
    w4_val = load_split("val", "datasets/w4")
    w4_outcomes = load_outcomes("datasets/w4", "val", w4_val)
    idx_a = [i for i, b in enumerate(w4_val["battle_ids"]) if val_half(str(b)) == "A"]
    evals = {"w4_val_A": loader(w4_val, w4_outcomes, idx_a),
             "pile_val": loader(pile_val, pile_val["outcomes"], short(pile_val))}
    train_set = ConcatDataset([Trajectories(t, t["outcomes"], short(t)) for t in trains])
    report = {"seed": SEED, "value_weight": VALUE_WEIGHT, "lr": 1e-4, "epochs": args.epochs,
              "data": args.data, "max_len": MAX_LEN, "train_trajectories": len(train_set),
              "dropped_long": {"train": sum(len(t["offsets"]) - 1 for t in trains)
                               - len(train_set),
                               "val": len(pile_val["offsets"]) - 1 - len(short(pile_val))},
              "C": {k: evaluate(model, v, device) for k, v in evals.items()}, "epochs_log": []}
    print(json.dumps({"C": report["C"]}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        data = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate,
                          generator=torch.Generator().manual_seed(SEED + epoch))
        objective = train_epoch(model, data, device, optimizer)
        row = {"epoch": epoch, "train_objective": objective,
               **{k: evaluate(model, v, device) for k, v in evals.items()}}
        report["epochs_log"].append(row)
        print(json.dumps(row), flush=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out / "best.pt")
    shutil.copy(C_DIR / "tokenizer.json", out / "tokenizer.json")
    recorded = yaml.safe_load((C_DIR / "config.yaml").read_text())
    recorded["model"] = vars(model.config)
    recorded |= {"policy_version": f"w6p-{out.name}", "init_checkpoint": str(C_DIR / "best.pt"),
                 "train_data": args.data}
    (out / "config.yaml").write_text(yaml.safe_dump(recorded, sort_keys=False))
    report["checkpoint"] = str(out)
    report["wall_time_s"] = time.monotonic() - started
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
