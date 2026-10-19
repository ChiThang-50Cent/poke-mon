"""W6-A: opponent-action head on C, prediction tier (W4 train; val A selects, val B reports).

Arms (fixed, no grid):
- frozen: C's weights untouched (asserted), only the opponent head trains.
- joint: init C + the frozen arm's head; BC + 0.5·value BCE + LAMBDA·(class CE + Tera BCE).
- control: same as joint without the opponent loss (isolates the auxiliary task).
  Fixed 3 epochs each; eligible only if final val-A policy loss <= C + 0.01.
Baselines on the same val-B rows and support: train frequency, and frequency conditioned on
the opponent's active species. W4 test is spent and never opened.

Run under the AGENTS.md resource wrapper (needs CUBLAS_WORKSPACE_CONFIG for determinism).
"""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from pokeai.data.opp_labels import load_labels
from pokeai.data.w3 import load_split
from pokeai.data.w4_outcomes import load_outcomes
from pokeai.models.pokedecision import (
    PokeDecision,
    PokeDecisionConfig,
    load_pokedecision,
    masked_logits,
    save_checkpoint,
)
from pokeai.training.opp_head import (
    OpponentSequences,
    PredictionMetrics,
    collate,
    move_class_mask,
    opponent_loss,
    support,
    val_half,
)
from pokeai.training.train_bc import eligible_steps, masked_bc_loss, seed_everything

C_DIR = Path("models/pokedecision-v1-value")
SEED = 20261019
LAMBDA = 0.5
VALUE_WEIGHT = 0.5
POLICY_SLACK = 0.01


def with_head(base: PokeDecision, classes: int, device: torch.device) -> PokeDecision:
    config = PokeDecisionConfig(**(vars(base.config) | {"opponent_classes": classes}))
    model = PokeDecision(config, base.feature_mean.cpu(), base.feature_std.cpu())
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if unexpected or sorted(missing) != ["opponent_head.bias", "opponent_head.weight"]:
        raise ValueError(f"Unexpected C transfer: missing={missing} unexpected={unexpected}")
    return model.to(device)


def loader(arrays, labels, outcomes, indices=None, shuffle=False, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(OpponentSequences(arrays, labels, outcomes, indices), batch_size=16,
                      shuffle=shuffle, collate_fn=collate, generator=generator)


def evaluate(model, data, device, always, switch_classes):
    """Opponent metrics (if the model has a head) and policy/value metrics."""
    model.eval()
    metrics = PredictionMetrics(switch_classes)
    policy_loss = policy_correct = steps = 0.0
    value_bce = value_steps = 0.0
    with torch.inference_mode():
        for batch in data:
            batch = {k: v.to(device) for k, v in batch.items()}
            tokens, numbers = batch["tokens"].long(), batch["numbers"]
            if model.opponent_head is not None:
                logits, values, opp, _ = model.forward_with_opponent(tokens, numbers)
                labelled = (batch["opp_class"] >= 0) & batch["real"]
                if labelled.any():
                    mask = support(batch["opp_preview"][labelled],
                                   batch["opp_active"][labelled], always)
                    probs = opp[..., :-1][labelled].masked_fill(~mask, -torch.inf).softmax(-1)
                    tera = (batch["opp_tera"] >= 0) & batch["real"]
                    metrics.update(probs, batch["opp_class"][labelled],
                                   opp[..., -1][tera].sigmoid(), batch["opp_tera"][tera])
            else:
                logits, values, _ = model.forward_with_value(tokens, numbers)
            valid = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
            n = int(valid.sum())
            if n:
                loss = masked_bc_loss(logits, batch["actions"], batch["masks"], valid)
                policy_loss += float(loss) * n
                predicted = masked_logits(logits[valid], batch["masks"][valid]).argmax(-1)
                policy_correct += float((predicted == batch["actions"][valid]).sum())
                steps += n
            real = batch["real"]
            value_bce += float(F.binary_cross_entropy_with_logits(
                values[real], batch["outcomes"][real], reduction="sum"))
            value_steps += int(real.sum())
    result = {"policy_loss": policy_loss / steps, "policy_accuracy": policy_correct / steps,
              "policy_steps": int(steps), "value_bce": value_bce / value_steps}
    if model.opponent_head is not None:
        result["opponent"] = metrics.result()
    return result


def train_epoch(model, data, device, optimizer, always, opp_weight):
    """opp_weight None: opponent loss only (frozen arm). Otherwise BC + value + weight·opp."""
    model.train()
    totals = {"objective": 0.0, "batches": 0}
    for batch in data:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, values, opp, _ = model.forward_with_opponent(
            batch["tokens"].long(), batch["numbers"])
        ce, bce, _, _ = opponent_loss(opp, batch, always)
        objective = sum(x for x in (ce, bce) if x is not None)
        if opp_weight is not None:
            objective = opp_weight * objective + VALUE_WEIGHT * F.binary_cross_entropy_with_logits(
                values[batch["real"]], batch["outcomes"][batch["real"]])
            valid = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
            bc = masked_bc_loss(logits, batch["actions"], batch["masks"], valid)
            if bc is not None:
                objective = objective + bc
        if not torch.is_tensor(objective):
            continue
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite W6-A objective (label outside support?)")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["objective"] += float(objective.detach())
        totals["batches"] += 1
    return totals["objective"] / max(totals["batches"], 1)


def baselines(ltrain, lval, rows_b, vocab, device, always, switch_classes):
    """Count baselines over val-B labelled rows with the same learner-view support."""
    k = len(vocab)
    labelled = ltrain["opp_class"] >= 0
    counts = np.bincount(ltrain["opp_class"][labelled], minlength=k).astype(np.float64) + 1
    global_p = counts / counts.sum()
    active = ltrain["opp_active"][labelled]
    known = active >= 0
    conditional = np.zeros((k, k))
    np.add.at(conditional, (active[known], ltrain["opp_class"][labelled][known]), 1)
    conditional = conditional + 10.0 * global_p[None, :]
    tera = ltrain["opp_tera"][ltrain["opp_tera"] >= 0]
    tera_rate = float(tera.mean())
    out = {}
    rows = rows_b[lval["opp_class"][rows_b] >= 0]
    for name in ("train_frequency", "species_conditional"):
        metrics = PredictionMetrics(switch_classes)
        for chunk in np.array_split(rows, max(1, len(rows) // 20000)):
            preview = torch.from_numpy(lval["opp_preview"][chunk]).to(device)
            act = torch.from_numpy(lval["opp_active"][chunk]).to(device)
            mask = support(preview, act, always)
            if name == "train_frequency":
                prior = torch.from_numpy(global_p).to(device).expand(len(chunk), k)
            else:
                index = np.where(lval["opp_active"][chunk] >= 0, lval["opp_active"][chunk], 0)
                table = conditional[index]
                table[lval["opp_active"][chunk] < 0] = global_p
                prior = torch.from_numpy(table).to(device)
            probs = prior.float().masked_fill(~mask, 0)
            probs = probs / probs.sum(-1, keepdim=True)
            labels = torch.from_numpy(lval["opp_class"][chunk]).long().to(device)
            tera_rows = lval["opp_tera"][chunk] >= 0
            metrics.update(probs, labels,
                           torch.full((int(tera_rows.sum()),), tera_rate, device=device),
                           torch.from_numpy(lval["opp_tera"][chunk][tera_rows]).to(device))
        out[name] = metrics.result()
    return out


def save(model, vocab, arm, extra):
    directory = Path("checkpoints/w6a") / arm
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, directory / "best.pt")
    shutil.copy(C_DIR / "tokenizer.json", directory / "tokenizer.json")
    recorded = yaml.safe_load((C_DIR / "config.yaml").read_text())
    recorded["model"] = vars(model.config)
    recorded |= {"policy_version": f"w6a-{arm}", "init_checkpoint": str(C_DIR / "best.pt"),
                 "opponent_vocab": vocab, **extra}
    (directory / "config.yaml").write_text(yaml.safe_dump(recorded, sort_keys=False))
    return str(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-epochs", type=int, default=8)
    parser.add_argument("--joint-epochs", type=int, default=3)
    parser.add_argument("--report", default="reports/w6a_opp_head.json")
    args = parser.parse_args()
    started = time.monotonic()
    seed_everything(SEED)
    device = torch.device("cuda")
    base, _, _ = load_pokedecision(C_DIR, device)
    train, val = load_split("train", "datasets/w4"), load_split("val", "datasets/w4")
    ltrain, vocab = load_labels("train", train)
    lval, _ = load_labels("val", val)
    otrain = load_outcomes("datasets/w4", "train", train)
    oval = load_outcomes("datasets/w4", "val", val)
    halves = np.array([val_half(str(b)) for b in val["battle_ids"]])
    idx_a, idx_b = np.flatnonzero(halves == "A").tolist(), np.flatnonzero(halves == "B").tolist()
    rows_b = np.concatenate([np.arange(val["offsets"][t], val["offsets"][t + 1]) for t in idx_b])
    always = move_class_mask(vocab).to(device)
    switch_classes = torch.tensor([c.startswith("switch:") for c in vocab], device=device)
    val_a = loader(val, lval, oval, idx_a)
    val_b = loader(val, lval, oval, idx_b)
    report = {"classes": len(vocab), "seed": SEED, "lambda": LAMBDA,
              "value_weight": VALUE_WEIGHT, "policy_slack": POLICY_SLACK,
              "val_trajectories": {"A": len(idx_a), "B": len(idx_b)}, "arms": {}}

    c_a = evaluate(base, val_a, device, always, switch_classes)
    report["C"] = {"val_A": c_a, "val_B": evaluate(base, val_b, device, always, switch_classes)}
    report["baselines_val_B"] = baselines(ltrain, lval, np.asarray(rows_b), vocab, device,
                                          always, switch_classes)
    print(json.dumps({"C": report["C"], "baselines": report["baselines_val_B"]}), flush=True)

    # frozen arm
    t0 = time.monotonic()
    frozen = with_head(base, len(vocab), device)
    for name, parameter in frozen.named_parameters():
        parameter.requires_grad_(name.startswith("opponent_head."))
    optimizer = torch.optim.AdamW(frozen.opponent_head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_nll, epochs = None, float("inf"), []
    for epoch in range(1, args.frozen_epochs + 1):
        objective = train_epoch(frozen, loader(train, ltrain, otrain, shuffle=True, seed=SEED + epoch),
                                device, optimizer, always, opp_weight=None)
        result = evaluate(frozen, val_a, device, always, switch_classes)
        epochs.append({"epoch": epoch, "train_objective": objective, "val_A": result})
        print(json.dumps({"frozen": epochs[-1]}), flush=True)
        if result["opponent"]["nll"] < best_nll:
            best_nll, best = result["opponent"]["nll"], copy.deepcopy(frozen.state_dict())
        else:
            break
    frozen.load_state_dict(best)
    trunk = {k: v for k, v in frozen.state_dict().items() if not k.startswith("opponent_head.")}
    for name, value in base.state_dict().items():
        if not torch.equal(value, trunk[name]):
            raise AssertionError(f"Frozen arm changed C parameter {name}")
    report["arms"]["frozen"] = {
        "epochs": epochs, "seconds": time.monotonic() - t0,
        "val_B": evaluate(frozen, val_b, device, always, switch_classes),
        "checkpoint": save(frozen, vocab, "frozen", {"arm": "frozen"}),
    }

    # Fine-tune arms from C + the frozen head, fixed epoch budget (no epoch selection):
    # joint = BC + value + LAMBDA·opp; control = the same without the opponent loss, so
    # joint − control isolates the auxiliary task from extra BC/value training.
    limit = c_a["policy_loss"] + POLICY_SLACK
    for arm, weight, offset in (("joint", LAMBDA, 100), ("control", 0.0, 100)):
        t0 = time.monotonic()
        model = copy.deepcopy(frozen)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        epochs = []
        for epoch in range(1, args.joint_epochs + 1):
            objective = train_epoch(
                model, loader(train, ltrain, otrain, shuffle=True, seed=SEED + offset + epoch),
                device, optimizer, always, weight)
            result = evaluate(model, val_a, device, always, switch_classes)
            epochs.append({"epoch": epoch, "train_objective": objective, "val_A": result})
            print(json.dumps({arm: epochs[-1]}), flush=True)
        eligible = epochs[-1]["val_A"]["policy_loss"] <= limit
        report["arms"][arm] = {
            "epochs": epochs, "seconds": time.monotonic() - t0, "opp_weight": weight,
            "policy_loss_limit_val_A": limit, "eligible": eligible,
            "val_B": evaluate(model, val_b, device, always, switch_classes),
        }
        if eligible:
            report["arms"][arm]["checkpoint"] = save(
                model, vocab, arm, {"arm": arm, "opp_weight": weight})
    report["total_wall_time_s"] = time.monotonic() - started
    report["gpu"] = torch.cuda.get_device_name(0)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": args.report, "wall_s": report["total_wall_time_s"]}), flush=True)


if __name__ == "__main__":
    main()
