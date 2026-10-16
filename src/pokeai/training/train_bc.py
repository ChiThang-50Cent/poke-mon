"""Masked behavior cloning on battle-grouped human replay sequences.

Each batch contains complete trajectories, padded only to its longest member.
Hidden state starts at zero at trajectory start and runs uninterrupted, matching
online serving. An explicit length mask excludes all padding. Validation uses
the same uninterrupted histories. MLP and GRU get the same seed, batches,
optimizer, maximum epochs, and early-stopping patience.
"""

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pokeai.models.pokedecision import (
    PokeDecision,
    PokeDecisionConfig,
    masked_logits,
    save_checkpoint,
)
from pokeai.repro import source_fingerprint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def eligible_steps(actions: Tensor, masks: Tensor, valid: Tensor) -> Tensor:
    eligible = valid.bool() & (actions >= 0) & (actions < 13) & masks.any(-1)
    return eligible & masks.gather(-1, actions.clamp(0, 12).unsqueeze(-1)).squeeze(-1)


def masked_bc_loss(
    logits: Tensor, actions: Tensor, masks: Tensor, valid: Tensor
) -> Tensor | None:
    eligible = eligible_steps(actions, masks, valid)
    if not eligible.any():
        return None
    return F.cross_entropy(
        masked_logits(logits[eligible], masks[eligible]), actions[eligible]
    )


class TrajectorySequences(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.windows = list(zip(arrays["offsets"][:-1], arrays["offsets"][1:]))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start, end = self.windows[index]
        result = {
            key: torch.from_numpy(self.arrays[key][start:end])
            for key in ("tokens", "numbers", "masks", "actions", "valid")
        }
        if "outcomes" in self.arrays:
            result["outcomes"] = torch.full(
                (end - start,), float(self.arrays["outcomes"][index])
            )
            result["real"] = torch.ones(end - start, dtype=torch.bool)
        return result


def pad_sequences(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    lengths = torch.tensor([len(item["actions"]) for item in batch])
    result = {
        key: torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True,
            padding_value=-1 if key == "actions" else 0,
        )
        for key in batch[0]
    }
    result["valid"] &= torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
    return result


def baselines(train: dict, val: dict) -> dict:
    counts = np.bincount(train["actions"][train["valid"]], minlength=13) + 1.0
    masks = val["masks"][val["valid"]]
    actions = val["actions"][val["valid"]]
    prior = masks * counts[None, :]
    prior /= prior.sum(axis=-1, keepdims=True)
    return {
        "uniform_legal": {"val_loss": float(np.log(masks.sum(-1)).mean())},
        "train_frequency_legal": {
            "val_loss": float(-np.log(prior[np.arange(len(actions)), actions]).mean()),
            "train_counts_laplace": counts.tolist(),
        },
    }


def value_loss(logits: Tensor, outcomes: Tensor, real: Tensor) -> Tensor:
    """Every observed step contributes, irrespective of BC eligibility."""
    return F.binary_cross_entropy_with_logits(
        logits[real.bool()], outcomes[real.bool()]
    )


def run_epoch(
    model: PokeDecision, loader: DataLoader, device: torch.device,
    optimizer: torch.optim.Optimizer | None, grad_clip: float,
    value_weight: float = 0.5, probe: bool = False,
) -> dict[str, float | int]:
    model.train(optimizer is not None)
    total_loss = correct = nonforced_correct = count = nonforced = 0
    total_value = real_count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            if model.config.value_head:
                logits, values, _ = model.forward_with_value(
                    batch["tokens"].long(), batch["numbers"]
                )
                auxiliary = value_loss(values, batch["outcomes"], batch["real"])
                steps = int(batch["real"].sum())
                total_value += float(auxiliary.detach()) * steps
                real_count += steps
            else:
                logits, _ = model(batch["tokens"].long(), batch["numbers"])
                auxiliary = None
            valid = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
            loss = masked_bc_loss(logits, batch["actions"], batch["masks"], valid)
            objective = (
                auxiliary if probe else
                (loss if auxiliary is None else
                 (0 if loss is None else loss) + value_weight * auxiliary)
            )
            if objective is None:
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        n = int(valid.sum())
        total_loss += float(loss.detach()) * n if loss is not None else 0
        predictions = masked_logits(logits[valid], batch["masks"][valid]).argmax(-1)
        matches = predictions == batch["actions"][valid]
        choices = batch["masks"][valid].sum(-1) > 1
        count += n
        correct += int(matches.sum())
        nonforced += int(choices.sum())
        nonforced_correct += int((matches & choices).sum())
    if not count:
        raise ValueError("No loss-eligible decisions in split")
    result = {"loss": total_loss / count, "accuracy": correct / count,
              "accuracy_nonforced": nonforced_correct / max(nonforced, 1),
              "valid_decisions": count, "nonforced_decisions": nonforced}
    if real_count:
        result["policy_loss"] = result["loss"]
        result["value_bce"] = total_value / real_count
        result["value_steps"] = real_count
        result["loss"] = (
            result["value_bce"] if probe else
            result["policy_loss"] + value_weight * result["value_bce"]
        )
    return result


def initialize_model(config: dict, arch: str, stats: dict, encoder: dict) -> PokeDecision:
    """Initialize scratch statistics or restore the entire pretrained state."""
    model_config = PokeDecisionConfig(
        **(config["model"] | {"arch": arch, "vocab_size": encoder["vocab_size"],
                              "token_len": encoder["token_len"]})
    )
    model = PokeDecision(model_config, stats["mean"], stats["std"])
    checkpoint_path = config.get("init_checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_config = PokeDecisionConfig(**checkpoint["config"])
        probe = config.get("value_mode") == "probe"
        expected_config = asdict(model_config) | ({"value_head": False} if probe else {})
        if asdict(checkpoint_config) != expected_config:
            raise ValueError("Initialization checkpoint architecture mismatch")
        missing, unexpected = model.load_state_dict(
            checkpoint["state_dict"], strict=not probe
        )
        if probe:
            if set(missing) != {"value_head.weight", "value_head.bias"} or unexpected:
                raise ValueError("Probe initialization must differ only by value head")
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("value_head."))
    parameters = sum(p.numel() for p in model.parameters())
    expected = config.get("expected_parameters")
    if expected is not None and parameters != expected:
        raise ValueError(f"Expected {expected} parameters, got {parameters}")
    if not 1_000_000 <= parameters <= 3_000_000:
        raise ValueError(f"Parameter budget violated: {parameters}")
    return model


def checkpoint_metadata(config: dict, tokenizer: Path) -> dict:
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    expected = config.get("tokenizer_sha256")
    if expected is not None and digest != expected:
        raise ValueError(f"Tokenizer hash mismatch: {digest} != {expected}")
    init = config.get("init_checkpoint")
    if init:
        pretrained_tokenizer = Path(init).parent / "tokenizer.json"
        if hashlib.sha256(pretrained_tokenizer.read_bytes()).hexdigest() != digest:
            raise ValueError("Initialization tokenizer hash mismatch")
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()),
        "dataset_version": config.get("dataset_version", "w3"),
        "train_tier": config.get("train_tier", "full"),
        "tokenizer_sha256": digest,
        "observation_space_version": "DefaultObservationSpace@metamon-0a00a75",
        "init_checkpoint": str(init) if init else None,
        "init_checkpoint_sha256": (
            hashlib.sha256(Path(init).read_bytes()).hexdigest() if init else None
        ),
    }


def train_arch(
    config: dict[str, Any], arch: str, train: dict, val: dict,
    stats: dict, encoder: dict, output: Path,
) -> dict:
    seed_everything(config["seed"])
    device = torch.device(config["train"]["device"])
    metadata = checkpoint_metadata(
        config, Path(config["data_root"]) / "encoder.json"
    )
    model = initialize_model(config, arch, stats, encoder).to(device)
    model_config = model.config
    params = sum(p.numel() for p in model.parameters())
    settings = config["train"]
    train_loader = DataLoader(
        TrajectorySequences(train),
        batch_size=settings["batch_size"], shuffle=True, collate_fn=pad_sequences,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    val_loader = DataLoader(
        TrajectorySequences(val),
        batch_size=settings["batch_size"], collate_fn=pad_sequences,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(config["data_root"]) / "encoder.json", output / "tokenizer.json"
    )
    recorded = config | metadata | {
        "model": asdict(model_config), "source_sha256": source_fingerprint(),
        "metamon_commit": "0a00a75", "python_version": platform.python_version(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "sequence_policy": "full trajectories; reset hidden only at trajectory start",
        "feature_stats": {
            "mean": model.feature_mean.detach().cpu().tolist(),
            "std": model.feature_std.detach().cpu().tolist(),
        },
    }
    (output / "config.yaml").write_text(yaml.safe_dump(recorded, sort_keys=False))
    best, stale, history = float("inf"), 0, []
    started = time.monotonic()
    best_epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        training = run_epoch(
            model, train_loader, device, optimizer, settings["grad_clip"],
            config.get("value_weight", 0.5), config.get("value_mode") == "probe",
        )
        validation = run_epoch(
            model, val_loader, device, None, settings["grad_clip"],
            config.get("value_weight", 0.5), config.get("value_mode") == "probe",
        )
        row = {"epoch": epoch, "train": training, "val": validation,
               "elapsed_s": time.monotonic() - started}
        history.append(row)
        print(json.dumps({"arch": arch, **row}), flush=True)
        if validation["loss"] < best:
            best, best_epoch, stale = validation["loss"], epoch, 0
            save_checkpoint(model, output / "best.pt")
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    result = {
        "arch": arch, "parameters": params, "best_epoch": best_epoch,
        "stopping_epoch": history[-1]["epoch"],
        "train_valid_decisions": history[0]["train"]["valid_decisions"],
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "best_val": history[best_epoch - 1]["val"], "epochs": history,
        "wall_time_s": time.monotonic() - started,
        "source_sha256": recorded["source_sha256"],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pokedecision-v0.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    # CUDA deterministic GEMM requires this setting before the first CUDA context.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(config["train"].get("cpu_threads", 4), 4))
    from pokeai.data.w3 import load_split
    root = Path(config["data_root"])
    if config.get("dataset_version") == "w4":
        from pokeai.data.w4 import load_train_tier
        train = load_train_tier(config.get("train_tier", "full"), root)
    else:
        train = load_split("train", root)
    val = load_split("val", root)
    stats = json.loads((root / "feature_stats.json").read_text())
    encoder = json.loads((root / "encoder.json").read_text())
    baseline_metrics = baselines(train, val)
    output = Path(config["output_dir"])
    architectures = config.get("architectures", ["gru", "mlp"])
    results = {}
    for index, arch in enumerate(architectures):
        destination = output if index == 0 else output.with_name(output.name + "-" + arch)
        results[arch] = train_arch(config, arch, train, val, stats, encoder, destination)
    if architectures != ["gru", "mlp"]:
        report = {"runs": results, "baselines": baseline_metrics}
        Path(config["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(config["report_path"]).write_text(json.dumps(report, indent=2) + "\n")
        return
    primary, comparison = results["gru"], results["mlp"]
    report = {
        "policy_version": "pokedecision-v0", "baselines": baseline_metrics,
        "gru": primary, "mlp": comparison,
        "beats_both_baselines": all(
            primary["best_val"]["loss"] < metric["val_loss"]
            for metric in baseline_metrics.values()
        ),
        "gru_minus_mlp_val_loss": (
            primary["best_val"]["loss"] - comparison["best_val"]["loss"]
        ),
        "total_wall_time_s": primary["wall_time_s"] + comparison["wall_time_s"],
    }
    Path(config["report_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["report_path"]).write_text(json.dumps(report, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": config["report_path"],
                      "beats_both_baselines": report["beats_both_baselines"]}))


if __name__ == "__main__":
    main()
