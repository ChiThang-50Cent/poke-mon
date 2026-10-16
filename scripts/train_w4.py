"""Run the fixed W4.2 grid sequentially and select on W4 validation only."""

import argparse
import copy
import gc
import json
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pokeai.data.w3 import load_split
from pokeai.data.w4 import load_train_tier
from pokeai.models.pokedecision import load_pokedecision
from pokeai.training.train_bc import (
    TrajectorySequences,
    baselines,
    checkpoint_metadata,
    pad_sequences,
    run_epoch,
    train_arch,
)


def evaluate(model, arrays, config):
    loader = DataLoader(
        TrajectorySequences(arrays), batch_size=config["train"]["batch_size"],
        collate_fn=pad_sequences,
    )
    return run_epoch(model, loader, torch.device(config["train"]["device"]), None, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pokedecision-v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(min(config["train"].get("cpu_threads", 4), 4))
    started = time.monotonic()
    root = Path(config["data_root"])
    checkpoint_metadata(config, root / "encoder.json")
    stats = json.loads((root / "feature_stats.json").read_text())
    encoder = json.loads((root / "encoder.json").read_text())
    val = load_split("val", root)
    w3_val = load_split("val", Path("datasets/w3"))
    full_train = load_train_tier("full", root)
    references = baselines(full_train, val)
    del full_train
    gc.collect()
    frozen, _, _ = load_pokedecision("models/pokedecision-v0", "cuda")
    references["v0_frozen"] = evaluate(frozen, val, config)
    del frozen
    runs = []
    for tier in config["tiers"]:
        train = load_train_tier(tier, root)
        for arm, settings in config["arms"].items():
            run_config = copy.deepcopy(config)
            run_config.update({"train_tier": tier, "arm": arm,
                               "init_checkpoint": settings["init_checkpoint"]})
            run_config["train"]["learning_rate"] = settings["learning_rate"]
            destination = Path(config["output_dir"]) / f"{arm}-{tier}"
            run_config["output_dir"] = str(destination)
            result = train_arch(
                run_config, "gru", train, val, stats, encoder, destination
            )
            model, _, _ = load_pokedecision(destination, "cuda")
            diagnostic = evaluate(model, w3_val, config)
            result.update({"arm": arm, "train_tier": tier,
                           "w3_val_loss_diagnostic": diagnostic["loss"],
                           "checkpoint_dir": str(destination)})
            (destination / "metrics.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            runs.append(result)
            del model
            torch.cuda.empty_cache()
        del train
        gc.collect()
    winner = min(runs, key=lambda run: run["best_val"]["loss"])
    selected = Path(config["selected_dir"])
    selected.mkdir(parents=True, exist_ok=True)
    source = Path(winner["checkpoint_dir"])
    for name in ("best.pt", "config.yaml", "tokenizer.json", "metrics.json"):
        shutil.copyfile(source / name, selected / name)
    reason = "Lowest W4 validation loss among the eight fixed runs; no test evaluation"
    metadata = yaml.safe_load((selected / "config.yaml").read_text())
    metadata.update({"selected_from": str(source), "selection_reason": reason})
    (selected / "config.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    selected_metrics = winner | {"selected_from": str(source), "selection_reason": reason}
    (selected / "metrics.json").write_text(json.dumps(selected_metrics, indent=2) + "\n")
    curve = [
        {"train_tier": row["train_tier"], "arm": row["arm"],
         "train_valid_decisions": row["train_valid_decisions"],
         "best_epoch": row["best_epoch"], "stopping_epoch": row["stopping_epoch"],
         "w4_val_loss": row["best_val"]["loss"],
         "w4_val_accuracy": row["best_val"]["accuracy"],
         "w4_val_accuracy_nonforced": row["best_val"]["accuracy_nonforced"],
         "w3_val_loss_diagnostic": row["w3_val_loss_diagnostic"],
         "wall_time_s": row["wall_time_s"]}
        for row in runs
    ]
    report = {"policy_version": "pokedecision-v1", "runs": runs,
              "references": references, "learning_curve": curve,
              "selected_from": str(source), "selection_reason": reason,
              "total_wall_time_s": time.monotonic() - started,
              "gpu": torch.cuda.get_device_name(),
              "test_splits_loaded": False}
    path = Path(config["report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_from": str(source), "report": str(path)}), flush=True)


if __name__ == "__main__":
    main()
