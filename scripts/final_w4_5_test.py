"""One-shot W4.5 holdout evaluation, run only after the fixed-arm battles.

Run under the caller's AGENTS.md CPU/memory scope; CUDA is required. This
script never trains, selects checkpoints, or changes the production probe.
The test-only outcome sidecar is gitignored and never replaces training labels.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from pokeai.data.w3 import load_split
from pokeai.data.w4_outcomes import build_outcomes, load_outcomes
from pokeai.models.pokedecision import load_pokedecision
from pokeai.training.train_bc import TrajectorySequences, pad_sequences, run_epoch
from pokeai.training.value_metrics import trajectory_metrics

MODELS = {
    "v0": "models/pokedecision-v0",
    "v1": "models/pokedecision-v1",
    "probe": "models/pokedecision-v1-value",
    "joint": "models/pokedecision-v1-joint-ablation",
}


class _PolicyOnly(torch.nn.Module):
    """Use the existing BC evaluator without requesting outcome labels."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(value_head=False)

    def forward(self, tokens, numbers):
        return self.model(tokens, numbers)


def _loader(arrays):
    return DataLoader(
        TrajectorySequences(arrays), batch_size=16, collate_fn=pad_sequences
    )


def _value_probabilities(model, arrays, device):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in _loader(arrays):
            _, logits, _ = model.forward_with_value(
                batch["tokens"].long().to(device), batch["numbers"].to(device)
            )
            predictions.append(logits.sigmoid().cpu().numpy()[batch["real"].numpy()])
    return np.concatenate(predictions)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_final_test(
    *, w4_root="datasets/w4", w3_root="datasets/w3",
    source_root=".cache/metamon/w4-human",
    output="reports/w4_5_test_results.json",
):
    """Evaluate the preregistered checkpoints once and return the JSON report.

    An exclusive claim file prevents accidental repeated holdout invocation,
    including after a partially failed run. No override/reselection is offered.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Holdout report already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("W4.5 final holdout evaluation requires CUDA")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    output.parent.mkdir(parents=True, exist_ok=True)
    claim = output.with_suffix(".started.json")
    with claim.open("x") as stream:
        json.dump({"purpose": "one-shot final holdout; no model selection"}, stream)
        stream.write("\n")
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "selection_performed": False,
        "value_used_for_action_selection": False,
        "policy": {}, "value": {},
        "checkpoints": {
            name: {"directory": directory,
                   "sha256": _sha256(Path(directory) / "best.pt")}
            for name, directory in MODELS.items()
        },
        "semantics": {
            "policy": "masked BC loss and accuracy on eligible decisions",
            "nonforced": "eligible decisions with more than one legal action",
            "value": "all pre-action decisions, including BC-ineligible steps",
            "sequence": "full trajectories; hidden state reset only at trajectory start",
            "joint": "ablation only; failed preregistered policy delta gate 0.0155 > 0.01",
        },
        "gpu": torch.cuda.get_device_name(),
    }
    for name, root in (("w4_test", w4_root), ("w3_test", w3_root)):
        arrays = load_split("test", root)
        report["policy"][name] = {}
        for model_name in ("v0", "v1", "joint"):
            model, _, _ = load_pokedecision(MODELS[model_name], device)
            metrics = run_epoch(_PolicyOnly(model), _loader(arrays), device, None, 1.0)
            report["policy"][name][model_name] = metrics | {"policy_loss": metrics["loss"]}
            del model
        if name == "w4_test":
            build_outcomes(w4_root, source_root, allow_holdout=True)
            arrays["outcomes"] = load_outcomes(
                w4_root, "test", arrays, allow_holdout=True
            )
            labels = np.repeat(arrays["outcomes"], np.diff(arrays["offsets"]))
            for model_name in ("probe", "joint"):
                model, _, _ = load_pokedecision(MODELS[model_name], device)
                report["value"][model_name] = trajectory_metrics(
                    _value_probabilities(model, arrays, device), labels, arrays["offsets"]
                )
                del model
            report["outcomes_file"] = str(Path(w4_root) / "outcomes_holdout.json")
        del arrays
        torch.cuda.empty_cache()
    report["total_wall_time_s"] = time.monotonic() - started
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w4-root", default="datasets/w4")
    parser.add_argument("--w3-root", default="datasets/w3")
    parser.add_argument("--source-root", default=".cache/metamon/w4-human")
    parser.add_argument("--output", default="reports/w4_5_test_results.json")
    args = parser.parse_args()
    run_final_test(**vars(args))


if __name__ == "__main__":
    main()
