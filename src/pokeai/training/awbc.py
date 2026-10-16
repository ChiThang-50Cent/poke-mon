"""W5.3 frozen-advantage BC on complete behavior-logged C trajectories.

Advantages and reference distributions are materialized before actor updates.
The critic never consumes the updated actor's features. Weights self-normalize
over eligible decisions in each batch; KL is D_KL(C || actor).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from metamon.interface import UniversalState
from torch.utils.data import DataLoader

from pokeai.data.dataset import resolve_run_path
from pokeai.data.selfplay_validate import replay_parity
from pokeai.data.validate import read_native
from pokeai.models.pokedecision import load_pokedecision, save_checkpoint
from pokeai.repro import source_fingerprint
from pokeai.training.critic import include_decision, validate_outcome
from pokeai.training.critic_metrics import PHASES
from pokeai.training.train_bc import (
    TrajectorySequences,
    eligible_steps,
    pad_sequences,
    seed_everything,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def awbc_weights(advantage, beta, maximum=5.0):
    if beta <= 0 or maximum <= 0:
        raise ValueError("beta and maximum must be positive")
    return torch.exp(torch.clamp(advantage.detach() / beta, max=np.log(maximum)))


def frozen_advantages(outcomes, values):
    return (torch.as_tensor(outcomes) - torch.as_tensor(values)).detach().clone()


def awbc_loss(logits, actions, masks, valid, weights, reference, kl_coefficient=0.1):
    selected = eligible_steps(actions, masks, valid)
    if not selected.any():
        return None
    mask = masks[selected].bool()
    logp = logits[selected].masked_fill(~mask, -torch.inf).log_softmax(-1)
    # Avoid 0 * -inf both on illegal slots and legal zero-probability slots.
    safe_logp = logp.masked_fill(~mask, 0)
    p = reference[selected].detach()
    kl = (torch.special.xlogy(p, p) - p * safe_logp).sum(-1).mean()
    nll = -logp.gather(-1, actions[selected].unsqueeze(-1)).squeeze(-1)
    selected_weights = weights[selected].detach()
    bc = (selected_weights / selected_weights.mean() * nll).mean()
    return bc + kl_coefficient * kl, bc, kl


def phase(index):
    return next(name for name, lower, upper in PHASES
                if index >= lower and (upper is None or index < upper))


def shuffle_advantages(advantages, opponents, phases, splits, seed):
    """Permute inside opponent/phase/split; never move a val label into train."""
    result = np.array(advantages, copy=True)
    groups = defaultdict(list)
    for i, group in enumerate(zip(opponents, phases, splits)):
        groups[group].append(i)
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        result[indices] = result[rng.permutation(indices)]
    return result


def select_median_seed(scores):
    """Deterministic middle seed by (pooled dev score, seed); no test input."""
    if len(scores) != 3:
        raise ValueError("Median selection requires exactly three training seeds")
    return sorted(scores, key=lambda seed: (scores[seed], int(seed)))[1]


def distribution(values):
    x = np.asarray(values, dtype=float)
    return {"count": len(x), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "quantiles": dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                  np.quantile(x, [.05, .25, .5, .75, .95]).tolist()))}


def weight_stats(advantage, beta):
    weights = awbc_weights(torch.as_tensor(advantage), beta).numpy()
    return {"advantage": distribution(advantage), "weights": distribution(weights),
            "ESS_absolute": float(weights.sum() ** 2 / (weights ** 2).sum()),
            "ESS": float(weights.sum() ** 2 / (weights ** 2).sum() / len(weights)),
            "clipped_fraction": float((np.asarray(advantage) / beta >= np.log(5)).mean())}


class AdvantageSequences(TrajectorySequences):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        start, end = self.windows[index]
        for key in ("weights", "reference"):
            result[key] = torch.from_numpy(self.arrays[key][start:end])
        return result


def prepare_dataset(run, model_dir, critic_dir, output, reports, contract_file=None):
    run, model_dir, critic_dir = map(Path, (run, model_dir, critic_dir))
    cache_path = critic_dir / "feature_cache.pt"
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    data, metadata = cached["data"], cached["metadata"]
    if metadata["source_run"] != str(run):
        raise ValueError("W5.2 feature cache source run mismatch")
    c_hash = digest(model_dir / "best.pt")
    for name, expected in metadata["c_hashes"].items():
        if digest(model_dir / name) != expected:
            raise ValueError(f"Frozen C hash mismatch: {name}")
    split_report = json.loads(Path("reports/w5_2_dataset_validation.json").read_text())
    if metadata["splits"] != split_report["splits"]:
        raise ValueError("W5.2 split mismatch")
    split_hash = hashlib.sha256(json.dumps(metadata["splits"], sort_keys=True).encode()).hexdigest()
    if split_hash != metadata["split_ids_sha256"] or split_hash != split_report["split_ids_sha256"]:
        raise ValueError("W5.2 split hash mismatch")
    if metadata["overall_status"] != "PASS":
        raise ValueError("W5.2 dataset QA not PASS")
    model, encoder, _ = load_pokedecision(model_dir, "cuda")
    model.requires_grad_(False).eval()
    head = torch.nn.Linear(data["h"].shape[-1], 1)
    head.load_state_dict(torch.load(critic_dir / "value.pt", map_location="cpu", weights_only=True)["state_dict"])
    with torch.inference_mode():
        values = head(data["h"].float()).sigmoid().squeeze(-1).numpy()
        probe = model.value_head(data["h"].to("cuda")).sigmoid().squeeze(-1).cpu().numpy()
    if not np.allclose(probe, data["probe"], atol=2e-5, rtol=1e-5):
        raise ValueError("Cached Probe does not match frozen C hidden states")
    advantages = {"E2": frozen_advantages(data["y"], values).numpy(),
                  "E3": frozen_advantages(data["y"], probe).numpy()}
    manifest = json.loads((run / "manifest.json").read_text())
    battles = {b["battle_id"]: b for b in map(json.loads, (run / "battles.jsonl").read_text().splitlines())}
    battle_indices = {b: i for i, b in enumerate(sorted(battles))}
    sequences, records = [], []
    cursor = 0
    max_difference = 0.0
    for entry in manifest["trajectory_files"]:
        meta = json.loads(resolve_run_path(run, entry["meta"]).read_text())
        y = validate_outcome(meta, battles[entry["battle_id"]])
        if meta.get("checkpoint_sha256") != c_hash or y is None:
            continue
        native = read_native(resolve_run_path(run, entry["file"]))
        parity = replay_parity(native, meta, model, encoder, "cuda")
        if parity["errors"]:
            raise ValueError(f"Replay parity failed: {parity}")
        max_difference = max(max_difference, parity["max_probability_difference"])
        seq = defaultdict(list)
        split = "train" if entry["battle_id"] in metadata["splits"]["train"] else "val"
        for i, (state, behavior) in enumerate(zip(native["states"][:-1], meta["behavior"])):
            if behavior["fallback_reason"]:
                encoded = encoder.encode(UniversalState.from_dict(copy.deepcopy(state)))
                for key in ("tokens", "numbers"):
                    seq[key].append(encoded[key])
                seq["masks"].append([False] * 13)
                seq["actions"].append(-1)
                seq["valid"].append(False)
                seq["reference"].append([0.] * 13)
                # Retained and observed by the actor, with no loss.
                seq["retry"].append(True)
                seq["row"].append(-1)
                continue
            record = dict(behavior)
            record.setdefault("checkpoint_sha256", meta.get("checkpoint_sha256"))
            if not include_decision(record, c_hash):
                raise ValueError("Non-C decision in C trajectory")
            if cursor >= len(data["y"]):
                raise ValueError("More replay decisions than W5.2 cache")
            checks = {"battle": battle_indices[entry["battle_id"]], "phase_index": i,
                      "actions": behavior["chosen_index"], "y": y}
            if any(data[k][cursor] != v for k, v in checks.items()):
                raise ValueError(f"W5.2 replay row alignment failed: {checks}")
            if not np.array_equal(data["masks"][cursor], behavior["mask"]) or not np.allclose(data["probs"][cursor], behavior["probabilities"], atol=1e-7):
                raise ValueError("W5.2 reference distribution alignment failed")
            encoded = encoder.encode(UniversalState.from_dict(copy.deepcopy(state)))
            for key in ("tokens", "numbers"):
                seq[key].append(encoded[key])
            seq["masks"].append(behavior["mask"])
            seq["actions"].append(behavior["chosen_index"])
            seq["valid"].append(True)
            seq["reference"].append(behavior["probabilities"])
            seq["retry"].append(bool(behavior.get("protocol_retry", False)))
            seq["row"].append(cursor)
            split = data["split"][cursor]
            records.append({"key": [entry["battle_id"], entry["player_side"], i],
                            "split": str(split), "opponent": str(data["opponent"][cursor]),
                            "phase": phase(i), "action_type": "move" if checks["actions"] < 4 else "switch" if checks["actions"] < 9 else "tera",
                            "Y": y, "V": float(values[cursor]), "Probe": float(probe[cursor]),
                            "A_V": float(advantages["E2"][cursor]), "A_Probe": float(advantages["E3"][cursor])})
            cursor += 1
        if seq:
            result = {k: np.asarray(v) for k, v in seq.items()}
            result["split"] = str(split)
            result["masks"] = result["masks"].astype(bool)
            result["numbers"] = result["numbers"].astype(np.float32)
            result["reference"] = result["reference"].astype(np.float32)
            sequences.append(result)
    if cursor != len(data["y"]) or len({tuple(r["key"]) for r in records}) != cursor:
        raise ValueError("Missing/duplicate per-decision cache alignment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "frozen_dataset.pt"
    provenance = {"C_sha256": c_hash, "C_hashes": metadata["c_hashes"],
                  "V_sha256": digest(critic_dir / "value.pt"),
                  "feature_cache_sha256": digest(cache_path), "split_ids_sha256": metadata["split_ids_sha256"],
                  "tokenizer_sha256": digest(model_dir / "tokenizer.json"), "source_run": str(run),
                  "memory_caveat": "Actor observes all states uninterrupted per contract. Frozen critic/reference match serving, which skips fallback updates and rolls back rejected protocol requests.",
                  "fallback_steps_in_actor_sequences": sum(int((s["row"] < 0).sum()) for s in sequences),
                  "protocol_retry_steps_in_actor_sequences": sum(int((s["retry"] & (s["row"] >= 0)).sum()) for s in sequences)}
    torch.save({"sequences": sequences, "advantages": advantages, "records": records,
                "provenance": provenance}, frozen_path)
    validation = {"overall_status": "PASS", "provenance": provenance, "splits": metadata["splits"],
                  "eligible_C_decisions": cursor, "unique_decision_keys": cursor,
                  "reference_probability_max_difference": max_difference,
                  "frozen_before_actor_updates": True, "full_trajectory_memory": True,
                  "actor_memory": "uninterrupted including non-model states",
                  "frozen_reference_memory": "serving parity (skip fallback; rollback retry)",
                  "dataset_sha256": digest(frozen_path), "decision_records": records,
                  "W5_2_exclusions": metadata["counts"]}
    if contract_file:
        validation["verbatim_contract"] = Path(contract_file).read_text()
        validation["contract_sha256"] = digest(contract_file)
    write_json(reports / "w5_3_dataset_validation.json", validation)
    stats = {"normalization": "weights divided by eligible batch mean",
             "critic_caveat": "W5.2 V fit on same train decisions; frozen Probe fit on human data",
             "phase_definition": "zero-based decision index; W5.2 buckets 0-5, 6-15, 16-30, 31+"}
    for exp, advantage in advantages.items():
        stats[exp] = {}
        for beta in (.5, 1., 2.):
            grouped = {}
            for column in ("split", "opponent", "phase", "action_type"):
                grouped[column] = {}
                for value in sorted({r[column] for r in records}):
                    indices = [i for i, r in enumerate(records) if r[column] == value]
                    grouped[column][value] = weight_stats(advantage[indices], beta)
            stats[exp][str(beta)] = {"all": weight_stats(advantage, beta), "grouped": grouped}
    write_json(reports / "w5_3_advantage_stats.json", stats)
    return frozen_path


def make_arrays(frozen, split, weights):
    seqs = [s for s in frozen["sequences"] if s["split"] == split]
    arrays = {k: np.concatenate([s[k] for s in seqs]) for k in
              ("tokens", "numbers", "masks", "actions", "valid", "reference")}
    arrays["weights"] = np.concatenate([np.where(s["row"] >= 0, weights[s["row"].clip(min=0)], 0) for s in seqs]).astype(np.float32)
    arrays["offsets"] = np.r_[0, np.cumsum([len(s["actions"]) for s in seqs])]
    return arrays


def forward_trajectory(model, batch):
    return model(batch["tokens"].long(), batch["numbers"])[0]


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    totals = defaultdict(float)
    all_kl = []
    frequency = {k: np.zeros(3) for k in ("greedy", "sampled", "reference_greedy", "reference_sampled")}
    count = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            logits = forward_trajectory(model, batch)
            losses = awbc_loss(logits, batch["actions"], batch["masks"], batch["valid"], batch["weights"], batch["reference"])
            if losses is None:
                continue
            loss, bc, kl = losses
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        selected = eligible_steps(batch["actions"], batch["masks"], batch["valid"])
        n = int(selected.sum())
        count += n
        probs = logits[selected].detach().masked_fill(~batch["masks"][selected], -torch.inf).softmax(-1)
        reference = batch["reference"][selected]
        logp = logits[selected].detach().masked_fill(~batch["masks"][selected], -torch.inf).log_softmax(-1)
        safe_logp = logp.masked_fill(~batch["masks"][selected], 0)
        decision_kl = (torch.special.xlogy(reference, reference) - reference * safe_logp).sum(-1)
        all_kl.extend(decision_kl.cpu().tolist())
        totals["unweighted_NLL"] += float(-logp.gather(-1, batch["actions"][selected].unsqueeze(-1)).sum())
        totals["greedy_action_agreement"] += int((probs.argmax(-1) == reference.argmax(-1)).sum())
        for key, value in (("loss", loss), ("weighted_bc", bc), ("KL_C_to_actor", kl)):
            totals[key] += float(value.detach()) * n
        totals["entropy"] += float(-torch.special.xlogy(probs, probs).sum())
        totals["reference_entropy"] += float(-torch.special.xlogy(reference, reference).sum())
        for prefix, p in (("", probs), ("reference_", reference)):
            action = p.argmax(-1)
            frequency[prefix + "greedy"] += np.array([int((action < 4).sum()), int(((action >= 4) & (action < 9)).sum()), int((action >= 9).sum())])
            frequency[prefix + "sampled"] += np.array([float(p[:, :4].sum()), float(p[:, 4:9].sum()), float(p[:, 9:].sum())])
    if not count:
        raise ValueError("No eligible decisions")
    result = {k: v / count for k, v in totals.items()}
    result["eligible_decisions"] = count
    for key, values in frequency.items():
        result[key + "_action_frequency"] = dict(zip(("move", "switch", "tera"), (values / count).tolist()))
    result["KL_C_to_actor_p95"] = float(np.quantile(all_kl, .95))
    result["weighted_NLL"] = result["weighted_bc"]
    entropy_change = abs(result["entropy"] - result["reference_entropy"]) / max(result["reference_entropy"], 1e-12)
    shifts = [abs(result["greedy_action_frequency"][k] - result["reference_greedy_action_frequency"][k]) for k in ("move", "switch", "tera")]
    result["drift_flags"] = {"KL": result["KL_C_to_actor"] > .5,
                             "entropy": entropy_change > .3, "action_frequency": max(shifts) > .1}
    result["abnormal"] = any(result["drift_flags"].values())
    return result


def train_actor(frozen_path, model_dir, output, reports, experiment, beta, seed, control=False, contract_file=None):
    started = time.perf_counter()
    seed_everything(seed)
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
    if frozen["provenance"]["C_sha256"] != digest(Path(model_dir) / "best.pt"):
        raise ValueError("Actor initialization differs from frozen dataset")
    records = frozen["records"]
    if experiment == "E1":
        if control:
            raise ValueError("Unweighted E1 has no advantage shuffle control")
        weights = np.ones(len(records), np.float32)
        beta = None
    else:
        advantage = frozen["advantages"][experiment].copy()
        if control:
            advantage = shuffle_advantages(advantage, [r["opponent"] for r in records],
                                           [r["phase"] for r in records], [r["split"] for r in records], seed)
        weights = awbc_weights(torch.as_tensor(advantage), beta).numpy()
    run_id = f"{experiment}{'-shuffled' if control else ''}-{str(float(beta)) if beta is not None else 'none'}-s{seed}"
    out = Path(output) / run_id
    if (out / "best.pt").exists():
        raise FileExistsError(f"Refusing to overwrite completed actor: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model, _, model_config = load_pokedecision(model_dir, "cuda")
    # A value head on an updated trunk would be stale: remove it entirely.
    model.value_head = None
    model.config.value_head = False
    loaders = {split: DataLoader(AdvantageSequences(make_arrays(frozen, split, weights)),
                                batch_size=16, shuffle=split == "train", collate_fn=pad_sequences,
                                generator=torch.Generator().manual_seed(seed)) for split in ("train", "val")}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    history = []
    for epoch in range(1, 6):
        training = run_epoch(model, loaders["train"], optimizer)
        with torch.no_grad():
            validation = run_epoch(model, loaders["val"])
        history.append({"epoch": epoch, "train": training, "val": validation})
    save_checkpoint(model, out / "best.pt")
    shutil.copyfile(Path(model_dir) / "tokenizer.json", out / "tokenizer.json")
    config = {"model": asdict(model_config), "experiment": experiment, "beta": beta, "seed": seed,
              "control": control, "run_id": run_id, "device": "cuda", "deterministic": True,
              "epochs": 5, "selected_epoch": 5, "selection": "fixed final epoch, no offline selection",
              "batch_size": 16, "lr": 1e-4, "weight_decay": 1e-4, "gradient_clip": 1.,
              "kl_coefficient": .1, "maximum_weight": 5., "optimizer": "AdamW",
              "loss_normalization": "weights divided by eligible batch mean; mean over eligible decisions",
              "team_preview": "unchanged pretrained C preview",
              "advantage_frozen": True, "value_head_removed": True,
              "provenance": frozen["provenance"], "frozen_dataset_sha256": digest(frozen_path),
              "source_fingerprint": source_fingerprint(), "checkpoint_sha256": digest(out / "best.pt")}
    config["wall_time_s"] = time.perf_counter() - started
    config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    if contract_file:
        config["verbatim_contract"] = Path(contract_file).read_text()
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(out / "training_manifest.json", config)
    write_json(out / "training_metrics.json", {"config": config, "history": history})
    drift = {}
    for split in ("train", "val"):
        with torch.no_grad():
            drift[split] = run_epoch(model, loaders[split])
    write_json(out / "policy_drift.json", drift)
    for filename, value in (("training_metrics", {"config": config, "history": history}), ("policy_drift", drift)):
        path = reports / f"w5_3_{filename}.json"
        existing = json.loads(path.read_text()) if path.exists() else {"runs": {}}
        existing["runs"][run_id] = value
        write_json(path, existing)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/w5-selfplay-pilot-20261016-final")
    parser.add_argument("--model", default="models/pokedecision-v1-value")
    parser.add_argument("--critic", default="models/pokedecision-w5-critic-pilot")
    parser.add_argument("--out", default="checkpoints/w5_3")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--contract-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--experiment", choices=("E1", "E2", "E3"))
    parser.add_argument("--beta", type=float, choices=(.5, 1., 2.))
    parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("W5.3 requires CUDA")
    torch.set_num_threads(1)
    seed_everything(1)
    args.reports.mkdir(parents=True, exist_ok=True)
    frozen_path = Path(args.out) / "frozen_dataset.pt"
    if args.prepare_only or not frozen_path.exists():
        frozen_path = prepare_dataset(args.run, args.model, args.critic, args.out, args.reports, args.contract_file)
    if args.prepare_only:
        return
    if args.control and (args.experiment not in ("E2", "E3") or args.beta is None):
        parser.error("--control requires the selected --experiment E2/E3 and --beta")
    for experiment in ([args.experiment] if args.experiment else ("E1", "E2", "E3")):
        betas = [None] if experiment == "E1" else ([args.beta] if args.beta else (.5, 1., 2.))
        for beta in betas:
            for seed in ([args.seed] if args.seed else (1, 2, 3)):
                train_actor(frozen_path, args.model, args.out, args.reports, experiment, beta, seed, args.control, args.contract_file)


if __name__ == "__main__":
    main()
