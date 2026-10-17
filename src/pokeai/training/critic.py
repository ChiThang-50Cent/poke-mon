"""Frozen-C Monte Carlo critic pilot; no policy optimization or battle collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from pokeai.data.dataset import resolve_run_path
from pokeai.data.selfplay_validate import replay_parity
from pokeai.data.validate import read_native
from pokeai.models.pokedecision import load_pokedecision

SEED = 20261016


def chosen_action_loss(logits, actions, targets):
    return F.binary_cross_entropy_with_logits(
        logits.gather(1, actions.long()[:, None]).squeeze(1), targets.float()
    )


def include_decision(record, c_hash):
    return (
        record.get("checkpoint_sha256") == c_hash
        and record.get("behavior_probs", record.get("probabilities")) is not None
    )


def validate_outcome(meta, battle):
    winner = battle.get("winner")
    if winner not in {"player", "opponent", "tie"} or battle.get("error"):
        raise ValueError("Incomplete/error battle outcome")
    learner = meta.get("learner")
    if type(learner) is not bool:
        if battle.get("learner_pov") not in {"p1", "p2"}:
            raise ValueError("Missing learner POV")
        learner = meta.get("player_side") == battle["learner_pov"]
    elif battle.get("learner_pov") in {"p1", "p2"}:
        if learner != (meta.get("player_side") == battle["learner_pov"]):
            raise ValueError("Learner POV mismatch")
    if type(battle.get("player_challenges")) is bool:
        learner_side = "p1" if battle["player_challenges"] else "p2"
        if learner != (meta.get("player_side") == learner_side):
            raise ValueError("Learner/challenger POV mapping mismatch")
    if meta.get("team_file") is not None:
        expected_team = battle.get("player_team" if learner else "opponent_team")
        if meta["team_file"] != expected_team:
            raise ValueError("Learner/team POV mapping mismatch")
    expected = (
        "tie"
        if winner == "tie"
        else "win"
        if (winner == "player") == learner
        else "loss"
    )
    if str(meta.get("final_result")).lower() != expected:
        raise ValueError("POV/outcome mismatch")
    return None if expected == "tie" else float(expected == "win")


def _pair(battle):
    return tuple(
        sorted(
            (
                battle.get("player_team_sha256", battle["player_team"]),
                battle.get("opponent_team_sha256", battle["opponent_team"]),
            )
        )
    )


def split_pairs(battles, seed=SEED):
    groups = defaultdict(list)
    for battle in battles:
        groups[_pair(battle)].append(battle)
    strata = defaultdict(list)
    for pair, rows in sorted(groups.items()):
        strata[tuple(sorted({b.get("opponent_type", "unknown") for b in rows}))].append(
            pair
        )
    rng = np.random.default_rng(seed)
    val = set()
    for pairs in strata.values():
        order = rng.permutation(len(pairs))
        n = round(0.2 * len(pairs))
        if len(pairs) > 1:
            n = max(1, min(len(pairs) - 1, n))
        val.update(pairs[i] for i in order[:n])
    return {b["battle_id"]: "val" if _pair(b) in val else "train" for b in battles}


def shuffled_actions(actions, masks, probs, seed=SEED):
    rng = np.random.default_rng(seed)
    result = np.asarray(actions).copy()
    groups = defaultdict(list)
    for i, mask in enumerate(masks):
        groups[np.asarray(mask, dtype=bool).tobytes()].append(i)
    for indices in groups.values():
        if len(indices) > 1:
            result[indices] = result[rng.permutation(indices)]
        else:
            i = indices[0]
            p = np.where(masks[i], probs[i], 0).astype(float)
            result[i] = rng.choice(13, p=p / p.sum())
    return result


def build_dataset(run, model_dir, reports):
    run, model_dir = Path(run), Path(model_dir)
    qa = json.loads(Path("reports/w5_1_selfplay_validation.json").read_text())
    if qa["overall_status"] != "PASS":
        raise ValueError("W5.1 QA not PASS")
    freeze = json.loads((model_dir / "freeze.json").read_text())
    hashes = freeze["policy_model_sha256"]
    digest = hashlib.sha256((model_dir / "best.pt").read_bytes()).hexdigest()
    if digest != hashes["best.pt"]:
        raise ValueError("Frozen C checkpoint mismatch")
    manifest = json.loads((run / "manifest.json").read_text())
    battles = [json.loads(s) for s in (run / "battles.jsonl").read_text().splitlines()]
    by_id = {b["battle_id"]: b for b in battles}
    loaded = []
    outcomes = defaultdict(list)
    provenance, excluded = Counter(), Counter()
    for entry in manifest["trajectory_files"]:
        meta = json.loads(resolve_run_path(run, entry["meta"]).read_text())
        battle = by_id[entry["battle_id"]]
        if (meta["battle_id"], meta["player_side"]) != (
            entry["battle_id"],
            entry["player_side"],
        ):
            raise ValueError("Manifest POV mismatch")
        y = validate_outcome(meta, battle)
        outcomes[entry["battle_id"]].append((meta["player_side"], y, meta["learner"]))
        battle["opponent_type"] = meta["opponent_type"]
        for behavior in meta.get("behavior", []):
            if behavior is None:
                excluded["heuristic" if not meta.get("model_dir") else "fallback"] += 1
                provenance[
                    f"{meta.get('policy_version')}|{meta.get('checkpoint_sha256')}"
                ] += 1
                continue
            key = f"{behavior.get('policy_version', meta.get('policy_version'))}|{behavior.get('checkpoint_sha256', meta.get('checkpoint_sha256'))}"
            provenance[key] += 1
            if behavior.get("probabilities") is None:
                excluded["heuristic" if not meta.get("model_dir") else "fallback"] += 1
            elif (
                behavior.get("checkpoint_sha256", meta.get("checkpoint_sha256"))
                != digest
            ):
                excluded["off_policy"] += 1
            elif y is None:
                excluded["tie"] += 1
        loaded.append((entry, meta, y))
    for battle_id, rows in outcomes.items():
        if len(rows) != 2 or {r[0] for r in rows} != {"p1", "p2"}:
            raise ValueError(f"Missing/duplicate POV: {battle_id}")
        if sum(r[2] for r in rows) != 1:
            raise ValueError("Learner POV mapping mismatch")
        ys = [r[1] for r in rows]
        if ys != [None, None] and (None in ys or sum(ys) != 1):
            raise ValueError("Noncomplementary outcomes")
    split = split_pairs(battles)
    train_ids = sorted(b for b in split if split[b] == "train")
    rng = np.random.default_rng(SEED)
    inner = set(rng.choice(train_ids, size=round(0.1 * len(train_ids)), replace=False))
    pairs = {p: i for i, p in enumerate(sorted({_pair(b) for b in battles}))}
    battle_indices = {b: i for i, b in enumerate(sorted(by_id))}
    model, encoder, _ = load_pokedecision(model_dir, "cuda")
    model.eval().requires_grad_(False)
    features, rows = [], []
    max_diff = 0.0
    for entry, meta, y in loaded:
        native = read_native(resolve_run_path(run, entry["file"]))
        expected = str(meta["final_result"]).lower()
        terminal = native["states"][-1]
        if bool(terminal.get("battle_won")) != (expected == "win") or bool(
            terminal.get("battle_lost")
        ) != (expected == "loss"):
            raise ValueError("Native terminal outcome mismatch")
        if meta.get("checkpoint_sha256") != digest or y is None:
            continue
        local_rows = []

        def collect(i, behavior, hidden, meta=meta, local_rows=local_rows):
            nonlocal max_diff
            record = dict(behavior)
            record.setdefault("checkpoint_sha256", meta.get("checkpoint_sha256"))
            if not include_decision(record, digest):
                return
            h = hidden[-1, 0].detach().clone()
            mask = torch.as_tensor(behavior["mask"], device="cuda", dtype=torch.bool)
            pred = model.policy(h).masked_fill(~mask, -torch.inf).softmax(-1)
            diff = float(
                np.max(
                    np.abs(pred.cpu().numpy() - np.asarray(behavior["probabilities"]))
                )
            )
            max_diff = max(max_diff, diff)
            features.append(h)
            local_rows.append((i, behavior))

        parity = replay_parity(
            native, meta, model, encoder, "cuda", decision_sink=collect
        )
        if parity["errors"] or max_diff > 1e-5:
            raise ValueError(f"Feature alignment failed: {parity}")
        battle = by_id[entry["battle_id"]]
        for j, (i, behavior) in enumerate(local_rows):
            rows.append(
                {
                    "y": y,
                    "actions": behavior["chosen_index"],
                    "masks": behavior["mask"],
                    "probs": behavior["probabilities"],
                    "probe": behavior["value"],
                    "battle": battle_indices[battle["battle_id"]],
                    "pair": pairs[_pair(battle)],
                    "phase_index": i,
                    "first": j == 0,
                    "last": j == len(local_rows) - 1,
                    "opponent": meta["opponent_type"],
                    "split": split[battle["battle_id"]],
                    "inner_val": battle["battle_id"] in inner,
                }
            )
    h_gpu = torch.stack(features).clone()
    data = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    data["masks"] = data["masks"].astype(bool)
    data["h"] = h_gpu
    teams = {}
    for side in ("train", "val"):
        teams[side] = {
            t for b in battles if split[b["battle_id"]] == side for t in _pair(b)
        }
    metadata = {
        "source_run": str(run),
        "c_hashes": hashes,
        "qa": qa,
        "overall_status": "PASS",
        "pov_outcome_checks": "PASS",
        "learner_challenger_team_pov_mapping": "PASS",
        "two_pov_battles_checked": len(outcomes),
        "ties_battles": sum(b["winner"] == "tie" for b in battles),
        "feature_alignment_max_diff": max_diff,
        "feature_alignment_decisions": len(rows),
        "counts": {
            "included_C": len(rows),
            "by_policy_version_and_hash": dict(provenance),
            "excluded": dict(excluded),
        },
        "splits": {
            s: sorted(k for k, v in split.items() if v == s) for s in ("train", "val")
        },
        "split_seed": SEED,
        "inner_validation_battles": sorted(inner),
        "team_pair_overlap": 0,
        "team_overlap": len(teams["train"] & teams["val"]),
        "leakage": "Only ObservationEncoder inputs to frozen C GRU; terminal outcomes labels only.",
        "temperature_caveat": "Values estimate sampled C at T=1.0, not production greedy C.",
    }
    metadata["split_ids_sha256"] = hashlib.sha256(
        json.dumps(metadata["splits"], sort_keys=True).encode()
    ).hexdigest()
    metadata["sample_counts"] = {
        s: int((data["split"] == s).sum()) for s in ("train", "val")
    }
    (reports / "w5_2_dataset_validation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return data, metadata


def train_head(data, width, actions=None):
    torch.manual_seed(SEED)
    head = nn.Linear(384, width, device="cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    x = data["h"]
    y = torch.as_tensor(data["y"], device="cuda", dtype=torch.float32)
    a = torch.as_tensor(
        data["actions"] if actions is None else actions, device="cuda", dtype=torch.long
    )
    train = torch.as_tensor(
        np.flatnonzero((data["split"] == "train") & ~data["inner_val"]), device="cuda"
    )
    val = torch.as_tensor(np.flatnonzero(data["inner_val"]), device="cuda")
    best, stale, history, best_state = float("inf"), 0, [], None
    for epoch in range(200):
        head.train()
        order = train[torch.randperm(len(train), device="cuda")]
        for indices in order.split(4096):
            optimizer.zero_grad(set_to_none=True)
            logits = head(x[indices])
            loss = (
                chosen_action_loss(logits, a[indices], y[indices])
                if width == 13
                else F.binary_cross_entropy_with_logits(logits[:, 0], y[indices])
            )
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(x[val])
            loss = float(
                (
                    chosen_action_loss(logits, a[val], y[val])
                    if width == 13
                    else F.binary_cross_entropy_with_logits(logits[:, 0], y[val])
                ).item()
            )
        history.append(loss)
        if loss < best:
            best, stale = loss, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    head.load_state_dict(best_state)
    with torch.no_grad():
        predictions = head(x).sigmoid().cpu().numpy()
    return (
        head,
        predictions[:, 0] if width == 1 else predictions,
        {
            "epochs": epoch + 1,
            "best_epoch": int(np.argmin(history)) + 1,
            "inner_val_bce": best,
            "history": history,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/w5-selfplay-pilot-20261016-final")
    parser.add_argument("--model", default="models/pokedecision-v1-value")
    parser.add_argument("--out", default="models/pokedecision-w5-critic-pilot")
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Finalize reports from already-trained heads without retraining",
    )
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    reports, out = Path("reports"), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "feature_cache.pt"
    if args.evaluate_existing and cache_path.exists():
        cache = torch.load(cache_path, map_location="cuda", weights_only=False)
        data, metadata = cache["data"], cache["metadata"]
    else:
        data, metadata = build_dataset(args.run, args.model, reports)
        torch.save(
            {
                "data": {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                },
                "metadata": metadata,
            },
            cache_path,
        )
    predictions, histories = {}, {}
    shuffled = data["actions"].copy()
    train = data["split"] == "train"
    shuffled[train] = shuffled_actions(
        data["actions"][train], data["masks"][train], data["probs"][train]
    )
    for name, width, actions, filename in (
        ("v", 1, None, "value.pt"),
        ("q", 13, None, "q_value.pt"),
        ("shuffled", 13, shuffled, "q_shuffled.pt"),
    ):
        if args.evaluate_existing:
            checkpoint = torch.load(
                out / filename, map_location="cuda", weights_only=True
            )
            head = nn.Linear(384, width, device="cuda")
            head.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                prediction = head(data["h"]).sigmoid().cpu().numpy()
            predictions[name] = prediction[:, 0] if width == 1 else prediction
            histories[name] = checkpoint.get(
                "training",
                {
                    "status": "Recovered trained checkpoint; epoch history unavailable after report-finalization failure"
                },
            )
        else:
            head, predictions[name], histories[name] = train_head(data, width, actions)
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "input_size": 384,
                    "output_size": width,
                    "training": histories[name],
                },
                out / filename,
            )
    config = {
        "source_run": args.run,
        "C_hashes": metadata["c_hashes"],
        "split_seed": SEED,
        "split_ids_sha256": metadata["split_ids_sha256"],
        "sample_counts": metadata["sample_counts"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "target_definition": "terminal win=1 loss=0 acting POV; ties excluded; Monte Carlo; no shaping",
        "temperature_caveat": metadata["temperature_caveat"],
        "encoder_frozen": True,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 4096,
        "max_epochs": 200,
        "patience": 10,
        "device": "cuda",
        "seed": SEED,
        "training": histories,
        "shuffled_control": "Permute train actions within identical legal masks; singleton samples from behavior probabilities",
    }
    if args.evaluate_existing:
        config["execution_recovery"] = {
            "training_repeated": False,
            "feature_extraction_repeated_for_report_recovery": True,
        }
        config["execution_recovery"]["reason"] = (
            "Finalize evaluation from saved trained heads"
        )
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    data["h"] = data["h"].cpu().numpy()
    from pokeai.training.critic_metrics import evaluate

    evaluate(data, predictions, metadata, reports)
    config["wall_seconds"] = time.monotonic() - started
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        json.dumps(
            {
                "wall_seconds": config["wall_seconds"],
                "samples": metadata["sample_counts"],
                "feature_alignment_max_diff": metadata["feature_alignment_max_diff"],
                "training": {
                    k: {s: v for s, v in h.items() if s != "history"}
                    for k, h in histories.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
