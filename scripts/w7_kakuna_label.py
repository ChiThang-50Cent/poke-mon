"""W7: offline soft labels from Metamon's pretrained Kakuna on pile states (no simulator).

For every trajectory of ``<dataset>/<split>.npz`` (same order as `w6p_build_pile.py`: sorted
filenames of ``--root`` filtered by ``split_of(battle_id)``) Kakuna is run over ITS OWN
observation sequence of the same ``json.lz4`` file and its action distribution is written in our
13-slot action space.

API path (nothing in third_party is patched):
  get_pretrained_model("Kakuna") -> flex-attention override -> initialize_agent(temperature=1)
  ParsedReplayDataset(pretrained.observation_space / action_space / reward_function)
      .load_filename -> MetamonAMAGODataset._process_data -> amago RLData -> RLData_pad_collate
  policy.tstep_encoder -> policy.traj_encoder (full sequence, no KV cache)
  -> policy.actor.actor_network_forward (masked logits, last gamma) -> softmax.

Action space: Kakuna uses ``DefaultActionSpace`` = identity on UniversalAction (13 slots:
4 moves, 5 switches, 4 move+Tera), so no remapping is needed (asserted on every trajectory).

Context: Kakuna plays live with a KV cache holding the last ``max_seq_len`` (128) steps and a
sliding attention window of 96 (``q - kv <= 96``), time index = global turn counter. The flex
mask is the same window, so a full-sequence causal pass over the whole battle gives the same
result as the live cached pass for ANY length (keys older than 96 are masked either way, and
the 128-step cache never drops a key inside the window). Nothing is chunked or truncated.
``--check-live N`` re-runs the first N trajectories step by step with the real cached
inference path and reports the max |delta prob|.

Labels are the *unclipped* masked softmax at temperature 1. Live play additionally clamps each
probability to [0.001, 0.99] before renormalising (and, without masking, gives illegal slots the
0.001 floor); that is a sampling-exploration artefact and is deliberately not baked in.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("METAMON_CACHE_DIR", str(Path(".cache/metamon").resolve()))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent))

NAME = re.compile(r"^metamon-gen9ou-(\d+)_")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pad_to(length: int, multiple: int = 16) -> int:
    return -(-length // multiple) * multiple


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="pile dir with gen9ou/*.json.lz4")
    parser.add_argument("--dataset", required=True, help="datasets/w7_x (train.npz, val.npz)")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-trajectories", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3, help="replay loading processes")
    parser.add_argument("--check-live", type=int, default=0,
                        help="re-run first N trajectories with the cached live path")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--max-batch-steps", type=int, default=8192,
                        help="cap on padded steps (B x padded max length) per forward")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="trajectories per resumable shard (<out>.shards/)")
    parser.add_argument("--longest", type=int, default=0,
                        help="debug: label only the N longest trajectories (output not a prefix)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import torch
    # Per-item obs dicts hold many small tensors; fd passing runs out of ancdata mid-run.
    torch.multiprocessing.set_sharing_strategy("file_system")
    from amago.loading import RLData_pad_collate
    from amago.nets.transformer import FlexAttention
    from metamon.data import ParsedReplayDataset
    from metamon.rl.metamon_to_amago import MetamonAMAGODataset
    from metamon.rl.pretrained import get_pretrained_model

    import pokeai  # noqa: F401  (must precede metamon imports: GenData constructor)

    worker = _load_script("metamon_h2h_worker")
    pile = _load_script("w6p_build_pile")
    worker.use_flex_attention()

    root = Path(args.root)
    with np.load(Path(args.dataset) / f"{args.split}.npz", allow_pickle=False) as z:
        ours = {k: z[k] for k in ("actions", "masks", "valid", "offsets", "battle_ids")}
    offsets, battle_ids = ours["offsets"], ours["battle_ids"]
    n_traj = len(battle_ids) if args.limit is None else min(args.limit, len(battle_ids))

    pretrained = get_pretrained_model("Kakuna")
    checkpoint = args.checkpoint if args.checkpoint is not None else pretrained.default_checkpoint
    agent = pretrained.initialize_agent(checkpoint=checkpoint, log=False, action_temperature=1.0)
    ckpt_path = pretrained.get_path_to_checkpoint(checkpoint)
    policy = agent.policy
    policy.eval()
    # FlexAttention's cache-free path asserts `self.training`; attention layers hold no
    # parameters and flex forces dropout 0, so flipping only these flags keeps eval semantics.
    flex = [m for m in policy.modules() if isinstance(m, FlexAttention)]
    if not flex:
        raise RuntimeError("no FlexAttention layers: use_flex_attention() did not take effect")
    for m in flex:
        m.training = True
    device = agent.DEVICE

    native = ParsedReplayDataset(
        dset_root=str(root), formats=["gen9ou"],
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function, shuffle=False)
    amago_ds = MetamonAMAGODataset(native, dset_name="kakuna-label")
    files = []
    for f in sorted(native.filenames):
        m = NAME.match(Path(f).name)
        if m is None:
            raise ValueError(f"Unexpected filename: {f}")
        if pile.split_of(m.group(1)) == args.split:
            files.append((f, m.group(1)))
    all_lengths = np.diff(offsets)
    if args.longest:  # debug: the N longest trajectories, kept in npz order (not a prefix)
        sel = np.sort(np.argsort(-all_lengths, kind="stable")[:args.longest])
    else:
        sel = np.arange(n_traj)
    n_traj = len(sel)
    if len(files) != len(battle_ids):
        print(f"WARNING: root has {len(files)} files, dataset {len(battle_ids)} trajectories",
              flush=True)
    if [files[i][1] for i in sel] != [str(battle_ids[i]) for i in sel]:
        raise ValueError("Pile file order differs from dataset battle_ids")

    lengths = np.diff(offsets)
    out_offsets = np.concatenate([[0], np.cumsum(lengths[sel])]).astype(offsets.dtype)
    n_dec = int(out_offsets[-1])
    shard_dir = Path(str(args.out) + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"split": args.split, "dataset": args.dataset, "checkpoint": checkpoint,
                "n_traj": n_traj, "longest": args.longest, "shard_size": args.shard_size,
                "first_battle": str(battle_ids[sel[0]]), "last_battle": str(battle_ids[sel[-1]]),
                "n_dec": n_dec}
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"{manifest_path} does not match this run; delete {shard_dir}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    n_shards = -(-n_traj // args.shard_size)

    def shard_path(k: int) -> Path:
        return shard_dir / f"shard_{k:05d}.npy"

    # Per shard: sort by length (descending), then greedily pack batches of at most
    # --batch-trajectories items whose padded size B*pad_to(maxlen) <= --max-batch-steps.
    pending = [k for k in range(n_shards) if not shard_path(k).exists()]
    batches: list[list[int]] = []  # trajectory indices (npz ids), shard by shard
    batch_shard: list[int] = []
    for k in pending:
        pos = range(k * args.shard_size, min((k + 1) * args.shard_size, n_traj))
        order = sorted((int(sel[p]) for p in pos), key=lambda i: -int(lengths[i]))
        cur: list[int] = []
        for i in order:
            if cur and (len(cur) >= args.batch_trajectories
                        or (len(cur) + 1) * pad_to(int(lengths[cur[0]])) > args.max_batch_steps):
                batches.append(cur)
                batch_shard.append(k)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
            batch_shard.append(k)

    class Seqs(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(files)

        def __getitem__(self, i):
            data = native.load_filename(files[i][0])
            info = data[1]
            chosen = np.asarray(info["chosen"], dtype=np.int64)
            legal = np.zeros((len(chosen), 13), dtype=bool)
            for s, acts in enumerate(info["legal"]):
                for a in acts:
                    if 0 <= a < 13:
                        legal[s, a] = True
            return amago_ds._process_data(data), chosen, legal

    loader = torch.utils.data.DataLoader(
        Seqs(), batch_sampler=batches, num_workers=args.workers,
        collate_fn=lambda items: items, prefetch_factor=2 if args.workers else None)

    @torch.no_grad()
    def logits_full(batch, amp: bool = True) -> torch.Tensor:
        obs = {k: v.to(device) for k, v in batch.obs.items()}
        rl2s, tidx = batch.rl2s.to(device), batch.time_idxs.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            tstep = policy.tstep_encoder(obs=obs, rl2s=rl2s)
            emb, _ = policy.traj_encoder(tstep, time_idxs=tidx, hidden_state=None)
            out = policy.actor.actor_network_forward(
                emb, straight_from_obs={"illegal_actions": obs["illegal_actions"]})
        return out[..., -1, :].float()  # [B, L, 13]; last gamma = test-time gamma

    @torch.no_grad()
    def probs_live(rl: object) -> np.ndarray:
        """Real cached inference path, one step at a time (batch 1)."""
        for m in flex:
            m.training = False
        hidden = policy.traj_encoder.init_hidden_state(1, device)
        rows = []
        for t in range(len(rl.actions)):
            obs = {k: v[t:t + 1].unsqueeze(0).to(device) for k, v in rl.obs.items()}
            rl2 = rl.rl2s[t:t + 1].unsqueeze(0).to(device)
            ti = rl.time_idxs[t:t + 1].unsqueeze(0).to(device)
            # fp32: the KV cache is fp32 and bf16 autocast cannot write into it
            tstep = policy.tstep_encoder(obs=obs, rl2s=rl2)
            emb, hidden = policy.traj_encoder(tstep, time_idxs=ti, hidden_state=hidden)
            out = policy.actor.actor_network_forward(
                emb, straight_from_obs={"illegal_actions": obs["illegal_actions"]})
            rows.append(torch.softmax(out[0, 0, -1].float(), -1).cpu().numpy())
        for m in flex:
            m.training = True
        return np.stack(rows)

    def padded_batch(rls):
        batch = RLData_pad_collate(rls)
        length = batch.rl2s.shape[1]
        extra = pad_to(length) - length
        if extra:  # bucket lengths so compiled flex kernels are reused
            batch.obs = {k: torch.nn.functional.pad(
                v, (0,) * (2 * (v.ndim - 2)) + (0, extra),
                value=4.0 if v.dtype.is_floating_point else 4)
                for k, v in batch.obs.items()}
            batch.rl2s = torch.nn.functional.pad(batch.rl2s, (0, 0, 0, extra), value=4.0)
            batch.time_idxs = torch.nn.functional.pad(batch.time_idxs, (0, 0, 0, extra), value=4)
        return batch

    oom_splits = 0

    def softmax_items(items) -> list[np.ndarray]:
        """Per-item [n,13] softmax; on CUDA OOM split the batch in halves and retry."""
        nonlocal oom_splits
        try:
            sm = torch.softmax(logits_full(padded_batch([it[0] for it in items])), -1).cpu().numpy()
            return [sm[b, :len(it[1])] for b, it in enumerate(items)]
        except torch.OutOfMemoryError:
            if len(items) == 1:
                raise
        oom_splits += 1
        torch.cuda.empty_cache()
        mid = len(items) // 2
        return softmax_items(items[:mid]) + softmax_items(items[mid:])

    live_delta = 0.0
    bf16_delta = 0.0
    live_checked = 0
    pos_of = {int(i): p for p, i in enumerate(sel)}
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    dec_done = traj_done = 0
    todo_dec = sum(int(out_offsets[min((k + 1) * args.shard_size, n_traj)]
                       - out_offsets[k * args.shard_size]) for k in pending)
    shard_probs: dict[int, np.ndarray] = {}
    shard_left = {k: 0 for k in pending}
    for k in batch_shard:
        shard_left[k] += 1
    for items_idx, items in zip(batches, loader):
        sms = softmax_items(items)
        for i, (rl, chosen, legal), sm in zip(items_idx, items, sms):
            p = pos_of[i]
            k = p // args.shard_size
            if k not in shard_probs:
                p0 = k * args.shard_size
                p1 = min(p0 + args.shard_size, n_traj)
                shard_probs[k] = np.full((int(out_offsets[p1] - out_offsets[p0]), 13), np.nan,
                                         dtype=np.float16)
            base = int(out_offsets[k * args.shard_size])
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            n = len(chosen)
            if n != hi - lo:
                raise ValueError(f"traj {i} {files[i][1]}: kakuna {n} steps vs ours {hi - lo}")
            valid = ours["valid"][lo:hi]
            if not np.array_equal(chosen[valid], ours["actions"][lo:hi][valid]):
                raise ValueError(f"traj {i} {files[i][1]}: data actions differ from npz")
            if not np.array_equal(legal, ours["masks"][lo:hi]):
                raise ValueError(f"traj {i} {files[i][1]}: legal masks differ from npz")
            shard_probs[k][int(out_offsets[p]) - base:int(out_offsets[p + 1]) - base] = \
                sm.astype(np.float16)
            if p < args.check_live:
                live = probs_live(rl)
                one = RLData_pad_collate([rl])
                ref = torch.softmax(logits_full(one, amp=False), -1)[0, :n].cpu().numpy()
                live_delta = max(live_delta, float(np.abs(live - ref).max()))
                bf16_delta = max(bf16_delta, float(np.abs(sm - ref).max()))
                live_checked += 1
        traj_done += len(items)
        dec_done += sum(len(it[1]) for it in items)
        k = pos_of[items_idx[0]] // args.shard_size
        shard_left[k] -= 1
        if shard_left[k] == 0:
            tmp = shard_path(k).with_suffix(".tmp.npy")
            np.save(tmp, shard_probs.pop(k))
            os.replace(tmp, shard_path(k))
            el = time.time() - start
            rate = dec_done / el
            print(json.dumps({
                "shard": k, "shards_total": n_shards, "trajectories_done_this_run": traj_done,
                "decisions_done_this_run": dec_done, "decisions_per_s": round(rate, 1),
                "eta_s": round((todo_dec - dec_done) / rate), "oom_splits": oom_splits,
                "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)
    torch.cuda.synchronize()
    wall = time.time() - start
    peak_gpu_mb = torch.cuda.max_memory_allocated() / 2**20

    probs = np.concatenate([np.load(shard_path(k)) for k in range(n_shards)])
    rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in sel])
    if probs.shape != (n_dec, 13) or len(rows) != n_dec:
        raise ValueError(f"assembled probs {probs.shape} != ({n_dec}, 13)")
    if args.longest == 0 and not (
            np.array_equal(out_offsets, offsets[:n_traj + 1]) and offsets[0] == 0
            and np.array_equal(rows, np.arange(n_dec))):
        raise ValueError("output is not the exact dataset prefix")

    valid_all = ours["valid"][rows]
    masks = ours["masks"][rows]
    actions = ours["actions"][rows]
    nan_rows = int(np.isnan(probs).any(axis=-1).sum())
    pv = probs[valid_all].astype(np.float32)
    av = actions[valid_all]
    row_sum = np.nansum(probs.astype(np.float32), axis=-1)
    illegal_mass = np.where(masks, 0.0, np.nan_to_num(probs.astype(np.float32))).sum(-1)
    has_legal = masks.any(-1)
    stats = {
        "trajectories": n_traj, "decisions": n_dec, "valid_decisions": int(valid_all.sum()),
        "wall_s_this_run": wall, "decisions_per_s_this_run": dec_done / wall if dec_done else None,
        "peak_gpu_mb": peak_gpu_mb, "oom_splits": oom_splits, "max_batch_steps": args.max_batch_steps,
        "nan_rows": nan_rows,
        "misaligned_rows": 0,  # any misalignment raises above
        "argmax_agree_valid": float((pv.argmax(-1) == av).mean()),
        "mean_prob_on_data_action_valid": float(pv[np.arange(len(av)), av].mean()),
        "illegal_mass_mean": float(illegal_mass[has_legal].mean()),
        "illegal_mass_max": float(illegal_mass[has_legal].max()),
        "rows_without_legal_action": int((~has_legal).sum()),
        "row_sum_min": float(row_sum.min()), "row_sum_max": float(row_sum.max()),
        "live_path_checked_trajectories": live_checked,
        "live_cached_vs_full_fp32_max_abs_prob_delta": live_delta if live_checked else None,
        "bf16_vs_fp32_full_max_abs_prob_delta": bf16_delta if live_checked else None,
    }
    meta = {
        "model": "Kakuna", "checkpoint": checkpoint,
        "checkpoint_path": os.path.realpath(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "attention": "flex (SlidingWindowFlexAttention, window 96, weights strict)",
        "observation_space": type(pretrained.observation_space.base_obs_space).__name__,
        "action_space": type(pretrained.action_space).__name__,
        "action_mapping": "identity: UniversalAction 13 slots (4 moves, 5 switches, 4 move+tera)",
        "temperature": 1.0, "precision": "bf16 autocast, probs stored float16",
        "probs": "unclipped softmax of illegal-masked logits (live clip [0.001,0.99] not applied)",
        "context": "full-battle causal pass, sliding window 96 == live cached inference "
                   "(cache 128 > window); no truncation/chunking; verified by --check-live",
        "root": str(root), "dataset": args.dataset, "split": args.split,
        "stats": stats,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, probs=probs, offsets=out_offsets, battle_ids=battle_ids[sel],
             meta=np.asarray(json.dumps(meta)))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
