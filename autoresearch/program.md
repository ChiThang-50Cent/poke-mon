# autoresearch: policy architecture/optimizer search for the gen9ou bot

Adapted from karpathy/autoresearch. You are an autonomous researcher improving how well a small
recurrent policy imitates strong self-play play, under a fixed training budget on one RTX 3060.

## Why this loop exists

The production policy (1.30M GRU) underfits the Metamon self-play pile: train and val loss are
almost equal, and 3x more data moved held-out accuracy only 0.560 -> 0.584
(`reports/w6p_pile_report.md`). The bottleneck is the model/optimizer, not data. Metamon found
the same for small GRUs and moved to 15M+ Transformers. This loop searches the space cheaply.

## Files

- `autoresearch/prepare.py` — fixed harness: data (memory-mapped), `evaluate` (the metric),
  `check_contract`, `save`, constants (`TIME_BUDGET`=300 s, `MAX_PARAMS`=20M, `MAX_STEP_MS`=10).
  **Read-only.**
- `autoresearch/train.py` — **the only file you edit**: model, optimizer, loop.
- `autoresearch/program.md` — this file (edited by the human).

## Setup (once per run, with the human)

1. Agree a tag (e.g. `oct20`) and an **experiment budget** (number of experiments or hours).
   The human's machine is shared: the loop is bounded, unlike the original "never stop".
2. `git checkout -b autoresearch/<tag>` from main.
3. Read `prepare.py`, `train.py`, this file.
4. Check `.cache/autoresearch/manifest.json` exists; otherwise ask the human to run
   `uv run python autoresearch/prepare.py`.
5. Create untracked `autoresearch/results.tsv` with the header below; first run = baseline as is.

## Running one experiment

```bash
systemd-run --user --unit=autoresearch-run -p Restart=no -p CPUQuota=400% -p MemoryMax=8G \
  --working-directory=$PWD --setenv=PATH=$PATH --setenv=METAMON_CACHE_DIR=$PWD/.cache/metamon \
  --setenv=OMP_NUM_THREADS=1 -p StandardOutput=file:$PWD/runs/autoresearch_run.log \
  -p StandardError=file:$PWD/runs/autoresearch_run.log \
  nice -n 10 /home/thangnc/.local/bin/uv run python autoresearch/train.py
```

Wait for the unit to finish (`systemctl --user is-active autoresearch-run`), then
`grep "^val_nll:\|^val_top1:\|^num_params:\|^peak_vram_mb:\|^step_ms:" runs/autoresearch_run.log`.
Empty output = crash: `tail -n 50 runs/autoresearch_run.log`. A run longer than 12 minutes is a
failure: `systemctl --user stop autoresearch-run`. One experiment at a time, never in parallel.

## What you may change

Anything in `train.py`: architecture (token encoder, attention over the 87 tokens, memory
type/size/depth, heads), optimizer, schedule, batch construction (e.g. length bucketing), mixed
precision, loss weights, regularization.

## Hard rules

- Edit only `autoresearch/train.py`. No new dependencies. Do not modify `prepare.py`, data, or
  any other file.
- The model section (above `# ---- training ----`) imports only `torch` and defines
  `build_model()`; all training code stays inside `main()`. The exported `model.py` is loaded by
  live play exactly like that.
- Model contract (enforced by `check_contract`, a violation crashes the run):
  `model(tokens, numbers, hidden) -> (logits[B,T,13], hidden)`, `forward_with_value`,
  `config.value_head=True`, stepwise logits == full-sequence logits (causal: no peeking at future
  turns), <= 20M parameters, <= 10 ms per live decision.
- Inputs are only the per-turn observation (tokens, numbers). Never feed actions, outcomes or
  anything from the future into the policy input.
- Never read `datasets/w6p/val.npz` (sealed promotion check), W3/W4 test splits, or any heldout
  team pool. Never run battles, never touch the Showdown server.
- The metric is `val_nll` (lower is better), from `prepare.evaluate`. Objective changes that
  intentionally stop imitating (winner-only filtering, advantage weighting) cannot be judged by
  this metric: note them as ideas for the battle track, do not run them here.

## Decide keep/discard

- Keep if `val_nll` improves by more than the noise threshold **0.005** (to be refined from
  repeat baselines), or is within noise and the code is simpler.
- Simplicity criterion as in the original: small gains that add ugly complexity are not worth it;
  equal results with less code are a win.
- Keep: commit stays, branch advances. Discard: `git reset --hard` to the previous keep.

## Logging

`autoresearch/results.tsv` (untracked, tab-separated):

```
commit	val_nll	val_top1	params_M	vram_gb	status	description
```

status is `keep`, `discard` or `crash` (0.000000 for crash metrics).

## Loop

Until the agreed budget is used up:
1. Pick one idea; edit `train.py`; `git commit`.
2. Run; read metrics; log the row.
3. Keep or reset.
4. Every 5 keeps, write a 3-line note at the end of `autoresearch/notes.md` (untracked): what
   helped, what did not, next hypotheses.

Then stop and report the best commit, its metrics and the notes.

## After the loop (human-gated, not part of the loop)

The proxy is imperfect: better imitation does not always mean more wins. A candidate is only
promoted by the project process: train it fully, check the sealed `datasets/w6p/val`, run the paired
dev benchmark against the champion (`configs/w6p_scale_dev.yaml` pattern; `model_dir` = the
exported checkpoint directory, loaded via `load_research_model`), then one fresh-pool final.
