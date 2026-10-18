# W5.BS1.3 — Incident report (heldout one-shot aborted)

**Status: INVALID / INCOMPLETE.** No heldout estimate of C+search strength exists; no promotion.
The `data/teams_w5/heldout` pool is spent (claimed). Gameplay results of the completed blocks are
**sealed**: no win/loss of any BS1.3 battle has been computed or viewed. They must not be merged
into any later benchmark nor used to select a fix.

**Root cause: under investigation.** Known so far is only the condition that raised the exception
(below), not why the rollout did not terminate.

## Release under test (RC3)

| Item | Value |
|---|---|
| Tag `w5-bs-rc3` | `b2870317651f8b9d74b823f5f4729bf15d81d842` (= HEAD at launch, clean tree) |
| Code commit in manifest | `f3c83b46ff1c5fcae4923bdca16f1fadf7fc86ca` |
| `reports/w5_bs_rc1_manifest.json` sha256 | `96d5ac1c0201fab37fdf8a03340a55fc3274179c26fa548a702479d63df78e6d` |
| `configs/w5_bs_rc1.yaml` sha256 | `9eeff61afc5c8e91a0c05966c5dce97e10db0127fb3fdb1daf2f4eeda163fbf2` |
| External watchdog `/tmp/bs13_watchdog.py` sha256 | `9aaaac6dbc74dd549b924cd46c0850e2d267ae0e07d5309f3705355ca63eb441` (copy: `runs/w5-bs-rc1-watchdog.py`) |
| Design | 600 battles: 50 team pairs / opponent × both directions × {C, C+search} × {Grunt, GymLeader, C}; seed 20261501; 4 pair blocks 12/13/12/13; 2 lanes × 4 search workers; D = 5.0 s |
| Unit | transient `w5-bs13.service`, CPUQuota 1000 %, MemoryMax 14G, Nice 10, Restart=no |

## Preflight (read-only, all passed)

Clean tree; HEAD = `w5-bs-rc3` = `b287031`; no `final.started.json`; no `final.window.*.json`;
local server HTTP 200; no `runs/w5-bs-rc1-b*`. The transient unit from the aborted RC2 launch
vanished on `reset-failed`, so it was recreated with the identical `systemd-run` command (one
instance) instead of `systemctl start`.

## Earlier pre-claim abort (RC2, not a heldout attempt)

2026-10-18 15:49:41 (UTC+7): the driver's `verify_release` refused RC2 (`collector.py`, `repro.py`
flagged as unfrozen: the module closure depended on the caller's imports). No lane started, no
claim was created; fixed in RC3.

## Timeline (UTC+7, 2026-10-18)

| Time | Event |
|---|---|
| 16:36:26 | `w5-bs13` started on `b287031`; watchdog `w5-bs13-watch` started 16:36:29 |
| 16:36:44 | Heldout claimed: `final.started.json` + windows 0of2/1of2 for pairs 0–12 |
| 17:51:23 | Block 1 done (both lanes exit 0, both valid); block 2 windows claimed |
| 19:12:20 | Lane 0: one rollout pair dropped for `CalledForeignConsecutive` (Metronome → Fire Spin), the known, allowlisted parser class; handled as designed (`runs/w5-bs-rc1-new-class/forensics/5ec9d15b_8.*`) |
| 19:24:31 | Block 2 done (both lanes exit 0, both valid); block 3 windows claimed |
| 20:10:11 | **Lane 0, block 3, cell C+search vs C, during its 3rd battle** (2 completed): a search worker raised `RuntimeError: battle exceeded max_rounds` inside a reconstructed-world rollout. Not allowlisted ⇒ new class: that decision returned C's legal action, traceback + root payload written, lane 0 aborted (`ABORT: new search exception class`) |
| 20:10:28–20:10:33 | Lane 1 kept running: it completed its 12 block-3 battles of arm **C (no search) vs C**, then started its C+search vs C cell |
| 20:10:40 | Watchdog confirmed the lane-0 failure (two observations 20 s apart) and stopped the whole unit; lane 1's in-progress C+search battle was killed (0 completed in that cell) |

Watchdog reaction: 29 s from lane-0 abort to unit stop. During that window lane 1 completed 12
battles (C arm only) and started one search battle that did not finish.

Unit totals: 3 h 34 min wall, 27.3 CPU-hours, peak memory 11.6 GB, exit status 143 (stopped).

## Block completion

| Block | Pairs / opponent | Exit codes | Lanes valid | Wall |
|---|---|---|---|---|
| 1 | 0–12 | 0, 0 | 2 / 2 | 4,479 s |
| 2 | 12–25 | 0, 0 | 2 / 2 | 5,588 s |
| 3 | 25–37 | lane 0 aborted, unit stopped | 0 / 2 | partial |
| 4 | 37–50 | not started | – | – |

Blocks 1–2: 300 battles in 24 runs, 0 battle errors, 0 model-illegal (lane validity flags only;
results sealed). Block 3 partial: both lanes completed Grunt and GymLeader cells for both arms
and the C-arm vs C cell; lane 0 completed 2 C+search vs C battles before the abort.
Projection after block 2: 20,134 s total, within the 16 h budget.

## The exception

```
src/pokeai/cf/recon_search.py:237  _decide_chunk → rollout(...)
src/pokeai/cf/branching.py:118      match.play(max_rounds=max_rounds)   # rollout default 600
src/pokeai/cf/offline.py:722        raise RuntimeError("battle exceeded max_rounds")
```

- `max_rounds` bounds the harness's `round_index` (protocol/delivery rounds of `OfflineMatch`),
  not the Showdown `turn`; the exception therefore does not by itself show a 600-turn battle.
- Search drops a (world, seed) pair only for the metamon `ForwardException` family; this
  `RuntimeError` propagated to the worker → controller → fail-fast, as designed for an unknown
  class.
- The class was never observed in BS0, BS1.0, BS1.1, BS1.2 or BS1.2b.

## Evidence (local, gitignored; preserved)

| File | sha256 |
|---|---|
| `runs/w5-bs-rc1-new-class/new_class_1d572d61_1792329011.txt` (traceback) | `9367daefb9f69a16d557fc75519dac456aaa95e0780c49167491cd5f49058cb2` |
| `runs/w5-bs-rc1-new-class/new_class_1d572d61_1792329011.payload.pkl` (root payload) | `ccab84976bf2a5bc9427bd36b3f51ab935b7905fa11f84cd15c86a4f382ddf18` |
| `runs/w5-bs-rc1-new-class/forensics/5ec9d15b_8.json` (known-class drop) | `eec1d3e652d947397b4fa66a42ea33d87e3015fb75869e1d8dc89521cf0b0b41` |
| `runs/w5-bs-rc1-new-class/forensics/5ec9d15b_8.payload.pkl` | `30bc03bb741e1854ddfc2d6977225f8da96e2f9a581896a59acbf23077181c5c` |
| `runs/w5-bs-rc1-projections.jsonl` | `460696cff56947953bf4935f61a723c6ec666bfbc4852be5b4b9d12062708b7c` |
| `runs/w5-bs-rc1-watchdog.log` | `37d079f0a4a0b44ef0d053bf3655d4d65a58bfb1697b536d5888eb1fcd16cb99` |

Backup (read-only, outside the repo, same physical disk):
`~/backups/poke-mon/w5-bs1-3-20261018/`: `runs-full.tar.zst` (all of `runs/`, sha256
`1b2bc4c7c72bc198b2072d3d9c34f36221591a11c4a781e49c0d6a89ace06b66`, `zstd -t` OK),
`bs1_3_evidence.sha256` (90 files: lane dirs, fail dir, logs, the 36 BS1.3 battle run dirs listed
in `bs1_3_battle_run_dirs.txt`), `heldout-claim/` + `heldout-claim.sha256`. The claim and window
files are also committed under `data/teams_w5/heldout/` as the permanent spent marker.

## Unknown / open

1. Why the rollout did not finish within 600 delivery rounds: a genuinely long simulated game,
   two bots that never close the game, or a harness loop (round_index advancing without turns).
2. Whether the reconstructed world in the failing payload is itself faithful (S1-type check).
3. The Showdown `turn`, HP/PP and last events of the failing rollout (to be reproduced from the
   saved payload and RNG seeds, without reading benchmark win/loss).

Next: BS1.3-P forensics (reproduce, classify, then a typed handling contract), QA on collect/dev,
and only then a fresh, truly independent heldout (BS1.4), reported as a new confirmation planned
after this incident, not as the project's first one-shot.
