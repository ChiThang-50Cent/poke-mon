# W5.BS1.4: incident report (fresh heldout confirmation aborted)

**Status: INVALID / INCOMPLETE.** There is no heldout estimate of C+search strength, and no promotion. The pool `data/teams_w5_bs14/heldout` is spent (claimed).

**Gameplay is sealed.** 156 battles completed: 144 in block 1 and 12 in block 2. No win or loss from any of them has been computed or viewed. They must not be merged into any later benchmark or used to choose a fix. This report contains operational evidence only.

BS1.4 was the fresh confirmatory evaluation planned after BS1.3 failed for technical reasons (`reports/w5_bs1_3_incident.md`). It is not the project's first one-shot.

**Root cause: an orchestration race in the release/claim tooling added during BS1.4 preparation.** It is not in the search, the model or the battle engine; `src/pokeai/` at launch is byte-identical to `w5-bs-rc4`.

## Bundle under test

| Item | Value |
|---|---|
| Tag `w5-bs14-eval` | `432c34387e10895a3f14f3b4fbf65d550d4dc5e3` (= HEAD at launch, clean tree) |
| Search release | `w5-bs-rc4` (`5031f36`, code `17e609a`); `git diff w5-bs-rc4 HEAD -- src/` empty |
| `reports/w5_bs14_manifest.json` sha256 | `de9e56aaba537c7f07f6293cfb341df337b43bf1196805fb94f4ec96728b8fc4` |
| `configs/w5_bs14.yaml` sha256 | `886597eb9388219f2d9d1f8704b304afacc86b34e1a1a5b139a07d0bdb73a3d7` |
| External watchdog `/tmp/bs14_watchdog.py` sha256 | `334348921397a87959870c2887c3549af6d762a436ea1a62e4f8ed2954e88deb` (copy: `runs/w5-bs14-watchdog.py`; differs from the BS1.3 watchdog only in unit name, log path and lane glob) |
| Design | 600 battles: 50 team pairs per opponent × both directions × {C, C+search} × {Grunt, GymLeader, C}. Schedule seed 20261911, search seed 20261101. 4 pair blocks of 12/13/12/13. 2 lanes × 4 search workers. D = 5.0 s. Budget 16 h |
| Unit | transient `w5-bs14.service`: CPUQuota 1000 %, MemoryMax 14G, Nice 10, Restart=no |

## Preflight (read-only)

These five checks all passed:
- clean tree;
- HEAD = `w5-bs14-eval`;
- `src/` identical to RC4;
- no `final.started.json`;
- no `final.window.*.json`.

Also confirmed: the local server answered HTTP 200, and there were no `runs/w5-bs14*` directories.

The first clean-tree check failed. The only untracked file was the empty BS1.3 lock `data/teams_w5/heldout/final.lock`. It was added to the local `.git/info/exclude` (not committed). The file was kept in place, and a copy is in the BS1.3 backup.

## Timeline (UTC+7)

| Time | Event |
|---|---|
| 2026-10-18 23:53:18 | `w5-bs14` started on `432c343`. Watchdog `w5-bs14-watch` started at 23:53:21 |
| 23:53 | The driver and both lanes passed `verify_release`. Claim `final.started.json` was created, plus windows 0of2 and 1of2 for pairs 0–12 |
| 2026-10-19 01:23:04 | Block 1 done: both lanes exit 0, both valid, 5,386 s. Projected total 22,443 s, within the 57,600 s budget |
| 01:23 | Block 2: lane 1 ran `verify_release` and claimed window `1of2.12-25` |
| 01:23:23 | **Lane 0, block 2: `verify_release` refused before claiming**: `team_pool: unexpected entry .final.started.json.bexb_lz0`. Lane 0 exited, window `0of2.12-25` was never created, and lane 0 ran no battle in block 2 |
| 01:23:33–01:23:37 | Lane 1 completed its 12 block-2 battles of arm **C (no search) vs Grunt**, then started its C+search vs Grunt cell |
| 01:23:46 | The watchdog confirmed the lane-0 failure (two observations) and stopped the whole unit. Lane 1's in-progress C+search battle was killed, with 0 completed in that cell |

**Watchdog reaction: about 23 s** from the lane-0 refusal (01:23:23, stdout mtime) to the unit stop (01:23:46). During that window lane 1 completed 12 battles, all in the C arm, and started one search battle that did not finish.

Unit totals: 1 h 30 min wall, 11.1 CPU-hours, peak memory 11.1 GB, exit status 143 (stopped). No `runs/w5-bs14-new-class` directory was created, since no search exception occurred.

## Block completion

| Block | Pairs per opponent | Exit codes | Lanes valid | Wall |
|---|---|---|---|---|
| 1 | 0–12 | 0, 0 | 2 / 2 | 5,386 s |
| 2 | 12–25 | lane 0 refused before claim; lane 1 stopped | 0 / 2 | partial |
| 3–4 | 25–50 | not started | – | – |

The 14 battle run dirs hold 156 battles (144 + 12) with 0 battle errors (`error` field only). Lane validity comes from the flags only; results are sealed.

## Root cause (confirmed from code at `432c343`)

1. `scripts/evaluate_w5.py::_create_exclusive()` writes the claim atomically. It calls `tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")`, then fsync, then `os.link`, then unlink. The temp file therefore exists briefly **in the pool root**.
2. In `scripts/bs14_manifest.py::pool_problems()`, called by `verify_release()`, any pool root entry other than `gen9ou` or `final.*` is refused. A `.final.started.json.*` temp file is such an entry.
3. `verify_release()` runs **outside** the `flock(final.lock)` that only `claim_pool()` takes.

At the start of block 2, lane 1 was inside `claim_pool()`: it re-attempted `final.started.json`, which already existed, then wrote its window, while lane 0 was listing the pool root in `verify_release()`. Lane 0 saw lane 1's temp file and refused, as its rule required.

Block 1 did not hit this only because of timing.

Tests at `432c343` covered concurrent claim vs claim, and each verifier refusal in isolation. They did not cover verify running concurrently with claim.

## Evidence (local, gitignored; preserved)

| File | sha256 |
|---|---|
| `runs/w5-bs14-b2/lane0/stdout.log` (refusal traceback) | `7a474f0200b2feb545822eb70bbbb87c701440d2947816aa9fd9805148954635` |
| `runs/w5-bs14-projections.jsonl` | `d7f744cad36ce5a38b64698664a4e151367418f63546873e0ab33e58df2ad86c` |
| `runs/w5-bs14-watchdog.log` | `9118d2e92896bdccb61be3923786d7ee8a8d9a8b9ca3eba52794362f9f34aa42` |

Committed claim markers under `data/teams_w5_bs14/heldout/` contain only `purpose` / `config` (arms, battles, opponents, pool, reference_arm, run), with no gameplay:

| File | sha256 |
|---|---|
| `final.started.json` | `f2e349bfbfdb89ce54a9a84687254af5767a37c3972a054e7b570d08fc55c883` |
| `final.window.0of2.0-12.json`, `final.window.1of2.0-12.json`, `final.window.1of2.12-25.json` | `02bb667b9bcf4d0d59972efabdb81e5210a4d305d19a1f8a2c5f05f6808d62d1` (identical content) |

`final.lock` is empty and is not committed.

Backup (read-only, outside the repo, same disk): `~/backups/poke-mon/w5-bs1-4-20261019/` contains:
- `bs1_4_evidence.tar.zst` (sha256 `91c649697f62bbd2a5807ecc8bfeafb2b8b78026e8cb8c5f6902b2da18ef948a`, `zstd -t` OK);
- `bs1_4_evidence.sha256` (38 files: lane dirs, projections, watchdog script and log, and the 14 battle run dirs listed in `bs1_4_battle_run_dirs.txt`);
- `heldout-claim/` and `heldout-claim.sha256`.

## Decision and next step

The user accepted INVALID / INCOMPLETE and approved continuing W5 conditionally. The next milestone is **BS1.4-P: infrastructure hardening and dress rehearsal**:
- verify and claim run inside one lock-owned critical section;
- staging is moved outside the pool;
- stress, crash, full-driver and failure-injection gates I1–I5 run on disposable rehearsal pools;
- no change to `src/pokeai/`.

No BS1.5 pool will be created before I1–I5 pass. BS1.5 is the last planned heldout confirmation in W5: if it is aborted again, W5 ends as "C+Search: dev-validated, independent confirmation unavailable".
