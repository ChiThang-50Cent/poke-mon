# W5.BS1.4-P: infrastructure hardening (tooling only)

Base `8922ef9`; infrastructure commit `d514078` (the rehearsals ran on it, clean tree). Nothing under `src/pokeai/` changed: `git diff 8922ef9 d514078 -- src/` is empty (I5). The spent pools `data/teams_w5/heldout` and `data/teams_w5_bs14` are unchanged. The only battles run were the I3/I4 rehearsals on disposable copies of the dev pool; no win rate was computed.

## Design (one lock protocol, `scripts/pool_claim.py`)
* `pool_lock(pool)` is the only code that takes `flock(LOCK_EX)` on `<pool>/final.lock`. It yields a `PoolLock` token. The token is invalid after the block exits and is bound to one pool.
* `verify_unlocked(lock, ...)` and `claim_unlocked(lock, ...)` require the token. They raise if it is missing, released, or from another pool. No nested flock exists anywhere.
* Lane (`evaluate_w5.py`): `verify_and_claim()` runs `with pool_lock: verify_unlocked; claim_unlocked` as one critical section.
* Driver (`bs1_2_run.py`): `verify_only()` runs under the lock before any lane starts. It also runs before every later block (decision: yes).
  * Why: it is cheap, and it stops a block from launching (and loading models) after a crashed claim left an orphan, or after anything in the bundle changed.
  * It cannot race a lane, because it takes the same lock. Lanes verify under the lock regardless.
* Staging: temp claim files are written to `<pool.parent>/.<pool.name>.claim-staging/` (same filesystem, outside the pool). The sequence is write + fsync, `os.link` into the pool (fails if the target exists), fsync the pool dir, unlink the temp file. `_claim_step(stage, path)` is a no-op test interleaving point that tests monkeypatch in child processes. It is not a production flag.
* Verifier (`bs14_manifest.pool_problems`) is strict. Allowed pool root entries are `gen9ou/`, an empty `final.lock`, `final.started.json` and `final.window.<l>of<n>.<a>-<b>.json`.
  * Claim artifacts must parse as complete JSON whose `config` equals the effective config.
  * Anything else is refused. There is no `.final*` or dotfile whitelist.
  * Any file in the staging dir is refused as an "orphan claim temp file" (fail closed, manual review). Nothing is ever deleted.
* Callers migrated, old paths removed: `claim_pool`, `verify_release` and the in-pool `_create_exclusive` are gone. `evaluate_w5.py`, `bs1_2_run.py`, `evaluate_w5_awbc.py`, `test_bs_rc1_claim.py`, `test_w5_benchmark.py` and `test_bs14_prep.py` are migrated.
* Manifest tool: `bs14_manifest.py` now takes `--config`/`--out`, so it is the single tool for the real bundle and for rehearsals. Its scripts section also hashes `pool_claim.py` and `bs_watchdog.py`.

## I1 concurrency (`tests/test_bs14p_hardening.py`)
* (a) Barrier test, using multiprocessing events. Process A is inside the claim, with the staging temp existing and the link not done. A hook monkeypatched in the child holds it there. The test checks that exactly 1 temp exists, outside the pool, and that 0 claims exist. Process B then starts verify+claim and blocks (checked 1 s later: still alive, no result). After release, both return `ok`. B claims its own window and nothing is refused. The pool holds `final.started.json` plus windows 0of2 and 1of2 of 12-25, and staging is empty. PASS.
* The same scenario FAILS on 432c343: `test_the_barrier_scenario_fails_on_the_432c343_implementation` extracts `scripts/` of 432c343 with `git archive`, runs it in a subprocess, and holds A at `os.link` after the temp exists in the pool root. B's `verify_release` then raises:
  `RuntimeError: release manifest mismatch; refusing the heldout claim: team_pool: unexpected entry .final.started.json.swvfmn2f`
  This is the exact BS1.4 incident. The test asserts that message.
* (b) Stress: 5 parametrized batches × 20 rounds = 100 rounds. Each round runs 6 processes released by a barrier on a fresh bundle, with 0–3 ms random jitter at the staging step. The processes are 2 driver verifies, 4 lane claims (the start claim, windows (0,2,0,12), (1,2,0,12), (0,2,12,25)) and a duplicate (1,2,0,12) attempt. Per round the test asserts:
  * 0 false verify refusals;
  * both driver verifies `ok`;
  * the duplicate window gives exactly one `ok` and one `FileExistsError`;
  * exactly the 4 expected claim files, each complete valid JSON with config equal;
  * exactly one `final.started.json`;
  * empty staging.

  Result: 100/100 rounds PASS.

## I2 crash safety
* `os._exit(9)` before the link: no claim exists, 1 orphan temp sits in staging, and the next verify (driver or lane) refuses with "orphan claim temp file". The refused lane claims nothing, and the orphan is not deleted.
* `os._exit(9)` after the link, before the unlink:
  * `final.started.json` exists with complete valid JSON, and the pool counts as claimed (a second start claim raises `FileExistsError`).
  * The orphan is detected by verify and not deleted.
* The verifier stays strict (parametrized, 9 cases). It refuses each of these:
  * a stray pool file;
  * a `.final.started.json.*` file;
  * a dotfile;
  * an edited team;
  * a non-empty `final.lock`;
  * a truncated start claim;
  * a claim with a different config;
  * a truncated window claim;
  * a malformed window name.
* The lock protocol is enforced: claim without a token, a released token and another pool's token are all refused.

## Watchdog
`scripts/bs_watchdog.py --unit --prefix --log` has the same logic as `runs/w5-bs14-watchdog.py` (`REPO` is derived from the file location). `watch()` takes injectable fakes.
* 3 tests cover the failure rule: lane process gone and `benchmark.json` not valid on 2 consecutive checks ⇒ stop the unit and log "LANE FAILURE".
* A running lane, a valid report, a single observation, and a failure that does not persist never stop the unit.

## Rehearsal tooling
* `scripts/bs_rehearsal_prep.py NAME [--layout 1x4|2x4]` creates `runs/rehearsal/NAME/`:
  * the 200 dev teams plus a manifest with role `heldout` and `"rehearsal": true`;
  * `config.yaml`, a copy of `configs/w5_bs14.yaml`;
  * `manifest.json`, made by `bs14_manifest.py --config/--out`.
* Only these config keys differ (test-checked): `pool`, `battles` (8), `run.out_prefix` (`w5-bs14p-NAME`), `run.release_manifest`, `run.block_pairs` ([[0,1],[1,2],[2,3],[3,4]]), `run.seed` (20262001) and `fail_dir`. Both `fail_dir` and `release_manifest` are inside the rehearsal dir.
* Dry-check of the real tool (a throwaway `dryrun`, since deleted): the manifest has 12 scripts, 201 pool files, and `verify_unlocked` returns exactly one problem, "manifest was generated from a dirty tree". On the committed tree that problem disappears.
* **Lane split finding:** the runner deals pair k to lane k % lanes (`lane_indices`).
  * With 1-pair blocks and 2 lanes, each block leaves ONE lane with no pair: battles per lane per block are [[2,0],[0,2],[2,0],[0,2]] (test-verified).
  * The idle lane still claims its window (8 claims in total) and runs 0 battles. `summarize([])` handles an empty record list, but I did not run an empty-lane evaluate end to end.
  * 1 pair/block therefore cannot exercise 2 lanes' battles concurrently.
  * Minimum change, with no `src/` edit: `--layout 2x4` (battles 16, blocks of 2 pairs). It gives 8 claims, both lanes busy in every block, and 96 battles (48 C+search) (test-verified [2,2] per block). The default stays the contract's 1x4.
* `scripts/bs_rehearsal_check.py NAME [--watchdog-log PATH]` reads only operational data and writes `check.json`. It checks:
  * the window claim set (no missing, duplicate or extra) and `final.started.json`;
  * every lane `valid: true` and the driver exit codes;
  * 48 battles with `error` null;
  * lane logs free of verifier refusals/tracebacks, and the watchdog log free of LANE FAILURE;
  * no orphan in staging.

  It computes no win rate (test-checked).

## Tests (targeted)
`tests/test_bs14p_hardening.py tests/test_bs14_prep.py tests/test_bs_rc1_claim.py tests/test_w5_benchmark.py`: **58 passed in 39.2 s** (subagent). Lint was run by the integrator (see the full suite section below).

## I3 rehearsal (integrator): PASS

**Setup.**
- Bundle `runs/rehearsal/r1/`, built by `bs_rehearsal_prep.py --layout 2x4 r1` on clean `d514078`. The pool is the 200 dev teams marked heldout + rehearsal. The config differs from `configs/w5_bs14.yaml` only in pool, out_prefix, fail_dir, release_manifest, `battles: 16`, `block_pairs` [[0,2],[2,4],[4,6],[6,8]] and seed 20262001.
- Layout: **2x4 was chosen over the contract's 1x4**. The runner deals pair k to lane k % 2, so 1x4 leaves one lane with 0 battles in every block, and half of the lane-blocks would not exercise a real lane. In BS1.5 every lane always has work (12/13 pairs per block). 2x4 keeps 4 blocks × 2 lanes = 8 lane-window claims, with both lanes busy: 96 battles, 48 C+search.
- Launch: the same path as a final run. Transient `w5-bs14p-r1.service` (CPUQuota 1000 %, MemoryMax 14G, Nice 10, Restart=no) runs `bs1_2_run.py --config runs/rehearsal/r1/config.yaml` (`final: true`, so the driver and lanes go through the lock protocol and manifest verification). The frozen `scripts/bs_watchdog.py` ran as `w5-bs14p-r1-watch`.

**Run.**
- 08:23:13–09:17:47 (UTC+7, 2026-10-19): 54 min 34 s wall, 6.75 CPU-hours, peak 11.3 GB, exit 0 (`Result=success`).
- Blocks: 725 / 918 / 969 / 640 s, all lanes exit 0. The block boundaries, where BS1.4 failed, crossed 3 times without a refusal.
- Watchdog: no LANE FAILURE; it logged "no longer active; watchdog exit" and exited 0.

**Checks** (`bs_rehearsal_check.py r1 --watchdog-log runs/rehearsal/r1/watchdog.log`, exit 0, `check.json` sha256 `39aa723a…3df74`):

| Check | Result |
|---|---|
| Window claims | 8 / 8 (`0of2`, `1of2` × `0-2`, `2-4`, `4-6`, `6-8`), 0 missing, 0 extra, 0 duplicate |
| `final.started.json` | present (exactly one) |
| Lane reports | 8 / 8 `valid: true` |
| Battles | 96 / 96, 0 with `error` |
| Driver exit codes | 4 / 4 blocks recorded, all 0 |
| Lane logs | 0 verifier refusals, 0 tracebacks |
| Claim staging | empty (0 orphans) |

Operator note: the first checker call failed only "watchdog log missing". The watchdog had been launched with `--log runs/rehearsal/r1/watchdog.log`, while the checker defaults to `runs/<prefix>-watchdog.log`. Passing `--watchdog-log` fixed it; no code changed. Final runs pass `--log runs/<prefix>-watchdog.log` (as r2 did).

## I4 failure injection (integrator): PASS

**Setup.** Separate disposable bundle `runs/rehearsal/r2/` (same layout), launched identically as `w5-bs14p-r2` + `w5-bs14p-r2-watch` (`--log runs/w5-bs14p-r2-watchdog.log`). Block-1 claims were created (started, `0of2.0-2`, `1of2.0-2`) and both lanes were playing.

**Injection.** At 09:22:01 the lane-0 `evaluate_w5.py` process (PID 2650288) got `kill -9`.

**Results.**

| Expectation | Observed |
|---|---|
| Watchdog detects the failure | 09:22:34 "LANE FAILURE confirmed: ['runs/w5-bs14p-r2-b1/lane0/benchmark.json']; stopping …; claim kept" |
| Whole unit stopped | unit inactive **34 s** after the kill (exit 143); 0 processes matching `w5-bs14p-r2` left (driver, lane 1 and all search workers gone) |
| No automatic restart | `NRestarts=0`, `Restart=no`; the unit stays `failed` |
| Claim and evidence preserved | started + 2 window claims still present; staging empty; both lane dirs (`benchmark.json`, `stdout.log`) kept |
| Watchdog exit status | 1 after stopping for a failure (r1: 0 after a normal end), so it is distinguishable |
| Checker flags the run | `bs_rehearsal_check.py r2` exit 1: windows, lane reports, battles, exit codes, lane logs and watchdog all fail as expected |

The 4 tracebacks in the lane-0 log are `BrokenPipeError` from its 4 search worker processes, which lost their parent pipe when the lane was killed. This is an expected consequence of the injection, and the checker counts them as failures.

## Full test suite (integrator)

On the committed infrastructure (`d514078`), after 11 lint fixes (8 ruff autofixes: unused import/noqa, `!=`, unused unpack; 3 explicit `check=False`), with no behaviour change:

| Run (2026-10-19, UTC+7) | Result |
|---|---|
| `uv run ruff check src scripts tests` | clean |
| `pytest -m "not server"` (resource wrapper, `OMP_NUM_THREADS=1`) | **481 passed** |
| `pytest -m server` | **4 passed** |

## Gate summary

| Gate | Result |
|---|---|
| I1 concurrency (barrier reproduces BS1.4 on 432c343 and passes on new code; 100-round × 6-process stress) | PASS |
| I2 crash safety (before/after link, orphan detected and never deleted; strict verifier 9 refusal cases) | PASS |
| I3 full driver rehearsal (4 blocks × 2 lanes, `final: true`, verifier + watchdog, 96 battles) | PASS |
| I4 failure handling (lane kill ⇒ whole unit stopped in 34 s, no restart, evidence kept) | PASS |
| I5 behaviour isolation (`src/` identical to RC4) | PASS |
