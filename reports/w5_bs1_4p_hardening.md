# W5.BS1.4-P: infrastructure hardening (tooling only)

Base `8922ef9`. Working tree, uncommitted. Nothing under `src/pokeai/` changed: `git diff 8922ef9 -- src/` is empty (I5). The spent pools `data/teams_w5/heldout` and `data/teams_w5_bs14` are unchanged (`git diff 8922ef9` on both is empty). No battle was run.

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
`tests/test_bs14p_hardening.py tests/test_bs14_prep.py tests/test_bs_rc1_claim.py tests/test_w5_benchmark.py`: **58 passed in 39.2 s**. `ruff` is not installed in the environment, so lint was not run.

## I3 rehearsal (integrator)
_(empty placeholder)_

## I4 failure injection (integrator)
_(empty placeholder)_

## Full test suite (integrator)
_(empty placeholder)_
