# W5.BS1.5 preflight (preparation only: NO battle, rollout, win or loss ran on or was viewed for the new pool)

BS1.5 is the **last planned heldout confirmation in W5**. It follows two confirmations that were aborted for technical reasons:
- BS1.3 (`reports/w5_bs1_3_incident.md`): harness stall, fixed in RC4 using the forensic payload only;
- BS1.4 (`reports/w5_bs1_4_incident.md`): a verify/claim race, fixed in BS1.4-P (`reports/w5_bs1_4p_hardening.md`, I1–I5 PASS).

No BS1.3 or BS1.4 win/loss was computed or used for any fix. If BS1.5 is aborted, W5 ends as "C+Search: dev-validated, independent confirmation unavailable", and BS1.6 is not requested automatically.

This report and `w5_bs1_5_preflight.json` were written **after** the manifest was generated on a clean tree. They are the final preflight state, not a pre-commit snapshot. (The BS1.4 `w5_bs1_4_preflight.json` was a pre-commit preparation snapshot; its final verify is recorded in that report's Markdown.)

## Release, infrastructure and bundle

| Item | Value |
|---|---|
| Search release | `w5-bs-rc4` (`5031f36`, code `17e609a`), unchanged |
| Infrastructure | BS1.4-P `d514078` (rehearsed: I3 r1 and I4 r2 ran on it); `git diff d514078 HEAD -- scripts tests` is empty |
| Pool + config commit | `3ca86360d37a5d544be3d443ff38c9b5cccc106e` |
| Bundle manifest | `reports/w5_bs15_manifest.json`, sha256 `372e6ec85e8b5adb968ba78d46375d3cefc1517fb139562aab47df6760226473`, generated at `3ca8636` with 0 uncommitted tracked files |
| Config | `configs/w5_bs15.yaml`, sha256 `5cd3254dd05c98b18e09d2c8adec0e2892fc581ad2256c8aec9a1c488e2e213e` |
| Pool | `data/teams_w5_bs15/heldout`; team `manifest.json` sha256 `81173b097da41295c8652349c3aeb0b449205d308c810cb380afed8ddd5eb36e`; 201 files frozen (200 teams + manifest) |
| Watchdog (frozen, in manifest) | `scripts/bs_watchdog.py`, sha256 `349da8f2121d6edec32d07efb8320ab5cbd2541c39fd10c9c0bad0559b1e4318` |
| Bundle tag | `w5-bs15-eval` (on the commit adding this report and the manifest) |

## RC4 decision path unchanged

- `git diff --name-only w5-bs-rc4 HEAD -- src/`: 0 files.
- Manifest sections identical to the BS1.4 bundle (itself RC4): `c_checkpoint`, `preview_checkpoint`, `tokenizer`, `belief_sampler_data` (92 files), `code_live_path` (21 modules), `sim_driver_js`, `showdown`, `versions`.
- Only infrastructure scripts differ from the BS1.4 bundle: `bs14_manifest.py`, `bs1_2_run.py`, `evaluate_w5.py`, the new `pool_claim.py` and the new `bs_watchdog.py`. All are covered by I1–I5.

## P1: fresh pool

Command: `build_teamset.py --w5-fresh-heldout --output data/teams_w5_bs15/heldout --exclude-dirs data/teams/gen9ou data/teams_w5/collect/gen9ou data/teams_w5/dev/gen9ou data/teams_w5/heldout/gen9ou data/teams_w5_bs14/heldout/gen9ou --seed 20262101`.

Source `hl_05_26` (43,764 teams). 209 candidates were drawn and 9 rejected as already excluded (ordered hash), leaving 200 that passed the server validator (Showdown `d62d3a39`).

Intersections, recomputed independently by the integrator from the files on disk:

| Pool | Files | Ordered `normalized_sha256` | Order-independent canonical |
|---|---|---|---|
| `data/teams/gen9ou` (W4) | 211 | 0 | 0 |
| `data/teams_w5/collect/gen9ou` | 1000 | 0 | 0 |
| `data/teams_w5/dev/gen9ou` (also the source of every rehearsal pool) | 200 | 0 | 0 |
| `data/teams_w5/heldout/gen9ou` (spent, BS1.3) | 200 | 0 | 0 |
| `data/teams_w5_bs14/heldout/gen9ou` (spent, BS1.4) | 200 | 0 | 0 |
| competitive / old W4 selection (builder's exclusion set) | 16 / 200 | 0 | 0 |
| within the new pool | 200 | 200 distinct | 200 distinct |

## P2: frozen config

Diff against `configs/w5_bs14.yaml`, parsed keys: `pool`, `run.seed` (20261911 → **20262111**), `run.out_prefix` (`w5-bs15`), `run.release_manifest` (`reports/w5_bs15_manifest.json`), `arms.C+search.search.fail_dir` (`runs/w5-bs15-new-class`), plus 3 header comment lines.

Unchanged:
- search seed 20261101;
- D 5.0 s, margin 0.3 s, min budget 0.5 s;
- mixture opponent model, 8 worlds × 8 seeds, 4 workers;
- 2 lanes, blocks 12/13/12/13, budget 16 h;
- `final: true`, 100 battles per arm per opponent;
- arms C / C+search, opponents Grunt / GymLeader / C.

The design is 600 battles.

## P3: verification

- `verify_bundle(reports/w5_bs15_manifest.json, configs/w5_bs15.yaml, config, pool)`: **0 problems**. It was called read-only, without `pool_lock`, so no `final.lock` was created; no claimer was running.
- At launch, the driver re-verifies under `pool_lock` before every block, and each lane verifies + claims in one critical section.
- Refusal tests (changed config, other config/pool, edited/added/removed team, orphan, stray file, invalid claim JSON) are in `tests/test_bs14_prep.py` and `tests/test_bs14p_hardening.py`. The suite on `d514078` passed: 481 offline, 4 server, ruff clean.
- New pool root: `gen9ou/` only. No `final.started.json`, no `final.window.*.json`, no `final.lock`, no staging dir.

## Launch plan (requires explicit approval)

Read-only preflight at `w5-bs15-eval`:
- clean tree;
- HEAD = tag;
- `src/` = RC4;
- no claim and no windows in the new pool.

Then:
- **Main unit:** `systemd-run --user --unit=w5-bs15 … -p CPUQuota=1000% -p MemoryMax=14G -p Nice=10 -p Restart=no … uv run python scripts/bs1_2_run.py --config configs/w5_bs15.yaml`.
- **Watchdog unit:** `w5-bs15-watch` running `scripts/bs_watchdog.py --unit w5-bs15.service --prefix w5-bs15 --log runs/w5-bs15-watchdog.log`.

No win/loss is read between blocks. Stop rule: technical error or budget. On any abort, keep the claim and evidence, and do not rerun.
