# W5.BS1.4 preflight (preparation only; NO battle, rollout, win or loss ran or was viewed on the new pool)

Base: HEAD 5031f36 (tag `w5-bs-rc4`, code 17e609a). Tooling only; `src/pokeai/**` untouched. Machine-readable copy: `reports/w5_bs1_4_preflight.json`.

## P1 fresh heldout pool (`data/teams_w5_bs14/heldout`)
Built by `scripts/build_teamset.py --w5-fresh-heldout --output data/teams_w5_bs14/heldout --exclude-dirs data/teams/gen9ou data/teams_w5/collect/gen9ou data/teams_w5/dev/gen9ou data/teams_w5/heldout/gen9ou --seed 20261901`. The path builds only the new pool (collect/dev/heldout are not rebuilt) and refuses an existing output directory. Source `hl_05_26`: 43,764 teams, one seeded shuffle, first 200 valid unique disjoint teams. Server validator: `validate-team gen9ou` at Showdown commit d62d3a3.

Exclusions, all loaded from disk and required non-empty (files):

| pool | files |
|---|---|
| `data/teams/gen9ou` (W4) | 211 |
| `data/teams_w5/collect/gen9ou` | 1000 |
| `data/teams_w5/dev/gen9ou` | 200 |
| `data/teams_w5/heldout/gen9ou` (spent BS1.3) | 200 |
| competitive set | 16 |
| old W4 sampled selection (seed 20261016) | 200 |
| W4 pool files as in `build_w5_pools` | 211 |

Discovery found no other team directory under `data/` and none referenced by `configs/*.yaml` beyond the four above (`pool`/`teams` keys: dev, collect, heldout).

Draw: 206 candidates, 6 rejected as `excluded:ordered_sha256`, 0 canonical-only exclusions, 0 duplicates (ordered or canonical), 0 validator rejections, 200 valid.

Disjointness is re-checked from the files on disk after writing. Hashes: (1) `normalized_sha256` (ordered packed team), (2) `canonical_team_sha256` (sorted multiset of per-Pokemon normalized set hashes, so permuted duplicates are caught; a test shows a permuted team differs under (1) and matches under (2)).

| versus | ordered intersection | canonical intersection |
|---|---|---|
| each of the 7 pools above | 0 | 0 |
| within the new pool | 200 distinct of 200 | 200 distinct of 200 |

The new pool holds no `final.started.json`, no window files and no `final.lock` (directory listing: `gen9ou` only).

## P2 `configs/w5_bs14.yaml`
Diff versus `configs/w5_bs_rc1.yaml` (everything else identical, incl. search seed 20261101, D 5.0 / M 0.3 / min 0.5, mixture, 8 worlds x 8 seeds, workers 4, lanes 2, blocks 12/13/12/13, 16 h, `final: true`, 100 battles/arm/opponent):
```
1,4,5  header comments only
run.release_manifest: reports/w5_bs_rc1_manifest.json -> reports/w5_bs14_manifest.json
run.seed:             20261501 -> 20261911
run.out_prefix:       w5-bs-rc1 -> w5-bs14
pool:                 data/teams_w5/heldout -> data/teams_w5_bs14/heldout
fail_dir:             runs/w5-bs-rc1-new-class -> runs/w5-bs14-new-class
```

## P3 bundle manifest and strict preflight
`reports/w5_bs14_manifest.json` ("BS1.4 evaluation bundle on RC4 code 17e609a") is made by `scripts/bs14_manifest.py`. It carries everything the RC manifest hashed: C checkpoint 5, preview 1, tokenizer 6, belief data 92, live-path modules 21, scripts 10 (adds `bs14_manifest.py`, `build_teamset.py`), `sim_driver.js`, Showdown, uv.lock, versions. It also carries the config path and sha256, the pool path, the `gen9ou/manifest.json` sha256 and 201 file hashes (200 teams + manifest).

`verify_release(config, config_path, pool=None)` now runs every RC check plus:
* the passed config file must equal the frozen path, sha256 and parsed content (in-memory config too);
* the config `pool` and any CLI `--pool` must equal the frozen pool;
* the pool's files must match exactly (missing, changed or extra file refused; team manifest sha256; unexpected top-level entries refused apart from `final.*`).

The driver calls it before any lane; each lane calls it before `claim_pool`.

Evidence on the real bundle: the only problem is "manifest was generated from a dirty tree" (the working tree is uncommitted). The integrator must commit, then regenerate with `scripts/bs14_manifest.py` and re-verify. Refusals are tested in `tests/test_bs14_prep.py` on temporary pools/configs: changed config, another config file, config pointing to another pool, CLI pool override, in-memory config differing from the file, edited / added / removed team file, edited team manifest, lane before claim (claim not called), driver before any Popen. Each refused with no claim file. Tests: `tests/test_bs14_prep.py` + `tests/test_bs_rc1_claim.py` = 29 passed (existing tests only gained the new `config_path` argument).

## Full test suite
Integrator runs (resource wrapper, `OMP_NUM_THREADS=1`):

| Code state | `pytest -m "not server"` | `pytest -m server` | `ruff check src scripts tests` |
|---|---|---|---|
| RC4 fix, before commit 17e609a (after 4 lint fixes in `tests/test_bs1_3p_fix.py`) | 442 passed | 4 passed | clean |
| BS1.4 preparation (this bundle; after 10 lint autofixes in `scripts/bs14_manifest.py` and `tests/test_bs14_prep.py`, no behaviour change), run 2026-10-18 23:38 UTC+7 | 456 passed | 4 passed | clean |

The RC4 QA report's "83 targeted tests; full suite not run" was the subagent's statement before the integrator's full run; an addendum there now points here.

## Final bundle verification (integrator, clean tree)

* Code / config / pool commit: `6f9e4dfadf896c7bb97f3673d5565091192e1bf9`; `src/` identical to tag `w5-bs-rc4` (code 17e609a), RC4 tag unchanged.
* `reports/w5_bs14_manifest.json` regenerated on the clean tree (manifest `git.commit` = 6f9e4df, no uncommitted tracked files); team pool section freezes 201 files (200 teams + `gen9ou/manifest.json`).
* `verify_release(config, "configs/w5_bs14.yaml")`, run in a process that imported `evaluate_w5` and `bs1_2_run` (the driver/lane context): **PASS, 0 problems**.
* `data/teams_w5_bs14/heldout/` contains only `gen9ou/`: no `final.started.json`, no window files, no `final.lock`.
* Bundle tag: `w5-bs14-eval` (on the manifest commit). BS1.4 is NOT approved to run.
