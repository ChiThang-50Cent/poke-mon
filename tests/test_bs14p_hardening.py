"""BS1.4-P: one lock protocol for verify + claim, staging outside the pool, crash safety, watchdog.

Pools/configs/bundles are temporary; no battle runs. The barrier test reproduces the exact BS1.4
interleaving and is shown to FAIL on the 432c343 implementation (`test_the_barrier_scenario_...`)."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import bs14_manifest
import bs_watchdog
import pool_claim
from bs14_bundle_helper import make_bundle
from pool_claim import (
    claim_unlocked,
    pool_lock,
    staging_dir,
    verify_and_claim,
    verify_only,
)

CTX = multiprocessing.get_context("fork")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(bs14_manifest, "ROOT", tmp_path)
    monkeypatch.setattr(bs14_manifest.M, "ROOT", tmp_path)
    monkeypatch.setattr(bs14_manifest.M, "verify", lambda path: [])
    pool, cfg, config = make_bundle(tmp_path, bs14_manifest)
    return pool, cfg, config


def _lane(pool, cfg, config, window, out, hook=None):
    """A lane process: the production critical section; `hook` replaces the test-only step."""
    if hook is not None:
        pool_claim._claim_step = hook
    try:
        verify_and_claim(pool, config, cfg, config, window)
        out.put((window, "ok", ""))
    except Exception as exc:  # noqa: BLE001 - the test classifies
        out.put((window, type(exc).__name__, str(exc)[:300]))


def _driver(pool, cfg, config, out):
    try:
        verify_only(pool, config, cfg)
        out.put((None, "ok", ""))
    except Exception as exc:  # noqa: BLE001
        out.put((None, type(exc).__name__, str(exc)[:300]))


def _claim_files(pool: Path):
    return sorted(p.name for p in pool.iterdir() if p.name.startswith("final.") and p.name != "final.lock")


# ---- I1 (a): the exact BS1.4 interleaving ---------------------------------------------------

def test_barrier_verify_waits_for_an_inflight_claim_and_is_not_refused(bundle):
    pool, cfg, config = bundle
    staged, release, b_started = CTX.Event(), CTX.Event(), CTX.Event()
    out_a, out_b = CTX.Queue(), CTX.Queue()

    def hold(stage, path):  # A: staging temp exists, before the link
        if stage == "staged":
            staged.set()
            assert release.wait(30)

    def b_proc():
        b_started.set()
        _lane(pool, cfg, config, (1, 2, 12, 25), out_b)

    a = CTX.Process(target=_lane, args=(pool, cfg, config, (0, 2, 12, 25), out_a, hold))
    a.start()
    assert staged.wait(30)
    assert len(list(staging_dir(pool).iterdir())) == 1  # the in-flight temp is OUTSIDE the pool
    assert _claim_files(pool) == []                     # ... and nothing is claimed yet
    b = CTX.Process(target=b_proc)
    b.start()
    assert b_started.wait(30)
    time.sleep(1.0)
    assert b.is_alive() and out_b.empty()               # B blocks on the lock
    release.set()
    a.join(30)
    b.join(30)
    assert out_a.get(timeout=5)[1] == "ok"
    assert out_b.get(timeout=5)[1] == "ok"              # 0 false refusals; B claims its window
    assert _claim_files(pool) == ["final.started.json", "final.window.0of2.12-25.json",
                                  "final.window.1of2.12-25.json"]
    assert list(staging_dir(pool).iterdir()) == []


OLD_SCENARIO = r'''
import json, multiprocessing as mp, os, sys, time
scripts, root, tests = sys.argv[1:4]
sys.path[:0] = [scripts, tests]
from pathlib import Path
root = Path(root)
import bs14_manifest as bm, evaluate_w5 as ev
from bs14_bundle_helper import make_bundle
bm.ROOT = root; bm.M.ROOT = root; bm.M.verify = lambda p: []
pool, cfg, config = make_bundle(root, bm)
ctx = mp.get_context("fork")
staged, release, q = ctx.Event(), ctx.Event(), ctx.Queue()
real_link = os.link
def a_proc():
    def slow_link(src, dst, *a, **k):          # claim temp exists in the pool root, link not done
        staged.set(); release.wait(30); return real_link(src, dst, *a, **k)
    os.link = slow_link
    ev.verify_release(config, cfg)
    ev.claim_pool(pool, True, config, (0, 2, 12, 25))
def b_proc():
    try:
        ev.verify_release(config, cfg)
        q.put("ok")
    except Exception as exc:
        q.put(type(exc).__name__ + ": " + str(exc))
a = ctx.Process(target=a_proc); a.start(); staged.wait(30)
b = ctx.Process(target=b_proc); b.start(); b.join(30)
release.set(); a.join(30)
print(json.dumps({"B": q.get(timeout=5)}))
'''


def test_the_barrier_scenario_fails_on_the_432c343_implementation(tmp_path):
    """Equivalent check: run the SAME interleaving against the 432c343 scripts (git archive)."""
    old = tmp_path / "old"
    old.mkdir()
    archive = subprocess.run(["git", "archive", "432c343", "scripts"], cwd=ROOT, check=True,
                             capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(old)], input=archive, check=True)
    root = tmp_path / "bundle_root"
    root.mkdir()
    script = tmp_path / "scenario.py"
    script.write_text(OLD_SCENARIO)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "METAMON_CACHE_DIR": os.environ.get("METAMON_CACHE_DIR", str(ROOT / ".cache/metamon"))}
    result = subprocess.run(
        [sys.executable, str(script), str(old / "scripts"), str(root), str(Path(__file__).parent)],
        check=False, capture_output=True, text=True, env=env, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    outcome = json.loads(result.stdout.strip().splitlines()[-1])["B"]
    assert outcome.startswith("RuntimeError") and "unexpected entry .final.started.json." in outcome


# ---- I1 (b): stress ---------------------------------------------------------------------------

def _stress_child(kind, pool, cfg, config, window, barrier, out, seed):
    rng = random.Random(seed)

    def jitter(stage, path):
        time.sleep(rng.random() * 0.003)

    pool_claim._claim_step = jitter
    barrier.wait(30)
    if kind == "driver":
        _driver(pool, cfg, config, out)
    else:
        _lane(pool, cfg, config, window, out)


@pytest.mark.parametrize("batch", range(5))
def test_stress_20_rounds_per_batch_5_processes(tmp_path, monkeypatch, batch):
    """5 batches x 20 rounds = 100 rounds x 6 processes: 2 driver verifies + 4 lane claims
    (start + 3 windows, one window attempted twice)."""
    monkeypatch.setattr(bs14_manifest.M, "verify", lambda path: [])
    windows = [(0, 2, 0, 12), (1, 2, 0, 12), (0, 2, 12, 25), (1, 2, 0, 12)]
    for rnd in range(20):
        root = tmp_path / f"r{rnd}"
        root.mkdir()
        monkeypatch.setattr(bs14_manifest, "ROOT", root)
        monkeypatch.setattr(bs14_manifest.M, "ROOT", root)
        pool, cfg, config = make_bundle(root, bs14_manifest)
        out, barrier = CTX.Queue(), CTX.Barrier(6)
        kinds = [("driver", None), ("driver", None)] + [("lane", w) for w in windows]
        procs = [CTX.Process(target=_stress_child,
                             args=(k, pool, cfg, config, w, barrier, out, batch * 1000 + rnd * 10 + i))
                 for i, (k, w) in enumerate(kinds)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(60)
            assert p.exitcode == 0
        results = [out.get(timeout=5) for _ in procs]
        by_class = [r[1] for r in results]
        assert "RuntimeError" not in by_class, results            # 0 false verify refusals
        assert [r for r in results if r[0] is None] == [(None, "ok", "")] * 2
        lane_results = [r for r in results if r[0] is not None]
        dup = [r[1] for r in lane_results if r[0] == (1, 2, 0, 12)]
        assert sorted(dup) == ["FileExistsError", "ok"]           # duplicate window refused once
        assert sorted(r[1] for r in lane_results if r[0] != (1, 2, 0, 12)) == ["ok", "ok"]
        names = _claim_files(pool)
        assert names == ["final.started.json", "final.window.0of2.0-12.json",
                         "final.window.0of2.12-25.json", "final.window.1of2.0-12.json"]
        for name in names:                                          # complete, valid JSON
            assert json.loads((pool / name).read_text())["config"] == config
        assert list(staging_dir(pool).iterdir()) == []


# ---- I2: crash safety ------------------------------------------------------------------------

def _crash_at(stage_name):
    def hook(stage, path):
        if stage == stage_name:
            os._exit(9)
    return hook


def _run_crashing_lane(pool, cfg, config, stage):
    p = CTX.Process(target=_lane, args=(pool, cfg, config, (0, 2, 0, 12), CTX.Queue(),
                                        _crash_at(stage)))
    p.start()
    p.join(30)
    assert p.exitcode == 9


def test_crash_before_the_link_leaves_no_claim_and_an_orphan_that_is_refused(bundle):
    pool, cfg, config = bundle
    _run_crashing_lane(pool, cfg, config, "staged")
    assert _claim_files(pool) == []
    orphans = list(staging_dir(pool).iterdir())
    assert len(orphans) == 1
    with pytest.raises(RuntimeError, match="orphan claim temp file"):
        verify_only(pool, config, cfg)
    with pytest.raises(RuntimeError, match="orphan claim temp file"):
        verify_and_claim(pool, config, cfg, config, (1, 2, 0, 12))
    assert _claim_files(pool) == []                      # nothing claimed by the refused lane
    assert list(staging_dir(pool).iterdir()) == orphans  # nothing deleted


def test_crash_after_the_link_leaves_a_valid_claim_and_an_orphan_that_is_refused(bundle):
    pool, cfg, config = bundle
    _run_crashing_lane(pool, cfg, config, "linked")
    assert _claim_files(pool) == ["final.started.json"]
    assert json.loads((pool / "final.started.json").read_text())["config"] == config
    orphans = list(staging_dir(pool).iterdir())
    assert len(orphans) == 1
    with pool_lock(pool) as lock, pytest.raises(FileExistsError):  # the pool counts as claimed
        claim_unlocked(lock, pool, True, config, None)
    with pytest.raises(RuntimeError, match="orphan claim temp file"):
        verify_only(pool, config, cfg)
    assert list(staging_dir(pool).iterdir()) == orphans


@pytest.mark.parametrize("damage, message", [
    (lambda p: (p / "stray.txt").write_text("x"), "unexpected entry stray.txt"),
    (lambda p: (p / ".final.started.json.abc").write_text("{}"), r"unexpected entry \.final\.started"),
    (lambda p: (p / ".hidden").write_text("x"), "unexpected entry .hidden"),
    (lambda p: (p / "gen9ou/t1.gen9ou_team").write_text("edited"), "changed .*t1"),
    (lambda p: (p / "final.lock").write_text("x"), "final.lock is not empty"),
    (lambda p: (p / "final.started.json").write_text('{"config": '), "not complete valid JSON"),
    (lambda p: (p / "final.started.json").write_text('{"config": {"battles": 9}}'),
     "config differs"),
    (lambda p: (p / "final.window.0of2.0-12.json").write_text('{"config"'), "not complete valid JSON"),
    (lambda p: (p / "final.window.x.json").write_text("{}"), "unexpected entry final.window.x.json"),
])
def test_the_verifier_stays_strict(bundle, damage, message):
    pool, cfg, config = bundle
    verify_and_claim(pool, config, cfg, config, (0, 2, 0, 12))
    verify_only(pool, config, cfg)                       # a legitimately claimed pool verifies
    damage(pool)
    with pytest.raises(RuntimeError, match=message):
        verify_only(pool, config, cfg)


def test_the_lock_protocol_is_enforced(bundle):
    pool, _cfg, config = bundle
    with pytest.raises(RuntimeError, match="requires the pool lock"):
        claim_unlocked(None, pool, True, config)
    with pool_lock(pool) as lock:
        pass
    with pytest.raises(RuntimeError, match="not held"):
        claim_unlocked(lock, pool, True, config)             # token used after release
    other = pool.parent / "other"
    (other / "gen9ou").mkdir(parents=True)
    (other / "gen9ou/manifest.json").write_text('{"role":"heldout"}')
    with pool_lock(pool) as lock, pytest.raises(RuntimeError, match="not held"):
        claim_unlocked(lock, other, True, config)            # token of another pool
    assert _claim_files(pool) == [] and _claim_files(other) == []


# ---- watchdog -------------------------------------------------------------------------------

def _lane_report(repo, prefix, block, lane, valid):
    d = repo / "runs" / f"{prefix}-b{block}" / f"lane{lane}"
    d.mkdir(parents=True)
    (d / "benchmark.json").write_text(json.dumps({"valid": valid}))
    return str(d.relative_to(repo) / "benchmark.json")


def _watch(tmp_path, cmdlines, active_for=5):
    stops, calls = [], iter(range(active_for))
    code = bs_watchdog.watch(
        "u.service", "wd", tmp_path / "wd.log", repo=tmp_path,
        is_active=lambda unit: next(calls, None) is not None, cmdlines=lambda: cmdlines,
        stop=stops.append, sleep=lambda s: None)
    return code, stops, (tmp_path / "wd.log").read_text()


def test_watchdog_stops_after_two_consecutive_failed_observations(tmp_path):
    _lane_report(tmp_path, "wd", 1, 0, False)            # process gone, not valid
    code, stops, log = _watch(tmp_path, [])
    assert code == 1 and stops == ["u.service"] and "LANE FAILURE" in log


def test_watchdog_ignores_running_valid_and_single_observations(tmp_path):
    rel = _lane_report(tmp_path, "wd", 1, 0, False)
    code, stops, log = _watch(tmp_path, [f"python evaluate_w5.py --out {rel}"])  # still running
    assert (code, stops) == (0, []) and "LANE FAILURE" not in log
    valid = tmp_path / "v"
    valid.mkdir()
    _lane_report(valid, "wd", 1, 1, True)                # valid and finished
    code, stops, _ = _watch(valid, [])
    assert (code, stops) == (0, [])
    one = tmp_path / "one"
    one.mkdir()
    _lane_report(one, "wd", 1, 0, False)
    code, stops, _ = _watch(one, [], active_for=1)       # a single observation never stops
    assert (code, stops) == (0, [])


def test_watchdog_failure_must_persist_between_checks(tmp_path):
    rel = _lane_report(tmp_path, "wd", 1, 0, False)
    seen = iter([[], [f"x evaluate_w5.py --out {rel}"], []])  # gone, running again, gone
    stops, calls = [], iter(range(3))
    code = bs_watchdog.watch(
        "u.service", "wd", tmp_path / "wd.log", repo=tmp_path,
        is_active=lambda unit: next(calls, None) is not None, cmdlines=lambda: next(seen),
        stop=stops.append, sleep=lambda s: None)
    assert (code, stops) == (0, [])


# ---- rehearsal tooling ------------------------------------------------------------------------

def _fake_root(tmp_path):
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/w5_bs14.yaml", root / "configs/w5_bs14.yaml")
    dev = root / "data/teams_w5/dev/gen9ou"
    dev.mkdir(parents=True)
    for i in range(5):
        (dev / f"hl-{i}.gen9ou_team").write_text(f"team {i}")
    (dev / "manifest.json").write_text(json.dumps({"role": "dev", "teams": []}))
    return root


def test_rehearsal_prep_changes_only_the_allowed_config_keys(tmp_path):
    import bs_rehearsal_prep as prep
    import yaml

    root = _fake_root(tmp_path)
    cfg = prep.prepare("t1", root, make_manifest=False)
    new = yaml.safe_load(cfg.read_text())
    base = yaml.safe_load((root / "configs/w5_bs14.yaml").read_text())
    new_run, base_run = new.pop("run"), base.pop("run")
    new["arms"]["C+search"]["search"].pop("fail_dir")
    base["arms"]["C+search"]["search"].pop("fail_dir")
    assert new.pop("pool") == "runs/rehearsal/t1/heldout" and base.pop("pool")
    assert new.pop("battles") == 8 and base.pop("battles") == 100
    assert new == base                                              # nothing else differs
    for key in ("out_prefix", "release_manifest", "block_pairs", "seed"):
        new_run.pop(key), base_run.pop(key)
    assert new_run == base_run                                      # final/lanes/budget kept
    manifest = json.loads((root / "runs/rehearsal/t1/heldout/gen9ou/manifest.json").read_text())
    assert manifest["role"] == "heldout" and manifest["rehearsal"] is True
    with pytest.raises(FileExistsError):
        prep.prepare("t1", root, make_manifest=False)


def test_a_one_pair_block_leaves_one_lane_without_a_pair():
    """Lane split of the contract layout (1x4): pair k goes to lane k % 2."""
    from pokeai.env.runner import lane_indices

    split = [[len(lane_indices(8, [lane, 2], [a, b])) for lane in range(2)]
             for a, b in [[0, 1], [1, 2], [2, 3], [3, 4]]]
    assert split == [[2, 0], [0, 2], [2, 0], [0, 2]]               # battles per lane per cell
    split2 = [[len(lane_indices(16, [lane, 2], [a, b])) for lane in range(2)]
              for a, b in [[0, 2], [2, 4], [4, 6], [6, 8]]]
    assert split2 == [[2, 2]] * 4


def test_rehearsal_check_reads_operational_data_only(tmp_path):
    import bs_rehearsal_check as chk
    import bs_rehearsal_prep as prep
    import yaml

    root = _fake_root(tmp_path)
    cfg = yaml.safe_load(prep.prepare("t2", root, make_manifest=False).read_text())
    prefix, pool = cfg["run"]["out_prefix"], root / cfg["pool"]
    (pool / "final.started.json").write_text('{"config": {}}')
    for a, b in cfg["run"]["block_pairs"]:
        for lane in range(2):
            (pool / f"final.window.{lane}of2.{a}-{b}.json").write_text('{"config": {}}')
    projections = []
    for k in range(1, 5):
        for lane in range(2):
            d = root / "runs" / f"{prefix}-b{k}" / f"lane{lane}"
            d.mkdir(parents=True)
            (d / "stdout.log").write_text("fine\n")
            opponents = {}
            for opp in cfg["opponents"]:
                opponents[opp] = {}
                for arm in cfg["arms"]:
                    log = d / f"{opp}-{arm}.jsonl"
                    n = 2 if lane == (k - 1) % 2 else 0
                    log.write_text("".join(json.dumps({"error": None}) + "\n" for _ in range(n)))
                    opponents[opp][arm] = {"battles_path": str(log)}
            (d / "benchmark.json").write_text(json.dumps({"valid": True, "opponents": opponents}))
        projections.append({"block": k, "exit_codes": [0, 0]})
    (root / "runs" / f"{prefix}-projections.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in projections))
    wd = root / "runs" / f"{prefix}-watchdog.log"
    wd.write_text("watchdog start\n")
    result = chk.check("t2", root)
    assert result["ok"], json.dumps(result, indent=1)
    assert result["checks"]["battles"]["found"] == 48 == result["checks"]["battles"]["expected"]
    assert result["checks"]["windows"]["present"] == 8
    assert "winner" not in json.dumps(result) and "win_rate" not in json.dumps(result)
    (pool / "final.window.0of2.9-10.json").write_text("{}")              # extra window
    wd.write_text("LANE FAILURE confirmed\n")
    (staging_dir(pool)).mkdir()
    (staging_dir(pool) / "orphan").write_text("x")
    bad = chk.check("t2", root)
    assert not bad["ok"]
    assert bad["checks"]["windows"]["extra"] == ["final.window.0of2.9-10.json"]
    assert not bad["checks"]["watchdog_log"]["ok"] and not bad["checks"]["no_orphan_claim_temp"]["ok"]
