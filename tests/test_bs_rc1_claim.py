"""BS1.2b: the one-shot heldout claim accepts the windows (lane/block) of ONE frozen config."""

import json
import multiprocessing
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_w5
from evaluate_w5 import claim_pool, verify_release


@pytest.fixture
def pool(tmp_path):
    (tmp_path / "gen9ou").mkdir()
    (tmp_path / "gen9ou/manifest.json").write_text(json.dumps({"role": "heldout"}))
    return tmp_path


CONFIG = {"battles": 100, "seed": 1}
# A large claim payload widens the window between creating and finishing the claim file; the
# pre-fix `open("x")` + `json.dump` claim lost a lane to JSONDecodeError in 30/30 races with it.
BIG = {**CONFIG, "pad": "x" * 200_000}


def test_unwindowed_claim_is_one_shot(pool):
    claim_pool(pool, True, CONFIG)
    with pytest.raises(FileExistsError):
        claim_pool(pool, True, CONFIG)


def test_windows_of_one_config_are_accepted_once_each(pool):
    claim_pool(pool, True, CONFIG, (0, 2, 0, 12))
    claim_pool(pool, True, CONFIG, (1, 2, 0, 12))
    claim_pool(pool, True, CONFIG, (0, 2, 12, 25))
    with pytest.raises(FileExistsError):
        claim_pool(pool, True, CONFIG, (1, 2, 0, 12))  # repeating a window is refused


def test_a_different_config_cannot_join_the_claim(pool):
    claim_pool(pool, True, CONFIG, (0, 2, 0, 12))
    with pytest.raises(FileExistsError):
        claim_pool(pool, True, {**CONFIG, "seed": 2}, (1, 2, 0, 12))
    with pytest.raises(FileExistsError):
        claim_pool(pool, True, CONFIG)  # an unwindowed run cannot reuse a claim


def test_heldout_still_requires_final(pool):
    with pytest.raises(ValueError):
        claim_pool(pool, False, CONFIG)


def _claim_in_process(pool, window, barrier, results):
    barrier.wait()  # every process enters claim_pool at the same moment
    try:
        claim_pool(Path(pool), True, BIG, window)
        results.put(("ok", window))
    except Exception as exc:  # noqa: BLE001 - the test inspects the class
        results.put((type(exc).__name__, window))


def _race(pool, windows):
    ctx = multiprocessing.get_context("fork")
    barrier, results = ctx.Barrier(len(windows)), ctx.Queue()
    procs = [ctx.Process(target=_claim_in_process, args=(str(pool), w, barrier, results))
             for w in windows]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    return [results.get(timeout=5) for _ in procs]


@pytest.mark.parametrize("round_", range(5))
def test_concurrent_lanes_all_claim_and_the_claim_is_complete(pool, round_):
    windows = [(lane, 8, 0, 12) for lane in range(8)]
    outcomes = _race(pool, windows)
    assert sorted(o for o, _ in outcomes) == ["ok"] * 8
    assert json.loads((pool / "final.started.json").read_text())["config"] == BIG
    assert len(list(pool.glob("final.window.*.json"))) == 8
    assert not list(pool.glob(".final.*"))  # no temp leftovers


def test_concurrent_unwindowed_claims_have_exactly_one_winner(pool):
    outcomes = _race(pool, [None] * 8)
    assert sorted(o for o, _ in outcomes) == ["FileExistsError"] * 7 + ["ok"]


def test_a_crash_while_writing_leaves_no_claim(pool, monkeypatch):
    def crash(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(evaluate_w5.json, "dump", crash)
    with pytest.raises(OSError):
        claim_pool(pool, True, CONFIG, (0, 2, 0, 12))
    assert not (pool / "final.started.json").exists()
    assert not list(pool.glob(".final.*"))


def test_final_requires_a_release_manifest():
    with pytest.raises(ValueError, match="release_manifest"):
        verify_release({"run": {"final": True}})


def test_a_changed_frozen_file_blocks_the_release(tmp_path):
    manifest = json.loads((ROOT / "reports/w5_bs_rc1_manifest.json").read_text())
    name = next(iter(manifest["code_live_path"]))
    manifest["code_live_path"][name] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=f"changed {name}"):
        verify_release({"run": {"release_manifest": str(path)}})


def test_the_driver_verifies_before_launching_any_lane(tmp_path, monkeypatch):
    import bs1_2_run

    config = tmp_path / "rc.yaml"
    config.write_text("run: {final: true}\nbattles: 2\n")
    launched = []
    monkeypatch.setattr(bs1_2_run.subprocess, "Popen", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(sys, "argv", ["bs1_2_run.py", "--config", str(config)])
    with pytest.raises(ValueError, match="release_manifest"):
        bs1_2_run.main()
    assert launched == []
    assert not list(tmp_path.glob("*.json"))


def test_frozen_live_closure_includes_the_entrypoints_whatever_the_caller_imported():
    """BS1.3 pre-claim abort: the driver imports evaluate_w5 (-> env.collector, repro); a module
    list taken from the caller's sys.modules flagged them as unfrozen. The closure is now computed
    in a fresh interpreter and includes the lane/driver entrypoints."""
    from bs_rc1_manifest import live_path_modules

    names = {str(p.relative_to(ROOT)) for p in live_path_modules()}
    assert {"src/pokeai/env/collector.py", "src/pokeai/repro.py",
            "src/pokeai/agents/live_search.py", "src/pokeai/cf/recon_search.py"} <= names
    import pokeai.env.collector  # noqa: F401 - the caller importing more must not change it

    assert {str(p.relative_to(ROOT)) for p in live_path_modules()} == names
