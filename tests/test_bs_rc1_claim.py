"""BS1.2b: the one-shot heldout claim accepts the windows (lane/block) of ONE frozen config."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_w5 import claim_pool


@pytest.fixture
def pool(tmp_path):
    (tmp_path / "gen9ou").mkdir()
    (tmp_path / "gen9ou/manifest.json").write_text(json.dumps({"role": "heldout"}))
    return tmp_path


CONFIG = {"battles": 100, "seed": 1}


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
