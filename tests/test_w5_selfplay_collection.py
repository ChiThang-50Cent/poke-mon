"""The collection mixture always allocates the exact requested battle total."""
import importlib.util
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "collect_w5_selfplay", Path(__file__).parents[1] / "scripts/collect_w5_selfplay.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
allocate_battles = _module.allocate_battles


def test_pilot_mix_has_registered_counts():
    config = yaml.safe_load(Path("configs/w5_selfplay.yaml").read_text())
    assert allocate_battles(config["opponents"], 1000) == [400, 100, 100, 100] + [60] * 5


def test_largest_remainder_preserves_total():
    opponents = [{"weight": 0.4}, {"weight": 0.3}, {"weight": 0.3}]
    assert allocate_battles(opponents, 11) == [5, 3, 3]
    assert allocate_battles(opponents, 1) == [1, 0, 0]


@pytest.mark.parametrize("weights,battles", [([0], 1), ([-1, 2], 1), ([1], 0)])
def test_invalid_mixture_rejected(weights, battles):
    with pytest.raises(ValueError):
        allocate_battles([{"weight": weight} for weight in weights], battles)
