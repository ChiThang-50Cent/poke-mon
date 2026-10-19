import copy
import math

import pytest

from pokeai.data.search_capture import override_z, trajectory_errors


def capture():
    """Two learner steps: a C-kept move, then a search override 0 -> 5, plus one fallback."""
    stats = [{"slot": 5, "delta": 0.3, "se": 0.1}, {"slot": 2, "delta": 0.1, "se": 0.1}]
    rows = [
        {"turn": 1, "category": "search_fallback_no_signal", "c_action": 1, "action": 1},
        {"turn": 2, "category": "search_override", "c_action": 0, "action": 5, "stats": stats},
        {"turn": 2, "category": "not_represented", "c_action": None, "action": None},
    ]
    behavior = [
        {"chosen_index": 1, "executed_index": 1, "search_row": 0},
        {"chosen_index": 0, "executed_index": 5, "search_row": 1},
        {"chosen_index": -1, "executed_index": -1, "search_row": 2},
    ]
    native = {"actions": [1, 5, -1, -1]}
    meta = {"learner": True, "search_augmented": True, "behavior": behavior,
            "decision_turns": [1, 2, 2]}
    return native, meta, rows


def test_exact_join_passes_and_keeps_step_order():
    native, meta, rows = capture()
    errors, joined = trajectory_errors(native, meta, rows)
    assert errors == []
    assert [r["action"] for r in joined] == [1, 5, None]
    assert override_z(joined[1]) == pytest.approx(3.0)
    assert override_z(joined[0]) is None


@pytest.mark.parametrize("corrupt, message", [
    (lambda n, m, r: n["actions"].__setitem__(1, 0), "native action"),  # pre-W6.0 collector bug
    (lambda n, m, r: m["decision_turns"].__setitem__(1, 3), "row turn"),
    (lambda n, m, r: m["behavior"][1].__setitem__("search_row", 0), "bad search_row"),
    (lambda n, m, r: r.append(dict(r[0])), "not joined"),
    (lambda n, m, r: r[1].__setitem__("category", "search_fallback_snr"), "override category"),
    (lambda n, m, r: m.__setitem__("search_augmented", False), "not marked"),
])
def test_join_mismatch_is_reported(corrupt, message):
    native, meta, rows = capture()
    corrupt(native, meta, rows)
    errors, _ = trajectory_errors(native, meta, rows)
    assert any(message in e for e in errors), errors


def test_opponent_pov_must_not_carry_search_fields():
    native, meta, _ = capture()
    meta = {**copy.deepcopy(meta), "learner": False, "search_augmented": False}
    assert trajectory_errors(native, meta, None)[0] == ["opponent POV carries search fields"]


def test_zero_variance_positive_override_is_confident():
    row = {"category": "search_override", "action": 5,
           "stats": [{"slot": 5, "delta": 0.25, "se": 0.0}]}
    assert override_z(row) == math.inf
