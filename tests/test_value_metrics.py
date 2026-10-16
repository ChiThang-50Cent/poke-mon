"""Synthetic value calibration, decision-index slicing, and rating references."""

import json

import numpy as np
import pytest

from pokeai.training.value_metrics import (
    rating_reference,
    trajectory_metrics,
    value_metrics,
)


@pytest.mark.parametrize(
    "probabilities, labels, expected",
    [
        ([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], 0.75),
        ([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1], 0.875),
        ([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 1], 0.5),
        ([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1], 0.0),
        ([0.1, 0.2], [0, 1], 1.0),
    ],
)
def test_auc_average_ranks(probabilities, labels, expected):
    assert value_metrics(probabilities, labels)["roc_auc"] == pytest.approx(expected)
    assert value_metrics(probabilities[::-1], labels[::-1])["roc_auc"] == (
        pytest.approx(expected)
    )


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_one_class_auc_is_undefined(labels):
    assert value_metrics([0.2, 0.8], labels)["roc_auc"] is None


def test_known_losses_and_calibration():
    metrics = value_metrics([0.25, 0.25, 0.75, 0.75], [0, 1, 0, 1])
    assert metrics["count"] == 4
    assert metrics["bce"] == pytest.approx(-0.5 * np.log(0.25 * 0.75))
    assert metrics["brier"] == pytest.approx(0.3125)
    assert metrics["accuracy"] == 0.5
    assert metrics["base_rate"] == metrics["mean_probability"] == 0.5
    assert metrics["ece"] == 0.25
    assert metrics["reliability"][2] == {
        "lower": 0.2, "upper": 0.3, "count": 2,
        "mean_probability": 0.25, "observed_rate": 0.5,
    }
    assert metrics["reliability"][7]["count"] == 2
    assert metrics["reliability"][0]["mean_probability"] is None
    assert metrics["reliability"][0]["observed_rate"] is None


def test_perfect_calibration_and_endpoint_bins():
    probabilities = [0.0] * 4 + [0.25] * 4 + [0.5] * 4 + [1.0] * 4
    labels = [0] * 4 + [0, 0, 0, 1] + [0, 0, 1, 1] + [1] * 4
    metrics = value_metrics(probabilities, labels)
    assert metrics["ece"] == 0.0
    assert metrics["reliability"][0]["count"] == 4
    assert metrics["reliability"][5]["count"] == 4
    assert metrics["reliability"][9]["count"] == 4
    assert np.isfinite(metrics["bce"])
    json.dumps(metrics, allow_nan=False)


def test_bins_are_left_closed_and_one_is_in_final_bin():
    metrics = value_metrics(np.arange(11) / 10, np.zeros(11))
    assert [row["count"] for row in metrics["reliability"]] == [1] * 9 + [2]
    assert metrics["reliability"][-1]["mean_probability"] == pytest.approx(0.95)


def test_empty_metrics_are_explicitly_undefined():
    metrics = value_metrics([], [])
    assert metrics["count"] == 0
    for key in ("bce", "brier", "roc_auc", "accuracy", "mean_probability",
                "base_rate", "ece"):
        assert metrics[key] is None
    assert len(metrics["reliability"]) == 10
    assert all(row["count"] == 0 for row in metrics["reliability"])
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "probabilities, labels",
    [([0.5], []), ([np.nan], [1]), ([np.inf], [1]),
     ([1.1], [1]), ([-0.1], [0]), ([0.5], [0.5]), ([[0.5]], [[1]])],
)
def test_invalid_metric_inputs_raise(probabilities, labels):
    with pytest.raises(ValueError):
        value_metrics(probabilities, labels)


def test_buckets_count_zero_based_decisions_and_restart_each_trajectory():
    # Two trajectories of lengths 33 and 7, plus an empty trajectory.
    probabilities = np.r_[np.arange(33) / 40, np.arange(7) / 10]
    labels = np.r_[np.zeros(33), np.ones(7)]
    metrics = trajectory_metrics(probabilities, labels, [0, 33, 33, 40])
    buckets = metrics["turn_buckets"]
    assert {key: value["count"] for key, value in buckets.items()} == {
        "0-5": 12, "6-15": 11, "16-30": 15, "31+": 2,
    }
    selections = {
        "0-5": np.r_[np.arange(6), np.arange(33, 39)],
        "6-15": np.r_[np.arange(6, 16), 39],
        "16-30": np.arange(16, 31),
        "31+": np.arange(31, 33),
    }
    for name, indices in selections.items():
        assert buckets[name] == value_metrics(probabilities[indices], labels[indices])
    assert metrics["first_step"] == value_metrics([0, 0], [0, 1])
    assert metrics["final_step"] == value_metrics([32 / 40, 0.6], [0, 1])
    assert metrics["count"] == 40


def test_short_and_empty_trajectories_have_empty_late_buckets():
    metrics = trajectory_metrics([0.2], [0], [0, 0, 1, 1])
    assert metrics["first_step"] == metrics["final_step"]
    assert metrics["first_step"]["count"] == 1
    assert metrics["turn_buckets"]["6-15"]["count"] == 0
    empty = trajectory_metrics([], [], [0])
    assert empty["first_step"]["count"] == empty["final_step"]["count"] == 0


@pytest.mark.parametrize("offsets", [[1, 2], [0, 1], [0, 2, 1, 2],
                                    [0.0, 2.0], [], [[0, 2]]])
def test_invalid_offsets_raise(offsets):
    with pytest.raises(ValueError, match="offsets"):
        trajectory_metrics([0.2, 0.8], [0, 1], offsets)


def test_rating_reference_is_deterministic_monotonic_and_train_only():
    ratings = [1000, 1100, 1200, 1300, 1400, 1500]
    outcomes = [0, 0, 0, 1, 1, 1]
    val = [900, 1250, 1600]
    probabilities, metadata = rating_reference(ratings, outcomes, val)
    repeated, repeated_metadata = rating_reference(ratings, outcomes, val)
    np.testing.assert_array_equal(probabilities, repeated)
    assert metadata == repeated_metadata
    assert metadata["converged"]
    assert metadata["slope"] > 0
    assert np.all(np.diff(probabilities) > 0)
    assert probabilities[1] == pytest.approx(0.5)
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert metadata["rating_mean"] == 1250
    _, other_metadata = rating_reference(ratings, outcomes, [-10000, 10000])
    assert metadata == other_metadata
    json.dumps(metadata, allow_nan=False)


def test_constant_ratings_fit_only_intercept():
    probabilities, metadata = rating_reference([1500] * 4, [0, 1, 1, 1], [0, 1500, 9999])
    assert metadata["rating_scale"] == 1
    assert metadata["slope"] == 0
    assert metadata["converged"]
    np.testing.assert_allclose(probabilities, 0.75, atol=0.001)


@pytest.mark.parametrize("outcome", [0, 1])
def test_regularized_one_class_rating_fit_is_finite(outcome):
    probabilities, metadata = rating_reference([1000, 1500], [outcome] * 2, [0, 2000])
    assert metadata["converged"]
    assert np.isfinite(metadata["intercept"])
    assert metadata["regularized_parameters"] == ["intercept", "slope"]
    assert np.all((probabilities > 0) & (probabilities < 1))
    np.testing.assert_allclose(probabilities, probabilities[0])
    assert (probabilities[0] > 0.5) == bool(outcome)


@pytest.mark.parametrize(
    "ratings, outcomes, val",
    [([], [], []), ([1000], [], []), ([np.nan], [0], []),
     ([1000], [0.5], []), ([1000], [0], [np.inf])],
)
def test_rating_reference_invalid_inputs_raise(ratings, outcomes, val):
    with pytest.raises(ValueError):
        rating_reference(ratings, outcomes, val)
