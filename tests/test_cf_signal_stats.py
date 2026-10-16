"""CF0.4 statistics: permutation null keeps pairing; Spearman/binomial exactness."""

import math

import numpy as np

from pokeai.cf.signal_stats import (
    binom_sf,
    permute_within_k,
    rankdata,
    root_and_global_tests,
    spearman,
    split_half_stat,
    statistic_T,
)


def test_rankdata_ties_and_spearman():
    assert list(rankdata(np.array([3.0, 1.0, 1.0, 2.0]))) == [4.0, 1.5, 1.5, 3.0]
    assert spearman(np.array([1, 2, 3.0]), np.array([10, 20, 30.0])) == 1.0
    assert spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == -1.0
    assert math.isnan(spearman(np.array([1, 1, 1.0]), np.array([1, 2, 3.0])))


def test_binom_sf_exact():
    assert abs(binom_sf(0, 10, 0.05) - 1.0) < 1e-12
    assert abs(binom_sf(10, 10, 0.5) - 0.5**10) < 1e-15
    assert abs(binom_sf(1, 20, 0.05) - (1 - 0.95**20)) < 1e-12


def test_permutation_preserves_columns_multiset():
    rng = np.random.default_rng(0)
    q = rng.integers(0, 2, size=(4, 16)).astype(float)
    permuted = permute_within_k(q, rng)
    assert np.array_equal(np.sort(q, axis=0), np.sort(permuted, axis=0))


def test_global_test_detects_signal_and_not_noise():
    rng = np.random.default_rng(1)
    signal = []
    for _ in range(20):
        q = (rng.random((3, 16)) < np.array([[0.2], [0.5], [0.8]])).astype(float)
        signal.append(q)
    assert root_and_global_tests(signal, 300, 5)["p_global"] < 0.01
    noise = [(rng.random((3, 16)) < 0.5).astype(float) for _ in range(20)]
    result = root_and_global_tests(noise, 300, 5)
    assert result["p_global"] > 0.05
    assert abs(result["T"].mean() - result["T_global_null_mean"] / 20) < 0.02


def test_split_half_stat_known_cases():
    stable = np.array([[0.0] * 16, [1.0, 0.0] * 8, [1.0] * 16])
    assert split_half_stat(stable) > 0.4
    two = np.array([[1.0] * 16, [0.0] * 16])
    assert split_half_stat(two) == 1.0
    assert statistic_T(np.ones((3, 4))) == 0.0
