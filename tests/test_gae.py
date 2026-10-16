import numpy as np
import pytest

from pokeai.training.gae import gae_advantages


def test_lambda_one_is_exact_monte_carlo():
    rng = np.random.default_rng(0)
    for length in (1, 2, 7, 60):
        values = rng.uniform(0.05, 0.95, length)
        for outcome in (0.0, 1.0):
            advantages = gae_advantages(values, outcome, 1.0)
            np.testing.assert_allclose(advantages, outcome - values, rtol=0, atol=1e-12)


def test_terminal_step_uses_zero_next_value():
    advantages = gae_advantages(np.array([0.4, 0.7]), 1.0, 0.0)
    # lambda = 0 is one-step TD: delta_0 = V1 - V0, delta_T = outcome - V_T.
    np.testing.assert_allclose(advantages, [0.3, 0.3], atol=1e-12)


def test_lambda_discounts_terminal_residual():
    values = np.full(31, 0.5)
    advantages = gae_advantages(values, 1.0, 0.95)
    # Constant V: only the terminal residual (0.5) is non-zero, decayed by lambda^k.
    np.testing.assert_allclose(advantages[0], 0.5 * 0.95**30, atol=1e-12)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        gae_advantages(np.array([0.5]), 1.0, 1.5)
    with pytest.raises(ValueError):
        gae_advantages(np.array([0.5, np.nan]), 1.0, 0.9)
