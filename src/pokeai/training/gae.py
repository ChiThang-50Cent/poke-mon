"""GAE(gamma=1, lambda) for terminal-only win/loss rewards on one POV trajectory."""

from __future__ import annotations

import numpy as np


def gae_advantages(values: np.ndarray, outcome: float, lam: float) -> np.ndarray:
    """Advantages for every step of one trajectory (fallback steps included).

    delta_t = r_t + (1 - d_t) V(s_{t+1}) - V(s_t), with r = 0 except r_T = outcome
    at the final step, where d_T = 1 and the next value is 0.
    A_t = delta_t + lam * (1 - d_t) * A_{t+1}.  With lam = 1 the intermediate
    values telescope and A_t = outcome - V(s_t).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not len(v):
        raise ValueError("values must be a non-empty 1-D trajectory")
    if not np.isfinite(v).all():
        raise ValueError("non-finite value in trajectory")
    next_v = np.append(v[1:], 0.0)
    rewards = np.zeros_like(v)
    rewards[-1] = float(outcome)
    delta = rewards + next_v - v
    advantages = np.empty_like(v)
    running = 0.0
    for t in range(len(v) - 1, -1, -1):
        running = delta[t] + lam * running
        advantages[t] = running
    return advantages
