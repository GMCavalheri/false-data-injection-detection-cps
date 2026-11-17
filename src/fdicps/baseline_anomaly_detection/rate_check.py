"""Rule-based checks of the kind every avionics data consumer applies: envelope and rate.

**Rate check.** For a channel with white noise of standard deviation sigma, the second
difference d2[k] = z[k] - 2 z[k-1] + z[k-2] removes any constant and any straight-line trend
of the true signal, and its noise has variance (1 + 4 + 1) sigma^2 = 6 sigma^2. The check
flags a sample when |d2| > kappa sigma sqrt(6). Under straight-and-level flight (constant
velocity, zero acceleration) the true part of d2 is exactly zero, so a single test's false-
alarm probability is exactly 2 (1 - Phi(kappa)). In manoeuvres the true second difference
is small but not zero (at most 1 m/s^2 x 1 s^2 on a position channel), so the rate is
slightly higher; Phase 2's results measure it on the normal scenarios.

**Envelope check.** Values outside what the aircraft can physically do.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS, nominal_sigma

KAPPA = 5.0  # default threshold in noise standard deviations

# (quantity, axis) -> (low, high): assumptions, wider than the generator's own ranges
ENVELOPE: dict[tuple[str, int], tuple[float, float]] = {
    ("pos", 1): (0.0, 15_000.0),
    ("vel", 0): (100.0, 320.0),
    ("vel", 1): (-30.0, 30.0),
    ("acc", 0): (-3.0, 3.0),
    ("acc", 1): (-3.0, 3.0),
}


def single_test_false_alarm(kappa: float) -> float:
    """P(|N(0, 1)| > kappa): the false-alarm probability of one rate test with no trend."""
    return float(2.0 * stats.norm.sf(kappa))


def rate_scores(values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float64]:
    """|d2| / (sigma sqrt 6) for every sample and channel; shape (..., T - 2, 13)."""
    d2 = values[..., 2:, :] - 2.0 * values[..., 1:-1, :] + values[..., :-2, :]
    return np.asarray(np.abs(d2) / (nominal_sigma(params) * np.sqrt(6.0)))


def envelope_violations(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True where a channel leaves its physical envelope; shape like ``values``."""
    out = np.zeros(values.shape, dtype=bool)
    for i, ch in enumerate(CHANNELS):
        lim = ENVELOPE.get((ch.quantity, ch.axis))
        if lim is not None:
            out[..., i] = (values[..., i] < lim[0]) | (values[..., i] > lim[1])
    return out


def rule_score(values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float64]:
    """Scenario score: the largest rate score, or infinity if the envelope is left.

    ``values`` has shape (N, T, 13); the result has shape (N,).
    """
    score = rate_scores(values, params).max(axis=(-2, -1))
    out_of_envelope = envelope_violations(values).any(axis=(-2, -1))
    return np.where(out_of_envelope, np.inf, score)
