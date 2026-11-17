"""Generic window features for an unsupervised anomaly detector.

These are the features a fault-detection engineer would reach for without any model of how
an attacker behaves: per-channel noise statistics and comparisons between redundant sensors
measuring the same quantity (the redundancy management every avionics system has). They say
*that* something is unusual; nothing in them encodes a hypothesis about *why*.

Every feature is normalised by the noise level it would have in a healthy chain, so that a
value near 1 means "as noisy as expected".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import CHANNEL_NAMES, nominal_sigma

# redundant comparisons: (name, channel a, channel b, weight of b); the baro is compared with
# the mean of the two receivers
PAIRS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("gnss.pos.x", ("gnss_1.pos.x",), ("gnss_2.pos.x",)),
    ("gnss.vel.x", ("gnss_1.vel.x",), ("gnss_2.vel.x",)),
    ("gnss.pos.h", ("gnss_1.pos.h",), ("gnss_2.pos.h",)),
    ("gnss.vel.h", ("gnss_1.vel.h",), ("gnss_2.vel.h",)),
    ("ins.acc.x", ("ins_1.acc.x",), ("ins_2.acc.x",)),
    ("ins.acc.h", ("ins_1.acc.h",), ("ins_2.acc.h",)),
    ("baro-gnss.pos.h", ("baro.pos.h",), ("gnss_1.pos.h", "gnss_2.pos.h")),
)


def _pair_difference(
    values: NDArray[np.float64], sigma: NDArray[np.float64], a: tuple[str, ...], b: tuple[str, ...]
) -> tuple[NDArray[np.float64], float]:
    ia = [CHANNEL_NAMES.index(n) for n in a]
    ib = [CHANNEL_NAMES.index(n) for n in b]
    diff = values[..., ia].mean(axis=-1) - values[..., ib].mean(axis=-1)
    noise = float(
        np.sqrt((sigma[ia] ** 2).sum() / len(ia) ** 2 + (sigma[ib] ** 2).sum() / len(ib) ** 2)
    )
    return diff, noise


def feature_names() -> list[str]:
    names = []
    for ch in CHANNEL_NAMES:
        names += [f"{ch} d2 std", f"{ch} d2 max"]
    for name, _, _ in PAIRS:
        names += [f"{name} diff mean", f"{name} diff std", f"{name} diff max"]
    return names


def generic_features(values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float64]:
    """Features of each scenario; ``values`` (N, T, 13) -> (N, 47)."""
    sigma = nominal_sigma(params)
    d2 = values[:, 2:, :] - 2.0 * values[:, 1:-1, :] + values[:, :-2, :]
    d2 = d2 / (sigma * np.sqrt(6.0))
    cols = [d2.std(axis=1), np.abs(d2).max(axis=1)]
    per_channel = np.stack(cols, axis=-1).reshape(values.shape[0], -1)
    pair_cols = []
    for _, a, b in PAIRS:
        diff, noise = _pair_difference(values, sigma, a, b)
        z = diff / noise
        pair_cols += [
            z.mean(axis=1),
            z.std(axis=1),
            np.abs(z - z.mean(axis=1, keepdims=True)).max(axis=1),
        ]
    return np.column_stack([per_channel, *pair_cols])
