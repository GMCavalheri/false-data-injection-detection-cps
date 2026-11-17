"""Physics-informed features: which physical relation broke, where, and how.

Each feature summarises one residual of :mod:`residuals` over the whole segment. They are
grouped by the question they answer:

* **noise structure, per unit**: is the output frozen (longest run of identical samples),
  too smooth (smallest windowed second-difference spread: noise-free values), or too noisy
  (largest windowed spread)?
* **redundancy, per pair**: by how many noise standard deviations do the redundant sources
  disagree at worst, at the end, and how abruptly did that happen (step vs ramp)?
* **kinematics, per relation**: does a receiver's position still follow its own velocity and
  the inertial acceleration, over 10 s and over 60 s? Is the velocity still consistent with
  the INS? Does the baro still follow the vertical velocity?
* **coordination, per axis**: are *both* receivers kinematically inconsistent at once?

A physical fault breaks the relations of one unit, in the way hardware fails. A crude attack
breaks them in ways hardware does not (no noise, a whole-unit offset ramped in); a careful
one breaks fewer of them; the most careful one breaks none.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fdicps.baseline_anomaly_detection.features import PAIRS
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import UNITS, channels_where
from fdicps.physical_vs_fdi_classification.residuals import (
    REFERENCE,
    kinematic_family,
    pair_z,
    second_difference_z,
    window_z,
)

NOISE_WINDOW = 30
PAIR_WINDOW = 20
STEP_WINDOW = 10
KIN_WINDOWS = (10, 60)


def _rolling_mean(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    return np.asarray((c[..., n:] - c[..., :-n]) / n)


def _rolling_std(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    return np.asarray(np.sqrt(np.maximum(m2 - m * m, 0.0)))


def _longest_zero_run(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Longest run of exactly-zero values along the last axis, as a fraction of its length."""
    zero = (x == 0.0).astype(np.int64)
    best = np.zeros(x.shape[:-1])
    run = np.zeros(x.shape[:-1])
    for k in range(x.shape[-1]):
        run = (run + 1) * zero[..., k]
        best = np.maximum(best, run)
    return np.asarray(best / x.shape[-1], dtype=np.float64)


def feature_names() -> list[str]:
    names = []
    for u in UNITS:
        names += [f"frozen {u}", f"smooth {u}", f"rough {u}"]
    for p in PAIRS:
        names += [f"disagree max {p[0]}", f"disagree end {p[0]}", f"disagree step {p[0]}"]
    for rel in _relations():
        names += [f"{rel} z{n}" for n in KIN_WINDOWS]
    names += ["both receivers inconsistent x", "both receivers inconsistent h"]
    return names


def _relations() -> list[str]:
    return [
        "kin gnss_1 x",
        "kin gnss_2 x",
        "vel-ins x",
        "kin gnss_1 h",
        "kin gnss_2 h",
        "vel-ins h",
        "kin baro h",
    ]


def physics_features(values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float64]:
    """Features of each scenario; ``values`` (N, T, 13) -> (N, 52)."""
    cols: list[NDArray[np.float64]] = []
    d1 = np.diff(values, axis=-2)
    d2 = second_difference_z(values, params)
    for u in UNITS:
        idx = channels_where(unit=u)
        cols.append(np.max(_longest_zero_run(np.moveaxis(d1[..., idx], -2, -1)), axis=-1))
        spread = _rolling_std(np.moveaxis(d2[..., idx], -2, -1), NOISE_WINDOW)
        cols.append(spread.min(axis=(-2, -1)))
        cols.append(spread.max(axis=(-2, -1)))
    pz = np.moveaxis(pair_z(values, params), -1, -2)  # (N, 7, T)
    roll = _rolling_mean(pz, PAIR_WINDOW)
    step = _rolling_mean(pz, STEP_WINDOW)
    step = np.abs(step[..., STEP_WINDOW:] - step[..., :-STEP_WINDOW]).max(axis=-1)
    for j in range(len(PAIRS)):
        cols += [np.abs(roll[:, j]).max(axis=-1), np.abs(roll[:, j, -1]), step[:, j]]
    kin = kinematic_family(values, params)
    peak = {}
    for rel in _relations():
        for n in KIN_WINDOWS:
            z = np.abs(window_z(kin[rel], n)[..., REFERENCE:]).max(axis=-1)
            peak[(rel, n)] = z
            cols.append(z)
    for ax in "xh":
        cols.append(np.minimum(peak[(f"kin gnss_1 {ax}", 10)], peak[(f"kin gnss_2 {ax}", 10)]))
    return np.column_stack(cols)
