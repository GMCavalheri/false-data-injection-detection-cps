"""Physics residuals: signals that are standard normal while the chain is healthy.

Each residual is built from one physical relation the telemetry must satisfy, and divided by
the standard deviation the sensor noise gives it, so that "3" always means "three standard
deviations". Which relation breaks, on which unit, and how, is what separates the two kinds
of wrong.

* **Kinematic residual of a receiver** (position against its own velocity and the inertial
  acceleration, the exact zero-order-hold relation of Lesson 1):
  r[k] = p[k+1] - p[k] - v[k] dt - abar[k] dt^2 / 2.
  A sum of n consecutive residuals telescopes, so its noise variance is
  2 sigma_p^2 + n sigma_v^2 dt^2 + n (dt^4 / 4) sigma_a^2 / 2, not n times one term.
* **Velocity-inertial residual** (mean receiver velocity against the mean INS acceleration):
  q[k] = vbar[k+1] - vbar[k] - abar[k] dt; window-sum variance
  sigma_v^2 + n dt^2 sigma_a^2 / 2.
* **Baro kinematic residual** (baro altitude against the mean vertical velocity).
* **Redundant-pair differences** (as in Phase 2), centred on a reference window.
* **Normalised second differences** of every channel (noise structure: stuck, noise-free,
  noisy).

The INS bias makes q's mean non-zero; it is constant over a segment, so every residual is
centred on the mean over a reference window at the start of the segment, before any anomaly
can begin (onsets are at 100 s or later; the reference is the first ``REFERENCE`` seconds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fdicps.baseline_anomaly_detection.features import PAIRS, _pair_difference
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import (
    CHANNEL_NAMES,
    channels_where,
    nominal_sigma,
)

REFERENCE = 60  # samples of clean history used to centre every residual


def _col(name: str) -> int:
    return CHANNEL_NAMES.index(name)


@dataclass(frozen=True)
class KinematicTerms:
    """Per-sample residuals (N, T-1) and the per-term variances of a window sum."""

    r: NDArray[np.float64]
    var_fixed: float  # variance part that does not grow with the window
    var_per_sample: float  # variance part that grows linearly with the window


def _mean_acc(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    return np.asarray(values[..., channels_where(quantity="acc", axis=axis)].mean(axis=-1))


def receiver_kinematic(
    values: NDArray[np.float64], params: SensorParameters, unit: str, axis: int
) -> KinematicTerms:
    ax = "xh"[axis]
    p = values[..., _col(f"{unit}.pos.{ax}")]
    v = values[..., _col(f"{unit}.vel.{ax}")]
    a = _mean_acc(values, axis)
    dt = params.dt
    r = p[..., 1:] - p[..., :-1] - v[..., :-1] * dt - 0.5 * a[..., :-1] * dt * dt
    sp = nominal_sigma(params)[_col(f"{unit}.pos.{ax}")]
    per = (params.gnss_vel_sigma * dt) ** 2 + 0.25 * dt**4 * params.ins_acc_sigma**2 / 2
    return KinematicTerms(r, 2 * sp**2, per)


def baro_kinematic(values: NDArray[np.float64], params: SensorParameters) -> KinematicTerms:
    b = values[..., _col("baro.pos.h")]
    v = values[..., channels_where(quantity="vel", axis=1)].mean(axis=-1)
    a = _mean_acc(values, 1)
    dt = params.dt
    r = b[..., 1:] - b[..., :-1] - v[..., :-1] * dt - 0.5 * a[..., :-1] * dt * dt
    per = (params.gnss_vel_sigma * dt) ** 2 / 2 + 0.25 * dt**4 * params.ins_acc_sigma**2 / 2
    return KinematicTerms(r, 2 * params.baro_sigma**2, per)


def velocity_inertial(
    values: NDArray[np.float64], params: SensorParameters, axis: int
) -> KinematicTerms:
    v = values[..., channels_where(quantity="vel", axis=axis)].mean(axis=-1)
    a = _mean_acc(values, axis)
    dt = params.dt
    q = v[..., 1:] - v[..., :-1] - a[..., :-1] * dt
    return KinematicTerms(q, params.gnss_vel_sigma**2, dt**2 * params.ins_acc_sigma**2 / 2)


def window_variance(terms: KinematicTerms, n: int) -> float:
    """Noise variance of a centred n-sample window sum that does not overlap the reference.

    The window sum S has variance f + n g (telescoping, see the module docstring). Centring
    subtracts n times the reference mean M, a sum over R samples divided by R, so
    Var(M) = (f + R g) / R^2; the two are independent when the windows do not overlap.
    """
    f, g, R = terms.var_fixed, terms.var_per_sample, REFERENCE
    return f + n * g + n * n * (f + R * g) / R**2


def window_z(terms: KinematicTerms, n: int) -> NDArray[np.float64]:
    """Centred window sums of a residual, divided by their noise standard deviation.

    Shape (N, T-1-n+1). Healthy and outside the reference window: N(0, 1).
    """
    r = terms.r - terms.r[..., :REFERENCE].mean(axis=-1, keepdims=True)
    c = np.concatenate([np.zeros(r.shape[:-1] + (1,)), np.cumsum(r, axis=-1)], axis=-1)
    s = c[..., n:] - c[..., :-n]
    return np.asarray(s / np.sqrt(window_variance(terms, n)))


def pair_z(values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float64]:
    """Redundant-pair differences centred on the reference window, unit noise; (N, T, 7)."""
    sigma = nominal_sigma(params)
    out = []
    for _, a, b in PAIRS:
        diff, noise = _pair_difference(values, sigma, a, b)
        z = diff / noise
        out.append(z - z[..., :REFERENCE].mean(axis=-1, keepdims=True))
    return np.stack(out, axis=-1)


def second_difference_z(
    values: NDArray[np.float64], params: SensorParameters
) -> NDArray[np.float64]:
    d2 = values[..., 2:, :] - 2.0 * values[..., 1:-1, :] + values[..., :-2, :]
    return np.asarray(d2 / (nominal_sigma(params) * np.sqrt(6.0)))


def kinematic_family(
    values: NDArray[np.float64], params: SensorParameters
) -> dict[str, KinematicTerms]:
    """Every kinematic relation, by name."""
    out = {}
    for axis, ax in enumerate("xh"):
        for unit in ("gnss_1", "gnss_2"):
            out[f"kin {unit} {ax}"] = receiver_kinematic(values, params, unit, axis)
        out[f"vel-ins {ax}"] = velocity_inertial(values, params, axis)
    out["kin baro h"] = baro_kinematic(values, params)
    return out


def residual_sequences(
    values: NDArray[np.float64], params: SensorParameters, window: int = 10
) -> tuple[NDArray[np.float64], list[str]]:
    """Every residual as a sequence, aligned and cropped to a common length.

    Returns (N, L, C) and the C channel names; this is the input of the sequence model.
    """
    d2 = second_difference_z(values, params)  # length T-2
    pz = pair_z(values, params)[..., 2:, :]  # length T-2
    L = d2.shape[-2] - window + 1
    kin = kinematic_family(values, params)
    kz = [window_z(t, window)[..., -L:] for t in kin.values()]
    seq = np.concatenate([d2[..., -L:, :], pz[..., -L:, :], np.stack(kz, axis=-1)], axis=-1)
    names = [f"d2 {c}" for c in CHANNEL_NAMES] + [f"pair {p[0]}" for p in PAIRS] + list(kin)
    return seq, names
