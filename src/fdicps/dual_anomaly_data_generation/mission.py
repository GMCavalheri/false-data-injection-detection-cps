"""Truth trajectory of one flight segment: along-track and vertical kinematics.

The aircraft state on each axis is position and velocity, driven by an acceleration that is
held constant over each sample interval (zero-order hold). With that convention the discrete
model is *exact*, not an approximation:

    p[k+1] = p[k] + v[k] dt + a[k] dt^2 / 2,     v[k+1] = v[k] + a[k] dt,

i.e. x[k+1] = F x[k] + G a[k] with F = [[1, dt], [0, 1]] and G = [dt^2/2, dt]. The same F and
G are used by the attacker model (a dynamically consistent injection) and by the Kalman filter
in Phase 5, so the three agree to rounding error.

The profile is a sequence of smooth manoeuvres: on each axis the velocity moves to a new
target through a raised-cosine acceleration pulse, whose integral is exactly the velocity
change. Ranges are assumptions chosen to look like a transport aircraft in climb, cruise or
descent (along-track 180-250 m/s, vertical speed within +/-15 m/s, peak manoeuvre
acceleration at most 1 m/s^2, about 0.1 g).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters

AXES = ("x", "h")  # along-track, vertical

ALONG_SPEED = (180.0, 250.0)  # m/s, assumption
VERTICAL_SPEED = (-15.0, 15.0)  # m/s, assumption
START_ALTITUDE = (5000.0, 10000.0)  # m, assumption: stays positive for any vertical profile
MAX_MANOEUVRES = 3
MAX_ACCELERATION = 1.0  # m/s^2, assumption: peak of any manoeuvre pulse


@dataclass(frozen=True)
class Truth:
    """True kinematics, shape (T, 2) per array, columns ordered as ``AXES``."""

    t: NDArray[np.float64]
    p: NDArray[np.float64]
    v: NDArray[np.float64]
    a: NDArray[np.float64]


def transition(dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The exact zero-order-hold kinematic model: F (2x2) and G (2,)."""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([0.5 * dt * dt, dt])
    return F, G


def integrate(
    p0: NDArray[np.float64] | float,
    v0: NDArray[np.float64] | float,
    a: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Position and velocity from an acceleration sequence (exact under zero-order hold).

    ``a`` has shape (T, ...) and ``a[k]`` acts over [k, k+1); the returned arrays have the
    same shape, with ``p[0] = p0`` and ``v[0] = v0``.
    """
    a = np.asarray(a, dtype=float)
    zero = np.zeros_like(a[:1])
    dv = np.concatenate([zero, np.cumsum(a[:-1] * dt, axis=0)])
    v = v0 + dv
    dp = v[:-1] * dt + 0.5 * a[:-1] * dt * dt
    p = p0 + np.concatenate([zero, np.cumsum(dp, axis=0)])
    return p, v


def _pulse_profile(
    rng: np.random.Generator, n: int, dt: float, v0: float, bounds: tuple[float, float]
) -> NDArray[np.float64]:
    """Acceleration made of up to MAX_MANOEUVRES raised-cosine pulses in separate slots."""
    a = np.zeros(n)
    k = int(rng.integers(0, MAX_MANOEUVRES + 1))
    if k == 0:
        return a
    slot = n // k
    v = v0
    for i in range(k):
        length = int(rng.integers(20, min(60, slot) + 1))
        start = i * slot + int(rng.integers(0, slot - length + 1))
        # a(t) = A (1 - cos(2 pi s / L)) / 2 on samples s = 0..L-1 integrates to A L dt / 2
        # (sum of the samples); the peak is dv max(shape) / (sum dt), so capping it at
        # MAX_ACCELERATION bounds the reachable velocity change
        s = np.arange(length)
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * (s + 0.5) / length))
        reach = MAX_ACCELERATION * float(shape.sum()) * dt / float(np.max(shape))
        target = float(rng.uniform(max(bounds[0], v - reach), min(bounds[1], v + reach)))
        dv = target - v
        a[start : start + length] = dv * shape / (shape.sum() * dt)
        v = target
    return a


def mission_profile(rng: np.random.Generator, params: SensorParameters) -> Truth:
    """One random flight segment (both axes)."""
    n, dt = params.n_steps, params.dt
    v0 = np.array([rng.uniform(*ALONG_SPEED), rng.uniform(*VERTICAL_SPEED)])
    p0 = np.array([0.0, rng.uniform(*START_ALTITUDE)])
    a = np.column_stack(
        [
            _pulse_profile(rng, n, dt, float(v0[0]), ALONG_SPEED),
            _pulse_profile(rng, n, dt, float(v0[1]), VERTICAL_SPEED),
        ]
    )
    p, v = integrate(p0, v0, a, dt)
    return Truth(t=np.arange(n, dtype=np.float64) * dt, p=p, v=v, a=a)
