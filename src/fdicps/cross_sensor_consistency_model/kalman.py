"""A Kalman filter per axis: the INS predicts, the receivers and the baro correct.

State per axis: x = [p, v, b], with b the mean accelerometer bias of the two units. The mean
measured acceleration u = a + b + n drives the prediction with the exact zero-order-hold
kinematics of Lesson 1:

    x[k+1] = F x[k] + G u[k] + w[k],
    F = [[1, dt, -dt^2/2], [0, 1, -dt], [0, 0, 1]],   G = [dt^2/2, dt, 0],

with process noise Q = G G^T sigma_a^2 / 2 (the mean of two independent accelerometer
noises) plus a tiny bias random walk. The measurements are the receivers' positions and
velocities and, vertically, the baro: z = H x + e.

The filter's covariance and gain do not depend on the data, so they are computed once and
the state recursion runs for every scenario at once. With the model correct, the innovations
nu[k] are independent N(0, S[k]) and the normalised innovation squared NIS = nu^T S^-1 nu is
chi-square with m degrees of freedom: that is the dynamic consistency test. In steady state
the prior covariance solves the discrete algebraic Riccati equation (tested against SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS, channels_where, nominal_sigma

BIAS_RANDOM_WALK = 1e-10  # (m/s^2)^2 per step, assumption: keeps the bias mode stabilisable
INITIAL_STD = (100.0, 5.0)  # m, m/s: a loose prior, overwritten by the first measurements


@dataclass(frozen=True)
class AxisModel:
    axis: int
    F: NDArray[np.float64]
    G: NDArray[np.float64]
    Q: NDArray[np.float64]
    H: NDArray[np.float64]
    R: NDArray[np.float64]  # (m, m)
    idx: list[int]  # measurement channels
    acc: list[int]  # INS channels averaged into the control input

    @property
    def m(self) -> int:
        return len(self.idx)


def axis_model(params: SensorParameters, axis: int, include: list[int] | None = None) -> AxisModel:
    """The filter of one axis. ``include`` restricts the measurement channels (default: all
    position and velocity channels of the axis, baro included vertically)."""
    dt = params.dt
    F = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
    G = np.array([0.5 * dt * dt, dt, 0.0])
    Q = np.outer(G, G) * params.ins_acc_sigma**2 / 2 + np.diag([0.0, 0.0, BIAS_RANDOM_WALK])
    idx = [i for i in channels_where(axis=axis) if CHANNELS[i].quantity in ("pos", "vel")]
    if include is not None:
        idx = [i for i in idx if i in include]
    H = np.array(
        [[1.0, 0.0, 0.0] if CHANNELS[i].quantity == "pos" else [0.0, 1.0, 0.0] for i in idx]
    )
    R = np.diag(nominal_sigma(params)[idx] ** 2)
    return AxisModel(axis, F, G, Q, H, R, idx, channels_where(quantity="acc", axis=axis))


@dataclass(frozen=True)
class Gains:
    P_prior: NDArray[np.float64]  # (T, 3, 3)
    S: NDArray[np.float64]  # (T, m, m)
    S_inv: NDArray[np.float64]
    K: NDArray[np.float64]  # (T, 3, m)


def gains(model: AxisModel, n: int, bias_std: float) -> Gains:
    P = np.diag([INITIAL_STD[0] ** 2, INITIAL_STD[1] ** 2, bias_std**2])
    I3 = np.eye(3)
    out_P, out_S, out_Si, out_K = [], [], [], []
    for _ in range(n):
        S = model.H @ P @ model.H.T + model.R
        S_inv = np.linalg.inv(S)
        K = P @ model.H.T @ S_inv
        out_P.append(P)
        out_S.append(S)
        out_Si.append(S_inv)
        out_K.append(K)
        A = I3 - K @ model.H
        P_post = A @ P @ A.T + K @ model.R @ K.T  # Joseph form
        P = model.F @ P_post @ model.F.T + model.Q
    return Gains(np.array(out_P), np.array(out_S), np.array(out_Si), np.array(out_K))


@dataclass(frozen=True)
class FilterRun:
    innovation: NDArray[np.float64]  # (N, T, m)
    nis: NDArray[np.float64]  # (N, T)
    state: NDArray[np.float64]  # (N, T, 3) posterior
    gains: Gains


def run_filter(
    values: NDArray[np.float64], params: SensorParameters, model: AxisModel
) -> FilterRun:
    """Run the filter on telemetry (N, T, 13) of every scenario at once."""
    n = values.shape[-2]
    g = gains(model, n, params.ins_bias_sigma)
    z = values[..., model.idx]
    u = values[..., model.acc].mean(axis=-1)
    pos = [j for j, i in enumerate(model.idx) if CHANNELS[i].quantity == "pos"]
    vel = [j for j, i in enumerate(model.idx) if CHANNELS[i].quantity == "vel"]
    x = np.zeros(values.shape[:-2] + (3,))
    x[..., 0] = z[..., 0, pos].mean(axis=-1)
    x[..., 1] = z[..., 0, vel].mean(axis=-1) if vel else 0.0
    nus, states = [], []
    for k in range(n):
        nu = z[..., k, :] - x @ model.H.T
        x = x + nu @ g.K[k].T
        nus.append(nu)
        states.append(x)
        x = x @ model.F.T + u[..., k, None] * model.G
    innovation = np.stack(nus, axis=-2)
    nis = np.einsum("...ti,tij,...tj->...t", innovation, g.S_inv, innovation)
    return FilterRun(innovation, nis, np.stack(states, axis=-2), g)


def steady_state_prior(model: AxisModel) -> NDArray[np.float64]:
    """The prior covariance the filter converges to: the DARE solution."""
    from scipy.linalg import solve_discrete_are

    return np.asarray(solve_discrete_are(model.F.T, model.H.T, model.Q, model.R))
