"""The IEEE 14-bus system's DC state-estimation model: the setting of the FDI literature.

The DC approximation of power flow makes the measurements linear in the bus voltage angles:
the real power flow on a branch from bus i to bus j is (theta_i - theta_j) / (x t), with x the
branch reactance and t the transformer tap ratio (1 where there is none), and a bus injection
is the sum of the flows leaving the bus. Bus 1 is the angle reference, so there are 13 states.
The meters are the 20 branch flows (from side) and the 14 bus injections: m = 34.

Branch data: MATPOWER ``case14`` (converted from the IEEE Common Data Format test case, UW
Power Systems Test Case Archive), reproduced in ``data/ieee14_branches.csv``. The B matrix
follows MATPOWER's ``makeBdc`` (series susceptance 1 / (x t), resistance and charging
neglected). The meter noise is an assumption.

The same :class:`~fdicps.cross_sensor_consistency_model.wls.WLS` code as the navigation chain
runs on it, which is the point: the stealthy-attack result is a property of linear state
estimation, not of power grids or of navigation.
"""

from __future__ import annotations

import csv
import io
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from fdicps.cross_sensor_consistency_model.wls import WLS

N_BUS = 14
REFERENCE_BUS = 1
METER_SIGMA = 0.01  # per unit on a 100 MVA base, assumption: 1 MW flow/injection meter noise


def branches() -> list[tuple[int, int, float, float]]:
    text = resources.files("fdicps.data").joinpath("ieee14_branches.csv").read_text()
    return [
        (int(r["from_bus"]), int(r["to_bus"]), float(r["x_pu"]), float(r["tap_ratio"]) or 1.0)
        for r in csv.DictReader(io.StringIO(text))
    ]


def measurement_matrix() -> NDArray[np.float64]:
    """H (34 x 13): 20 branch flows, then 14 bus injections, against the 13 non-reference
    angles (buses 2..14)."""
    br = branches()
    A = np.zeros((len(br), N_BUS))  # branch-bus incidence weighted by susceptance
    for k, (f, t, x, tap) in enumerate(br):
        b = 1.0 / (x * tap)
        A[k, f - 1] = b
        A[k, t - 1] = -b
    incidence = np.sign(A)
    flows = A
    injections = incidence.T @ A  # injection at a bus = sum of flows leaving it
    H = np.vstack([flows, injections])
    return np.delete(H, REFERENCE_BUS - 1, axis=1)


def model() -> WLS:
    H = measurement_matrix()
    return WLS(H, np.full(H.shape[0], METER_SIGMA))


def meters_to_move_one_state(H: NDArray[np.float64] | None = None) -> list[int]:
    """For each non-reference bus j, how many meters an attacker must alter for the stealthy
    attack a = H e_j (shift bus j's angle only): the nonzeros of column j."""
    H = measurement_matrix() if H is None else H
    return [int(np.count_nonzero(np.abs(H[:, j]) > 1e-12)) for j in range(H.shape[1])]
