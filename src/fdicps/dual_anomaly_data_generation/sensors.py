"""The navigation chain's sensors and the telemetry they put on the data bus.

The architecture is the one of CPS-02 and CPS-01: two embedded GNSS/INS units (``egi_1`` and
``egi_2``) feed the mission computer over the data bus. Each unit contributes a GNSS receiver
(position and velocity) and an inertial measurement (acceleration, with a constant per-unit
bias). This project adds one dissimilar source, a barometric altitude from an air-data unit
(``air_data``), labelled an addition to the CPS-02 architecture.

Telemetry is an array of shape (T, 13) with the columns in ``CHANNELS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.mission import AXES, Truth
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters

UNITS = ("gnss_1", "gnss_2", "ins_1", "ins_2", "baro")

# hardware unit -> component of the CPS-02 / CPS-01 navigation chain
UNIT_COMPONENT = {
    "gnss_1": "egi_1",
    "ins_1": "egi_1",
    "gnss_2": "egi_2",
    "ins_2": "egi_2",
    "baro": "air_data",  # addition to the CPS-02 architecture
}


@dataclass(frozen=True)
class Channel:
    unit: str
    quantity: str  # "pos", "vel" or "acc"
    axis: int  # index into AXES

    @property
    def name(self) -> str:
        return f"{self.unit}.{self.quantity}.{AXES[self.axis]}"


CHANNELS: tuple[Channel, ...] = tuple(
    [
        ch
        for axis in range(2)
        for ch in (
            Channel("gnss_1", "pos", axis),
            Channel("gnss_1", "vel", axis),
            Channel("gnss_2", "pos", axis),
            Channel("gnss_2", "vel", axis),
            Channel("ins_1", "acc", axis),
            Channel("ins_2", "acc", axis),
        )
    ]
    + [Channel("baro", "pos", 1)]
)
N_CHANNELS = len(CHANNELS)
CHANNEL_NAMES = tuple(ch.name for ch in CHANNELS)


def channels_where(
    *, unit: str | None = None, quantity: str | None = None, axis: int | None = None
) -> list[int]:
    """Indices of the channels matching every given attribute."""
    return [
        i
        for i, ch in enumerate(CHANNELS)
        if (unit is None or ch.unit == unit)
        and (quantity is None or ch.quantity == quantity)
        and (axis is None or ch.axis == axis)
    ]


def nominal_sigma(params: SensorParameters) -> NDArray[np.float64]:
    """White-noise standard deviation of each channel, shape (13,)."""
    sigma = np.empty(N_CHANNELS)
    for i, ch in enumerate(CHANNELS):
        if ch.unit == "baro":
            sigma[i] = params.baro_sigma
        elif ch.quantity == "pos":
            sigma[i] = params.gnss_pos_sigma_x if ch.axis == 0 else params.gnss_pos_sigma_h
        elif ch.quantity == "vel":
            sigma[i] = params.gnss_vel_sigma
        else:
            sigma[i] = params.ins_acc_sigma
    return sigma


def measurement_model(axis: int) -> tuple[list[int], NDArray[np.float64]]:
    """Snapshot measurement model of one axis: z = H [p, v] + e.

    Returns the channel indices (positions and velocities of both receivers, plus the baro
    altitude on the vertical axis) and H, with one row per channel and columns [p, v].
    """
    idx = [i for i in channels_where(axis=axis) if CHANNELS[i].quantity in ("pos", "vel")]
    H = np.array([[1.0, 0.0] if CHANNELS[i].quantity == "pos" else [0.0, 1.0] for i in idx])
    return idx, H


@dataclass(frozen=True)
class Telemetry:
    values: NDArray[np.float64]  # (T, 13)
    ins_bias: NDArray[np.float64]  # (2 units, 2 axes), constant over the segment


def clean_signal(truth: Truth, ins_bias: NDArray[np.float64]) -> NDArray[np.float64]:
    """What every channel would read with no noise: shape (T, 13)."""
    out = np.empty((truth.t.size, N_CHANNELS))
    for i, ch in enumerate(CHANNELS):
        if ch.quantity == "pos":
            out[:, i] = truth.p[:, ch.axis]
        elif ch.quantity == "vel":
            out[:, i] = truth.v[:, ch.axis]
        else:
            unit = 0 if ch.unit == "ins_1" else 1
            out[:, i] = truth.a[:, ch.axis] + ins_bias[unit, ch.axis]
    return out


def measure(truth: Truth, params: SensorParameters, rng: np.random.Generator) -> Telemetry:
    """Noisy telemetry of the healthy chain (independent white Gaussian noise per channel)."""
    ins_bias = rng.normal(0.0, params.ins_bias_sigma, size=(2, 2))
    noise = rng.normal(size=(truth.t.size, N_CHANNELS)) * nominal_sigma(params)
    return Telemetry(values=clean_signal(truth, ins_bias) + noise, ins_bias=ins_bias)
