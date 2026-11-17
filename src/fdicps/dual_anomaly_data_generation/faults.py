"""Genuine physical sensor faults.

A physical fault lives in one piece of hardware, so it changes the outputs of one unit and
only in ways that hardware can fail:

- ``bias``: a step offset (GNSS position, INS acceleration or baro altitude);
- ``drift``: an offset that grows linearly from the onset;
- ``stuck``: every output of the unit frozen at its last good value, with no noise;
- ``noise``: the unit's noise standard deviation multiplied by a factor;
- ``common_mode``: the one physical event that touches both receivers at once, a shared,
  slowly growing and saturating position error (ionospheric-like). It is included on purpose:
  "both redundant units moved together" is *not* proof of an attack.

None of these is shaped to fool a consistency check: that is the difference from FDI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS, channels_where, nominal_sigma

FaultKind = Literal["bias", "drift", "stuck", "noise", "common_mode"]
FAULT_KINDS: tuple[FaultKind, ...] = ("bias", "drift", "stuck", "noise", "common_mode")
FAULT_UNITS = ("gnss_1", "gnss_2", "ins_1", "ins_2", "baro")


@dataclass(frozen=True)
class PhysicalFault:
    """One physical fault. ``magnitude`` is a step size, a drift rate, a noise factor or a
    common-mode rate depending on ``kind``; ``signs`` gives its direction on each axis."""

    unit: str  # one of FAULT_UNITS, or "gnss" for common_mode
    kind: FaultKind
    onset: int
    magnitude: float
    signs: tuple[float, float] = (1.0, 1.0)
    cap: float = float("inf")

    @property
    def subtype(self) -> str:
        return f"{self.unit} {self.kind}"


def offset_channels(fault: PhysicalFault) -> list[int]:
    """Channels a bias, drift or common-mode fault adds its offset to."""
    if fault.kind == "common_mode":
        return channels_where(unit="gnss_1", quantity="pos") + channels_where(
            unit="gnss_2", quantity="pos"
        )
    if fault.unit.startswith("gnss"):
        return channels_where(unit=fault.unit, quantity="pos")
    return channels_where(unit=fault.unit)


def affected_channels(fault: PhysicalFault) -> list[int]:
    if fault.kind in ("stuck", "noise"):
        return channels_where(unit=fault.unit)
    return offset_channels(fault)


def apply_fault(
    values: NDArray[np.float64],
    fault: PhysicalFault,
    params: SensorParameters,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """A copy of ``values`` (T, 13) with the fault applied from ``fault.onset`` on."""
    out = values.copy()
    k0 = fault.onset
    tau = (np.arange(values.shape[0] - k0)) * params.dt
    if fault.kind == "stuck":
        idx = affected_channels(fault)
        out[k0:, idx] = values[k0 - 1, idx]
    elif fault.kind == "noise":
        idx = affected_channels(fault)
        extra = nominal_sigma(params)[idx] * np.sqrt(fault.magnitude**2 - 1.0)
        out[k0:, idx] += rng.normal(size=(tau.size, len(idx))) * extra
    else:
        if fault.kind == "bias":
            shape = np.full(tau.size, fault.magnitude)
        elif fault.kind == "drift":
            shape = fault.magnitude * tau
        else:  # common_mode
            shape = np.minimum(fault.magnitude * tau, fault.cap)
        for i in offset_channels(fault):
            out[k0:, i] += fault.signs[CHANNELS[i].axis] * shape
    return out
