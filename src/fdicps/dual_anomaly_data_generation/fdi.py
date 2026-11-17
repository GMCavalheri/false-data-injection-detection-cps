"""False data injection (FDI) at four levels of sophistication.

FDI is modelled at the level the estimator sees it: values on the data bus that differ from
what the sensors measured. How an attacker would gain that ability is out of scope (see
``docs/scope.md``). The attacker wants an *effect*, a navigation solution displaced by some
amount, and the levels differ only in how much physics the injection respects:

- **L0 naive** (``replace``): one channel replaced by a noise-free straight-line extrapolation
  plus a jump;
- **L1 individually plausible** (``offset``, consistency 0): the offset is added to one real
  measurement, so range, rate and noise all look normal, but the peers disagree;
- **L2 statically consistent** (``offset``, consistency 1): the same offset on every position
  channel of the axis, a = H c with c = [dp, 0]. Every snapshot residual is unchanged, but
  position moves while velocity does not;
- **L3 dynamically consistent** (``drag``, consistency 1): a drag-off. A small acceleration
  da is integrated exactly with the aircraft's own kinematics into position and velocity
  offsets, and the matching acceleration is added to the INS channels. Nothing in the
  modelled physics can see it; only a sensor outside the attacker's reach can.

``consistency`` (s in [0, 1]) blends the single-channel and the fully consistent version of
the same injection, a(s) = s A + (1 - s) A_target, so the induced effect on the target
channel is the same at every s and only the stealth changes.

``include_baro`` and ``include_ins`` say whether the attacker reaches the barometric and
inertial channels. With both false and the drag pattern, the injection is the effect of
CPS-02's ``gnss_spoof`` leaf: both receivers walked away together, the INS untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.mission import AXES, integrate
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS, N_CHANNELS, channels_where

Pattern = Literal["replace", "offset", "drag"]
LEVELS = (0, 1, 2, 3)
EXTRAPOLATION_WINDOW = 10  # samples the naive attacker fits its straight line to


@dataclass(frozen=True)
class FalseDataInjection:
    level: int
    pattern: Pattern
    axis: int
    onset: int
    magnitude: float  # offset in m (replace/offset) or drag acceleration in m/s^2 (drag)
    ramp: float  # ramp-in time in s (replace/offset) or drag duration in s (drag)
    target: int  # channel that carries the single-channel part
    consistency: float = 1.0
    include_baro: bool = True
    include_ins: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.consistency <= 1.0:
            raise ValueError("consistency must be in [0, 1]")
        if CHANNELS[self.target].axis != self.axis:
            raise ValueError("target channel must be on the attacked axis")

    @property
    def subtype(self) -> str:
        return f"L{self.level} {self.pattern} {AXES[self.axis]}"


def position_channels(fdi: FalseDataInjection) -> list[int]:
    idx = [
        i
        for i in channels_where(quantity="pos", axis=fdi.axis)
        if fdi.include_baro or CHANNELS[i].unit != "baro"
    ]
    return idx


def consistent_injection(
    fdi: FalseDataInjection, params: SensorParameters, n: int
) -> NDArray[np.float64]:
    """The fully consistent injection A (shape (n, 13)) that the attack is built from."""
    A = np.zeros((n, N_CHANNELS))
    k0 = fdi.onset
    tau = np.arange(n - k0) * params.dt
    if fdi.pattern in ("offset", "replace"):
        shape = (
            fdi.magnitude * np.clip(tau / fdi.ramp, 0.0, 1.0)
            if fdi.ramp > 0
            else np.full(tau.size, fdi.magnitude)
        )
        A[k0:, position_channels(fdi)] = shape[:, None]
        return A
    # drag: an acceleration da held for `ramp` seconds, integrated with the exact kinematics
    da = np.where(tau < fdi.ramp, fdi.magnitude, 0.0)
    dp, dv = integrate(0.0, 0.0, da, params.dt)
    A[k0:, position_channels(fdi)] = dp[:, None]
    A[k0:, channels_where(quantity="vel", axis=fdi.axis)] = dv[:, None]
    if fdi.include_ins:
        A[k0:, channels_where(quantity="acc", axis=fdi.axis)] = da[:, None]
    return A


def injection(fdi: FalseDataInjection, params: SensorParameters, n: int) -> NDArray[np.float64]:
    """The additive injection a(s) = s A + (1 - s) A_target, shape (n, 13)."""
    A = consistent_injection(fdi, params, n)
    single = np.zeros_like(A)
    single[:, fdi.target] = A[:, fdi.target]
    return fdi.consistency * A + (1.0 - fdi.consistency) * single


def apply_fdi(
    values: NDArray[np.float64], fdi: FalseDataInjection, params: SensorParameters
) -> NDArray[np.float64]:
    """A copy of ``values`` (T, 13) as the mission computer receives it under the attack."""
    n = values.shape[0]
    if fdi.pattern != "replace":
        return values + injection(fdi, params, n)
    out = values.copy()
    k0, j, w = fdi.onset, fdi.target, EXTRAPOLATION_WINDOW
    t = np.arange(n) * params.dt
    slope, intercept = np.polyfit(t[k0 - w : k0], values[k0 - w : k0, j], 1)
    out[k0:, j] = intercept + slope * t[k0:] + consistent_injection(fdi, params, n)[k0:, j]
    return out


def affected_channels(fdi: FalseDataInjection) -> list[int]:
    """Channels whose values the attack changes."""
    if fdi.pattern == "replace" or fdi.consistency == 0.0:
        return [fdi.target]
    idx = position_channels(fdi)
    if fdi.pattern == "drag":
        idx += channels_where(quantity="vel", axis=fdi.axis)
        if fdi.include_ins:
            idx += channels_where(quantity="acc", axis=fdi.axis)
    return sorted(idx)


# CPS-02 (cpsrbd v1.0.0) leaves whose *effect* these injections model, and the leaf whose
# assumed probability (0.1) this project measures as a function of sophistication.
CPS02_BUS_INJECTION = "inject_false_nav_messages"
CPS02_GNSS_SPOOF = "gnss_spoof"
CPS02_INTEGRITY_MISS = "integrity_monitor_miss"


def cps02_leaf(fdi: FalseDataInjection) -> str:
    """The CPS-02 attack-tree leaf this injection is the effect of.

    An RF spoofer reaches only the receivers, so a drag-off that leaves the INS and the baro
    untouched is ``gnss_spoof``; anything that also rewrites inertial or air-data values, or
    touches a single message, is injection on the bus.
    """
    if fdi.pattern == "drag" and not fdi.include_ins and not fdi.include_baro:
        return CPS02_GNSS_SPOOF
    return CPS02_BUS_INJECTION
