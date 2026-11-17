"""Attack sets for the robustness analysis: one attack, more and more physics.

* **Consistency sweep**: the same attacks (same seeds, same intended effect on the target
  channel) at stealth s = 0, 0.1, ..., 1, for the static pattern (L1 -> L2) and the dynamic
  one (single-channel drag-off -> L3).
* **Fault mimicry**: an attacker who copies a fault. *Near mimicry* is a single-channel step
  of fault size (no ramp, offset drawn from the receiver-bias range). *Exact mimicry*
  reproduces a receiver bias fault's effect on the telemetry exactly; its data then have the
  same distribution as the fault's, so no detector can attribute it better than the prior.
* **Spoofer reach**: the drag-off of an attacker who reaches only the receivers (the effect
  of CPS-02's ``gnss_spoof`` leaf), against one who also rewrites inertial and air data on the
  bus (``inject_false_nav_messages``).
"""

from __future__ import annotations

import numpy as np

from fdicps.dual_anomaly_data_generation.faults import PhysicalFault
from fdicps.dual_anomaly_data_generation.fdi import FalseDataInjection
from fdicps.dual_anomaly_data_generation.parameters import (
    load_anomaly_ranges,
    load_sensor_parameters,
)
from fdicps.dual_anomaly_data_generation.scenarios import (
    Dataset,
    Scenario,
    make_scenario,
    sample_fdi,
    stack,
)
from fdicps.dual_anomaly_data_generation.sensors import channels_where

CONSISTENCY_GRID = tuple(round(0.1 * i, 1) for i in range(11))


def _anomaly_rng(seed: int) -> np.random.Generator:
    """The same anomaly stream make_scenario uses, so variants of one seed are paired."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])


def consistency_variant(seed: int, pattern: str, s: float) -> Scenario:
    """The seed's attack with stealth s. Pattern ``offset`` goes from L1 (s = 0) to L2,
    ``drag`` from a single-channel drag-off to L3; the nominal level is the s = 1 one."""
    params, ranges = load_sensor_parameters(), load_anomaly_ranges()
    level = 2 if pattern == "offset" else 3
    fdi = sample_fdi(_anomaly_rng(seed), params, ranges, level, consistency=s)
    return make_scenario(seed, "fdi", anomaly=fdi, params=params)


def consistency_sweep(seeds: range | list[int], pattern: str) -> dict[float, Dataset]:
    return {
        s: stack([consistency_variant(seed, pattern, s) for seed in seeds])
        for s in CONSISTENCY_GRID
    }


def near_mimic(seed: int) -> Scenario:
    """A single-channel step of receiver-bias size on one receiver position (no ramp)."""
    params, ranges = load_sensor_parameters(), load_anomaly_ranges()
    rng = _anomaly_rng(seed)
    base = sample_fdi(rng, params, ranges, 1, include_baro=False)
    magnitude = np.sign(base.magnitude) * ranges["gnss_bias"].sample(rng)
    fdi = FalseDataInjection(
        level=1,
        pattern="offset",
        axis=base.axis,
        onset=base.onset,
        magnitude=float(magnitude),
        ramp=0.0,
        target=base.target,
        consistency=0.0,
    )
    return make_scenario(seed, "fdi", anomaly=fdi, params=params)


def receiver_bias_fault(seed: int) -> Scenario:
    """A real receiver bias fault (a random receiver, onset, size and direction)."""
    params, ranges = load_sensor_parameters(), load_anomaly_ranges()
    rng = _anomaly_rng(seed)
    unit = ("gnss_1", "gnss_2")[int(rng.integers(2))]
    onset = int(rng.integers(params.onset_min, params.onset_max + 1))
    signs = rng.choice([-1.0, 1.0], size=2)
    size = ranges["gnss_bias"].sample(rng)
    fault = PhysicalFault(unit, "bias", onset, size, (float(signs[0]), float(signs[1])))
    return make_scenario(seed, "fault", anomaly=fault, params=params)


def exact_mimic(seed: int) -> Scenario:
    """The effect of a receiver bias fault, produced by an attacker: the telemetry has the same
    distribution as :func:`receiver_bias_fault`'s, but the label is FDI."""
    s = receiver_bias_fault(seed)
    return Scenario(seed=seed, label="fdi", anomaly=None, values=s.values, truth=s.truth)


def spoofer_variant(seed: int, reach: str) -> Scenario:
    """An L3 drag-off by an attacker who reaches ``receivers`` only or the whole ``bus``."""
    params, ranges = load_sensor_parameters(), load_anomaly_ranges()
    on_bus = reach == "bus"
    fdi = sample_fdi(_anomaly_rng(seed), params, ranges, 3, include_baro=on_bus, include_ins=on_bus)
    return make_scenario(seed, "fdi", anomaly=fdi, params=params)


def receiver_position_channels() -> list[int]:
    return channels_where(unit="gnss_1", quantity="pos") + channels_where(
        unit="gnss_2", quantity="pos"
    )
