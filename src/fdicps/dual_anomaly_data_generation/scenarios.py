"""Scenarios and datasets with ground truth by construction.

A scenario is one flight segment with at most one anomaly. Its seed is split into three
independent streams (``numpy.random.SeedSequence.spawn``): the trajectory, the sensor noise
and the anomaly. The same seed therefore gives the same clean telemetry whatever anomaly is
injected, which makes paired comparisons exact (the anomaly is the only difference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fdicps.dual_anomaly_data_generation.faults import (
    FAULT_KINDS,
    FAULT_UNITS,
    PhysicalFault,
    apply_fault,
)
from fdicps.dual_anomaly_data_generation.fdi import FalseDataInjection, Pattern, apply_fdi
from fdicps.dual_anomaly_data_generation.mission import Truth, mission_profile
from fdicps.dual_anomaly_data_generation.parameters import (
    Range,
    SensorParameters,
    load_anomaly_ranges,
    load_sensor_parameters,
)
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS, channels_where, measure

Label = Literal["normal", "fault", "fdi"]
LABELS: tuple[Label, ...] = ("normal", "fault", "fdi")
LABEL_CODE = {name: i for i, name in enumerate(LABELS)}

# (unit, kind) pairs a physical fault is drawn from: every unit x {bias, drift, stuck, noise},
# plus the shared receiver error
FAULT_MENU: tuple[tuple[str, str], ...] = tuple(
    (u, k) for u in FAULT_UNITS for k in FAULT_KINDS if k != "common_mode"
) + (("gnss", "common_mode"),)


@dataclass(frozen=True)
class Scenario:
    seed: int
    label: Label
    anomaly: PhysicalFault | FalseDataInjection | None
    values: NDArray[np.float64]
    truth: Truth

    @property
    def subtype(self) -> str:
        return "normal" if self.anomaly is None else self.anomaly.subtype

    @property
    def level(self) -> int:
        return self.anomaly.level if isinstance(self.anomaly, FalseDataInjection) else -1

    @property
    def onset(self) -> int:
        return -1 if self.anomaly is None else self.anomaly.onset


def _signs(rng: np.random.Generator) -> tuple[float, float]:
    s = rng.choice([-1.0, 1.0], size=2)
    return float(s[0]), float(s[1])


def sample_fault(
    rng: np.random.Generator, params: SensorParameters, ranges: dict[str, Range]
) -> PhysicalFault:
    unit, kind = FAULT_MENU[int(rng.integers(len(FAULT_MENU)))]
    onset = int(rng.integers(params.onset_min, params.onset_max + 1))
    signs = _signs(rng)
    family = unit.rstrip("_12")
    if kind == "common_mode":
        return PhysicalFault(
            unit,
            "common_mode",
            onset,
            ranges["common_mode_rate"].sample(rng),
            signs,
            cap=ranges["common_mode_cap"].sample(rng),
        )
    if kind == "noise":
        return PhysicalFault(unit, "noise", onset, ranges["noise_factor"].sample(rng), signs)
    if kind == "stuck":
        return PhysicalFault(unit, "stuck", onset, 0.0, signs)
    return PhysicalFault(unit, kind, onset, ranges[f"{family}_{kind}"].sample(rng), signs)  # type: ignore[arg-type]


def sample_fdi(
    rng: np.random.Generator,
    params: SensorParameters,
    ranges: dict[str, Range],
    level: int,
    *,
    consistency: float | None = None,
    pattern: Pattern | None = None,
    include_baro: bool = True,
    include_ins: bool = True,
) -> FalseDataInjection:
    """One FDI attack of the given nominal level.

    ``pattern`` and ``consistency`` default to the level's definition (L0 replace, L1 offset
    at s = 0, L2 offset at s = 1, L3 drag at s = 1); overriding them gives the continuous
    sophistication sweep of Phase 4.
    """
    if level not in (0, 1, 2, 3):
        raise ValueError("level must be 0, 1, 2 or 3")
    default_pattern: Pattern = ("replace", "offset", "offset", "drag")[level]
    pat: Pattern = pattern or default_pattern
    s = consistency if consistency is not None else (0.0, 0.0, 1.0, 1.0)[level]
    axis = int(rng.integers(2))
    onset = int(rng.integers(params.onset_min, params.onset_max + 1))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    candidates = [
        i
        for i in channels_where(quantity="pos", axis=axis)
        if include_baro or CHANNELS[i].unit != "baro"
    ]
    target = candidates[int(rng.integers(len(candidates)))]
    if pat == "drag":
        magnitude = sign * ranges["fdi_drag_acc"].sample(rng)
        ramp = ranges["fdi_drag_duration"].sample(rng)
    else:
        magnitude = sign * ranges["fdi_offset"].sample(rng)
        ramp = ranges["fdi_ramp"].sample(rng) if pat == "offset" else 0.0
    return FalseDataInjection(
        level=level,
        pattern=pat,
        axis=axis,
        onset=onset,
        magnitude=magnitude,
        ramp=ramp,
        target=target,
        consistency=s,
        include_baro=include_baro,
        include_ins=include_ins,
    )


def make_scenario(
    seed: int,
    label: Label,
    *,
    level: int | None = None,
    anomaly: PhysicalFault | FalseDataInjection | None = None,
    params: SensorParameters | None = None,
    **fdi_options: object,
) -> Scenario:
    """Build one scenario. Pass ``anomaly`` to inject a given one instead of sampling it."""
    params = params or load_sensor_parameters()
    ranges = load_anomaly_ranges()
    s_truth, s_noise, s_anom = np.random.SeedSequence(seed).spawn(3)
    truth = mission_profile(np.random.default_rng(s_truth), params)
    values = measure(truth, params, np.random.default_rng(s_noise)).values
    rng = np.random.default_rng(s_anom)
    if label == "fault":
        anomaly = anomaly or sample_fault(rng, params, ranges)
        assert isinstance(anomaly, PhysicalFault)
        values = apply_fault(values, anomaly, params, rng)
    elif label == "fdi":
        if anomaly is None:
            anomaly = sample_fdi(rng, params, ranges, 0 if level is None else level, **fdi_options)  # type: ignore[arg-type]
        assert isinstance(anomaly, FalseDataInjection)
        values = apply_fdi(values, anomaly, params)
    else:
        anomaly = None
    return Scenario(seed=seed, label=label, anomaly=anomaly, values=values, truth=truth)


@dataclass
class Dataset:
    """Stacked scenarios: ``values`` has shape (N, T, 13)."""

    values: NDArray[np.float64]
    label: NDArray[np.int64]
    level: NDArray[np.int64]
    onset: NDArray[np.int64]
    seed: NDArray[np.int64]
    subtype: list[str] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)


def stack(scenarios: list[Scenario]) -> Dataset:
    return Dataset(
        values=np.stack([s.values for s in scenarios]),
        label=np.array([LABEL_CODE[s.label] for s in scenarios], dtype=np.int64),
        level=np.array([s.level for s in scenarios], dtype=np.int64),
        onset=np.array([s.onset for s in scenarios], dtype=np.int64),
        seed=np.array([s.seed for s in scenarios], dtype=np.int64),
        subtype=[s.subtype for s in scenarios],
        scenarios=scenarios,
    )


def generate_dataset(
    seeds: range | list[int],
    *,
    fdi_levels: tuple[int, ...] = (0, 1, 2, 3),
    params: SensorParameters | None = None,
) -> Dataset:
    """A balanced dataset: seed i is normal, fault or FDI by i mod 3, and FDI scenarios cycle
    through ``fdi_levels``. Seeds, not windows, are the unit of any train/test split."""
    params = params or load_sensor_parameters()
    out: list[Scenario] = []
    for n, seed in enumerate(seeds):
        label = LABELS[n % 3]
        level = fdi_levels[(n // 3) % len(fdi_levels)] if label == "fdi" else None
        out.append(make_scenario(seed, label, level=level, params=params))
    return stack(out)
