"""Sensor and anomaly parameters, read from the CSV tables in ``fdicps.data``.

Every value in those tables carries a ``basis`` column: either an explicit assumption or the
public source it was derived from. Nothing here is field data.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from importlib import resources

import numpy as np


@dataclass(frozen=True)
class SensorParameters:
    """Noise and timing of the navigation chain's sensors (SI units)."""

    dt: float
    duration: int
    gnss_pos_sigma_x: float
    gnss_pos_sigma_h: float
    gnss_vel_sigma: float
    ins_acc_sigma: float
    ins_bias_sigma: float
    baro_sigma: float
    onset_min: int
    onset_max: int

    @property
    def n_steps(self) -> int:
        return int(round(self.duration / self.dt))


@dataclass(frozen=True)
class Range:
    """A closed interval from which an anomaly magnitude is drawn uniformly."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


def _read(name: str) -> list[dict[str, str]]:
    text = resources.files("fdicps.data").joinpath(name).read_text()
    return list(csv.DictReader(io.StringIO(text)))


def load_sensor_parameters() -> SensorParameters:
    rows = {r["name"]: float(r["value"]) for r in _read("sensor_parameters.csv")}
    return SensorParameters(
        dt=rows["dt"],
        duration=int(rows["duration"]),
        gnss_pos_sigma_x=rows["gnss_pos_sigma_x"],
        gnss_pos_sigma_h=rows["gnss_pos_sigma_h"],
        gnss_vel_sigma=rows["gnss_vel_sigma"],
        ins_acc_sigma=rows["ins_acc_sigma"],
        ins_bias_sigma=rows["ins_bias_sigma"],
        baro_sigma=rows["baro_sigma"],
        onset_min=int(rows["onset_min"]),
        onset_max=int(rows["onset_max"]),
    )


def load_anomaly_ranges() -> dict[str, Range]:
    return {
        r["name"]: Range(float(r["low"]), float(r["high"])) for r in _read("anomaly_parameters.csv")
    }


def parameter_bases() -> dict[str, str]:
    """The ``basis`` column of both tables, keyed by parameter name (for the docs)."""
    out = {r["name"]: r["basis"] for r in _read("sensor_parameters.csv")}
    out.update({r["name"]: r["basis"] for r in _read("anomaly_parameters.csv")})
    return out
