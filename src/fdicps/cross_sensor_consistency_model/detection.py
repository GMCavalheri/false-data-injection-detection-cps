"""Dynamic consistency monitors built on the Kalman filter, and consistency features.

* **Per-sample NIS test**: NIS[k] ~ chi2(m) independently in time when the model holds, so
  with every sample after the reference window tested on both axes, a per-test level from
  :func:`per_test_level` makes the scenario false-alarm rate exact.
* **Windowed NIS test**: the sum of NIS over a block of n samples is chi2(n m); blocks that do
  not overlap are independent, so the same exact construction applies. Small persistent
  inconsistencies that no single sample shows add up here.
* **CUSUM** on NIS - m (Page, 1954): g[k] = max(0, g[k-1] + NIS[k] - m - drift). It reacts to a
  persistent excess fastest, but its scenario false-alarm rate has no simple closed form, so
  its threshold is calibrated on normal training scenarios (an empirical quantile).

The consistency features summarise these statistics (and the snapshot check of
:mod:`navigation`) for the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fdicps.adversarial_robustness_analysis.monitors import per_test_level
from fdicps.cross_sensor_consistency_model.kalman import FilterRun, axis_model, run_filter
from fdicps.cross_sensor_consistency_model.navigation import snapshot_statistics
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS
from fdicps.physical_vs_fdi_classification.residuals import REFERENCE

WINDOW = 20
CUSUM_DRIFT = 0.5  # per unit of m: allowance before NIS excess accumulates


@dataclass(frozen=True)
class AxisRuns:
    runs: dict[int, FilterRun]
    dofs: dict[int, int]


def filter_axes(
    values: NDArray[np.float64], params: SensorParameters, exclude: list[int] | None = None
) -> AxisRuns:
    """Run both axes' filters; ``exclude`` drops channels from the filter (e.g. the baro)."""
    runs, dofs = {}, {}
    for axis in (0, 1):
        model = axis_model(params, axis)
        if exclude:
            model = axis_model(params, axis, include=[i for i in model.idx if i not in exclude])
        runs[axis] = run_filter(values, params, model)
        dofs[axis] = model.m
    return AxisRuns(runs, dofs)


def nis_alarm(runs: AxisRuns, scenario_false_alarm: float) -> NDArray[np.bool_]:
    n = runs.runs[0].nis.shape[-1] - REFERENCE
    alpha = per_test_level(scenario_false_alarm, 2 * n)
    alarm = None
    for axis, run in runs.runs.items():
        hit = (run.nis[..., REFERENCE:] > stats.chi2.isf(alpha, runs.dofs[axis])).any(axis=-1)
        alarm = hit if alarm is None else alarm | hit
    assert alarm is not None
    return np.asarray(alarm)


def window_sums(nis: NDArray[np.float64], window: int = WINDOW) -> NDArray[np.float64]:
    """Sums of NIS over non-overlapping blocks after the reference; shape (..., blocks)."""
    x = nis[..., REFERENCE:]
    blocks = x.shape[-1] // window
    return np.asarray(x[..., : blocks * window].reshape(*x.shape[:-1], blocks, window).sum(axis=-1))


def window_alarm(
    runs: AxisRuns, scenario_false_alarm: float, window: int = WINDOW
) -> NDArray[np.bool_]:
    sums = {a: window_sums(r.nis, window) for a, r in runs.runs.items()}
    n_tests = sum(s.shape[-1] for s in sums.values())
    alpha = per_test_level(scenario_false_alarm, n_tests)
    alarm = None
    for axis, s in sums.items():
        hit = (s > stats.chi2.isf(alpha, window * runs.dofs[axis])).any(axis=-1)
        alarm = hit if alarm is None else alarm | hit
    assert alarm is not None
    return np.asarray(alarm)


def cusum(nis: NDArray[np.float64], m: int, drift: float = CUSUM_DRIFT) -> NDArray[np.float64]:
    """Page's CUSUM of NIS - m - drift m after the reference; shape like ``nis[..., ref:]``."""
    x = nis[..., REFERENCE:] - m * (1.0 + drift)
    g = np.zeros(x.shape[:-1])
    out = np.empty_like(x)
    for k in range(x.shape[-1]):
        g = np.maximum(0.0, g + x[..., k])
        out[..., k] = g
    return out


def cusum_score(runs: AxisRuns) -> NDArray[np.float64]:
    """The largest CUSUM over both axes, in units of the axis dimension m."""
    return np.asarray(
        np.maximum(
            *(cusum(r.nis, runs.dofs[a]).max(axis=-1) / runs.dofs[a] for a, r in runs.runs.items())
        )
    )


def consistency_feature_names() -> list[str]:
    names = []
    for ax in "xh":
        names += [
            f"snapshot J max {ax}",
            f"NIS max {ax}",
            f"NIS window max {ax}",
            f"CUSUM max {ax}",
            f"innovation pos max {ax}",
            f"innovation vel max {ax}",
            f"bias estimate drift {ax}",
        ]
    return names


def _rolling_mean(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    return np.asarray((c[..., n:] - c[..., :-n]) / n)


def consistency_features(
    values: NDArray[np.float64], params: SensorParameters
) -> NDArray[np.float64]:
    """Features of the model-based checks; ``values`` (N, T, 13) -> (N, 14)."""
    runs = filter_axes(values, params)
    J = snapshot_statistics(values, params)[..., REFERENCE:, :]
    cols = []
    for axis, run in runs.runs.items():
        m = runs.dofs[axis]
        model = axis_model(params, axis)
        sd = np.sqrt(np.diagonal(run.gains.S, axis1=-2, axis2=-1))  # (T, m)
        nz = np.moveaxis(run.innovation / sd, -1, -2)[..., REFERENCE:]  # (N, m, T')
        roll = np.abs(_rolling_mean(nz, WINDOW) * np.sqrt(WINDOW)).max(axis=-1)  # (N, m)
        pos = [j for j, i in enumerate(model.idx) if CHANNELS[i].quantity == "pos"]
        vel = [j for j, i in enumerate(model.idx) if CHANNELS[i].quantity == "vel"]
        b = run.state[..., 2]
        b_sd = np.sqrt(run.gains.P_prior[-1, 2, 2])
        cols += [
            J[..., axis].max(axis=-1) / (m - 2),
            run.nis[..., REFERENCE:].max(axis=-1) / m,
            window_sums(run.nis).max(axis=-1) / (WINDOW * m),
            cusum(run.nis, m).max(axis=-1) / m,
            roll[:, pos].max(axis=-1),
            roll[:, vel].max(axis=-1),
            np.abs(b[..., -1] - b[..., REFERENCE]) / b_sd,
        ]
    return np.column_stack(cols)
