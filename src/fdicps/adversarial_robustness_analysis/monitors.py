"""Scenario-level integrity monitors with an exact false-alarm rate.

The snapshot check of :mod:`fdicps.cross_sensor_consistency_model.navigation` runs 2 T
independent chi-square tests per scenario (white noise: samples are independent in time,
and the two axes share no channel). Setting each test's level to
alpha = 1 - (1 - F)^(1 / 2T) makes the scenario false-alarm rate exactly F.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fdicps.cross_sensor_consistency_model.navigation import (
    snapshot_models,
    snapshot_statistics,
)
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters

SCENARIO_FALSE_ALARM = 0.01


def per_test_level(scenario_false_alarm: float, n_tests: int) -> float:
    """Level of each of n independent tests so that P(any alarm) = scenario_false_alarm."""
    return float(-np.expm1(np.log1p(-scenario_false_alarm) / n_tests))


def snapshot_alarm(
    values: NDArray[np.float64],
    params: SensorParameters,
    scenario_false_alarm: float = SCENARIO_FALSE_ALARM,
) -> NDArray[np.bool_]:
    """True for scenarios where any sample on either axis fails the chi-square test."""
    J = snapshot_statistics(values, params)  # (N, T, 2)
    alpha = per_test_level(scenario_false_alarm, 2 * J.shape[-2])
    models = snapshot_models(params)
    thresholds = np.array([models[a][1].threshold(alpha) for a in (0, 1)])
    alarms = np.greater(J, thresholds)
    return np.asarray(alarms.any(axis=(-2, -1)))


def snapshot_scenario_detection(
    lambdas: NDArray[np.float64], dofs: NDArray[np.int64], alpha: float
) -> float:
    """Closed-form P(alarm) for a scenario whose tests have noncentralities ``lambdas`` and
    degrees of freedom ``dofs`` (independent tests at level alpha each)."""
    thresholds = stats.chi2.isf(alpha, dofs)
    p_quiet = np.where(
        lambdas > 0,
        stats.ncx2.cdf(thresholds, dofs, np.maximum(lambdas, 1e-300)),
        1.0 - alpha,
    )
    return float(-np.expm1(np.log(p_quiet).sum()))
