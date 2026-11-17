"""The navigation chain's model-based consistency checks.

Snapshot check: on each axis, the position and velocity channels obey z = H [p, v] + e
(``sensors.measurement_model``), so WLS with the chi-square test runs at every sample. Its
degrees of freedom are 4 - 2 = 2 along-track and 5 - 2 = 3 vertically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fdicps.cross_sensor_consistency_model.wls import WLS
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.dual_anomaly_data_generation.sensors import measurement_model, nominal_sigma


def snapshot_models(params: SensorParameters) -> dict[int, tuple[list[int], WLS]]:
    """Per axis: the channel indices and the WLS model of the snapshot check."""
    sigma = nominal_sigma(params)
    out = {}
    for axis in (0, 1):
        idx, H = measurement_model(axis)
        out[axis] = (idx, WLS(H, sigma[idx]))
    return out


def snapshot_statistics(
    values: NDArray[np.float64], params: SensorParameters
) -> NDArray[np.float64]:
    """J at every sample on both axes; ``values`` (..., T, 13) -> (..., T, 2)."""
    cols = [model.statistic(values[..., idx]) for idx, model in snapshot_models(params).values()]
    return np.stack(cols, axis=-1)
