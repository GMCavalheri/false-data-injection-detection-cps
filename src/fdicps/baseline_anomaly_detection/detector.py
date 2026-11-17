"""The undifferentiated baseline: an Isolation Forest trained on normal flight only.

It learns what normal telemetry looks like (novelty detection) and returns one number per
scenario, larger meaning more anomalous. By construction it cannot say whether an anomaly is
a fault or an attack: that is the limitation Phase 2 measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from fdicps.baseline_anomaly_detection.features import generic_features
from fdicps.dual_anomaly_data_generation.parameters import SensorParameters


@dataclass
class BaselineDetector:
    n_estimators: int = 300
    seed: int = 0
    false_alarm_rate: float = 0.01
    forest: IsolationForest | None = field(default=None, repr=False)
    threshold: float = float("nan")

    def fit(self, normal_values: NDArray[np.float64], params: SensorParameters) -> BaselineDetector:
        """Train on normal scenarios only; set the alarm threshold at the (1 - far) quantile
        of their own scores."""
        X = generic_features(normal_values, params)
        self.forest = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed)
        self.forest.fit(X)
        self.threshold = float(np.quantile(self._score(X), 1.0 - self.false_alarm_rate))
        return self

    def _score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.forest is not None, "call fit() first"
        return np.asarray(-self.forest.score_samples(X), dtype=float)

    def score(self, values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float64]:
        return self._score(generic_features(values, params))

    def alarm(self, values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.bool_]:
        return np.asarray(self.score(values, params) > self.threshold)
