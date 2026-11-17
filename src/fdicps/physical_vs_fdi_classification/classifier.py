"""The physics-informed classifier: normal, physical fault or false data injection.

A histogram gradient-boosting classifier (scikit-learn) on the features of :mod:`features`.
Gradient boosting suits this problem: the features are heterogeneous, the useful boundaries
are thresholds ("frozen for more than 10 % of the segment", "kinematically inconsistent by
more than 6 standard deviations"), and their interactions matter. Fixed seed, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.physical_vs_fdi_classification.features import physics_features


@dataclass
class PhysicsClassifier:
    seed: int = 0
    max_iter: int = 300
    learning_rate: float = 0.1
    model: HistGradientBoostingClassifier | None = field(default=None, repr=False)

    def fit_features(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> PhysicsClassifier:
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            early_stopping=False,
        )
        self.model.fit(X, y)
        return self

    def fit(
        self, values: NDArray[np.float64], y: NDArray[np.int64], params: SensorParameters
    ) -> PhysicsClassifier:
        return self.fit_features(physics_features(values, params), y)

    def predict_proba_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.model is not None, "call fit() first"
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def predict_features(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.asarray(self.predict_proba_features(X).argmax(axis=1), dtype=np.int64)

    def predict(self, values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.int64]:
        return self.predict_features(physics_features(values, params))
