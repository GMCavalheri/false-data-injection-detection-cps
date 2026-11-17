"""Weighted least squares state estimation and the chi-square bad-data test.

For a linear measurement model z = H x + e with e ~ N(0, R), R diagonal and W = R^-1:

* estimate      x_hat = (H^T W H)^-1 H^T W z;
* residual      r = z - H x_hat = S z, with S = I - H (H^T W H)^-1 H^T W;
* test          J = r^T W r ~ chi2(m - n) when the model holds (m measurements, n states);
* under an additive attack a, J ~ noncentral chi2(m - n, lambda), lambda = a^T W S a.

S H = 0, so any a = H c gives lambda = 0: the attack is invisible to the test whatever its
size (Liu, Ning & Reiter, 2009). The same code serves the navigation chain's snapshot check
and the IEEE 14-bus DC model of Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class WLS:
    H: NDArray[np.float64]  # (m, n)
    sigma: NDArray[np.float64]  # (m,) measurement noise standard deviations
    _gain: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        W = np.diag(1.0 / self.sigma**2)
        gain = np.linalg.solve(self.H.T @ W @ self.H, self.H.T @ W)  # (n, m)
        object.__setattr__(self, "_gain", gain)

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def n(self) -> int:
        return int(self.H.shape[1])

    @property
    def dof(self) -> int:
        return self.m - self.n

    @property
    def W(self) -> NDArray[np.float64]:
        return np.diag(1.0 / self.sigma**2)

    @property
    def S(self) -> NDArray[np.float64]:
        """Residual operator I - H (H^T W H)^-1 H^T W."""
        return np.eye(self.m) - self.H @ self._gain

    def estimate(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """State estimates for measurements z of shape (..., m)."""
        return np.asarray(z @ self._gain.T)

    def residual(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(z - self.estimate(z) @ self.H.T)

    def statistic(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """J = r^T W r for every measurement vector; shape (...)."""
        r = self.residual(z)
        return np.asarray(((r / self.sigma) ** 2).sum(axis=-1))

    def threshold(self, false_alarm: float) -> float:
        return float(stats.chi2.isf(false_alarm, self.dof))

    def noncentrality(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        """lambda = a^T W S a for attack vectors a of shape (..., m)."""
        sa = a @ self.S.T
        return np.asarray(((sa / self.sigma) ** 2).sum(axis=-1))

    def detection_probability(
        self, a: NDArray[np.float64], false_alarm: float
    ) -> NDArray[np.float64]:
        """P(J > threshold) under attack a: the noncentral chi-square tail."""
        lam = self.noncentrality(a)
        t = self.threshold(false_alarm)
        return np.asarray(
            np.where(lam > 0, stats.ncx2.sf(t, self.dof, np.maximum(lam, 1e-300)), false_alarm)
        )
