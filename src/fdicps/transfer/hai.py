"""Transfer of the detection layer to public ICS testbed data: the HAI dataset.

HAI (HIL-based Augmented ICS security dataset, the Affiliated Institute of ETRI,
https://github.com/icsdataset/hai) records a boiler, turbine and water-treatment testbed with a
hardware-in-the-loop simulator at 1 Hz, with labelled attacks. Version 21.03 is used: plain
gzip CSV files, a whole-system ``attack`` label and per-process labels ``attack_P1..P3``.
The data are never committed or redistributed; :mod:`fdicps.transfer.fetch` downloads them,
pinned to one commit of the dataset repository and verified against their git blob hashes.

There is no physical model of that plant here, so the two consistency checks of this project
are rebuilt from data:

* **static** (the analogue of the WLS bad-data test): the tags' normal linear relations are
  the principal subspace of their standardised training values; the squared prediction error
  SPE = ||(I - V V^T) z||^2 is the residual outside it (Jackson & Mudholkar, 1979);
* **dynamic** (the analogue of the Kalman innovation): a linear one-step predictor of the
  principal scores, fitted by ridge least squares; its innovation's Mahalanobis norm.

Thresholds are empirical quantiles on training data (normal operation only). HAI labels
attacks but not physical faults, so only *detection* can transfer; fault-versus-attack
attribution cannot be evaluated on it, and no claim is made about it.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DATASET_COMMIT = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
VERSION = "hai-21.03"
# file -> (size in bytes, git blob SHA-1) at DATASET_COMMIT, from the GitHub contents API
FILES = {
    "train1.csv.gz": (30_651_421, "2940eeeb2018c68fc01586da77986a5771b12585"),
    "test1.csv.gz": (6_149_549, "7dee74e72e440cbf2d18fd61fc8de3a7066dda29"),
    "test2.csv.gz": (16_765_328, "bef13733742259bbd54912cde2ee6ec5fcb2fe54"),
}
LABEL_PREFIX = "attack"
VARIANCE_KEPT = 0.99
RIDGE = 1e-3
SMOOTH = 10  # samples (seconds) of moving average on both statistics


def git_blob_sha1(data: bytes) -> str:
    """The SHA-1 git stores for a file: sha1(b"blob <size>\\0" + content)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def verify(path: Path) -> None:
    size, sha = FILES[path.name]
    data = path.read_bytes()
    if len(data) != size or git_blob_sha1(data) != sha:
        raise ValueError(f"{path.name} does not match HAI {VERSION} at {DATASET_COMMIT[:7]}")


@dataclass(frozen=True)
class HaiTable:
    tags: list[str]
    values: NDArray[np.float64]  # (T, n_tags)
    attack: NDArray[np.bool_]  # (T,) whole-system label
    process_labels: dict[str, NDArray[np.bool_]]  # attack_P1.. -> (T,)


def read_table(path: Path) -> HaiTable:
    """Read one HAI CSV (gzip or plain): first column a timestamp, then tags, then labels."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",")[1:] for line in f if line.strip()]
    names = header[1:]
    data = np.array(rows, dtype=float)
    labels = [i for i, n in enumerate(names) if n.lower().startswith(LABEL_PREFIX)]
    tags = [i for i in range(len(names)) if i not in labels]
    by_name = {names[i]: data[:, i] > 0 for i in labels}
    attack = by_name.pop(next(n for n in by_name if n.lower() == LABEL_PREFIX))
    return HaiTable([names[i] for i in tags], data[:, tags], attack, by_name)


def moving_average(x: NDArray[np.float64], n: int = SMOOTH) -> NDArray[np.float64]:
    """Trailing moving average (causal), same length as x; the first n-1 use what exists."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    k = np.arange(1, x.size + 1)
    lo = np.maximum(0, k - n)
    return np.asarray((c[k] - c[lo]) / (k - lo))


@dataclass
class DataDrivenConsistency:
    """The static (PCA residual) and dynamic (one-step predictor) checks, fitted on normal data."""

    variance_kept: float = VARIANCE_KEPT
    ridge: float = RIDGE
    keep: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    scale: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    V: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    A: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    innov_inv: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    thresholds: dict[str, float] = field(default_factory=dict)

    def _standardise(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((values[:, self.keep] - self.mean) / self.scale)

    def fit(self, values: NDArray[np.float64], quantile: float = 0.999) -> DataDrivenConsistency:
        sd = values.std(axis=0)
        self.keep = sd > 1e-9  # constant tags carry no relation
        self.mean, self.scale = values[:, self.keep].mean(axis=0), sd[self.keep]
        z = self._standardise(values)
        _, s, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
        share = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(share, self.variance_kept) + 1)
        self.V = vt[:k].T
        t = z @ self.V
        X, Y = np.column_stack([t[:-1], np.ones(len(t) - 1)]), t[1:]
        self.A = np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ Y)
        e = Y - X @ self.A
        self.innov_inv = np.linalg.inv(np.cov(e, rowvar=False) + 1e-9 * np.eye(k))
        stats = self.statistics(values)
        self.thresholds = {name: float(np.quantile(v, quantile)) for name, v in stats.items()}
        return self

    @property
    def n_components(self) -> int:
        return int(self.V.shape[1])

    def statistics(self, values: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Smoothed SPE (static) and innovation norm (dynamic), each of length T."""
        z = self._standardise(values)
        t = z @ self.V
        spe = ((z - t @ self.V.T) ** 2).sum(axis=1)
        pred = np.column_stack([t[:-1], np.ones(len(t) - 1)]) @ self.A
        e = t[1:] - pred
        innov = np.concatenate([[0.0], np.einsum("ti,ij,tj->t", e, self.innov_inv, e)])
        return {
            "static (PCA residual)": moving_average(spe),
            "dynamic (one-step innovation)": moving_average(innov),
        }

    def alarms(self, values: NDArray[np.float64]) -> dict[str, NDArray[np.bool_]]:
        return {k: np.asarray(v > self.thresholds[k]) for k, v in self.statistics(values).items()}


def events(label: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Contiguous runs of True as (start, end) half-open intervals."""
    d = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def event_recall(alarm: NDArray[np.bool_], label: NDArray[np.bool_], grace: int = 0) -> float:
    """Share of attack events with at least one alarm inside them (or within ``grace`` after)."""
    ev = events(label)
    if not ev:
        return float("nan")
    hit = [alarm[s : min(len(alarm), e + grace)].any() for s, e in ev]
    return float(np.mean(hit))


def false_alarm_episodes_per_hour(alarm: NDArray[np.bool_], label: NDArray[np.bool_]) -> float:
    """Alarm episodes that start outside any attack, per hour of normal operation (1 Hz)."""
    starts = [s for s, _ in events(alarm) if not label[s]]
    hours = float((~label).sum()) / 3600.0
    return len(starts) / hours if hours else float("nan")
