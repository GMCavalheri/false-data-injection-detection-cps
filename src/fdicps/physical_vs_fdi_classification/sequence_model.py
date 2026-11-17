"""A small 1-D convolutional network on the residual sequences (PyTorch, CPU).

The gradient-boosting classifier sees hand-made summaries (maxima, end values, step sizes).
This model sees the residual sequences themselves (:func:`residuals.residual_sequences`) and
learns its own summaries: three dilated convolutions widen the receptive field to about 50 s,
then global max and mean pooling over time feed a linear layer with three outputs. Comparing
the two says whether the hand-made summaries throw away anything useful.

Reproducibility: the model is seeded, runs on one CPU thread with deterministic algorithms,
and the batches are drawn from a seeded generator, so the same inputs give the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from fdicps.dual_anomaly_data_generation.parameters import SensorParameters
from fdicps.physical_vs_fdi_classification.residuals import residual_sequences

CLIP = 50.0  # residuals are in noise standard deviations; clip the rare huge ones


def prepare(values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.float32]:
    """Residual sequences as a float32 (N, C, L) array, clipped to +/- CLIP."""
    seq, _ = residual_sequences(values, params)
    return np.clip(np.moveaxis(seq, -1, -2), -CLIP, CLIP).astype(np.float32)


class ResidualCNN(nn.Module):
    def __init__(self, channels: int, hidden: int = 32, classes: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = channels
        for dilation in (1, 2, 4):
            layers += [
                nn.Conv1d(width, hidden, 7, dilation=dilation, padding=3 * dilation),
                nn.ReLU(),
            ]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        pooled = torch.cat([h.amax(dim=-1), h.mean(dim=-1)], dim=-1)
        out: torch.Tensor = self.head(pooled)
        return out


def _deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@dataclass
class SequenceClassifier:
    seed: int = 0
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    model: ResidualCNN | None = field(default=None, repr=False)
    loss_history: list[float] = field(default_factory=list)

    def fit_prepared(self, x: NDArray[np.float32], y: NDArray[np.int64]) -> SequenceClassifier:
        _deterministic(self.seed)
        self.model = ResidualCNN(x.shape[1])
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
        rng = np.random.default_rng(self.seed)
        self.loss_history = []
        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(xt))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = torch.from_numpy(order[start : start + self.batch_size])
                opt.zero_grad()
                loss = loss_fn(self.model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            self.loss_history.append(total / len(xt))
        return self

    def fit(
        self, values: NDArray[np.float64], y: NDArray[np.int64], params: SensorParameters
    ) -> SequenceClassifier:
        return self.fit_prepared(prepare(values, params), y)

    def predict_proba_prepared(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        assert self.model is not None, "call fit() first"
        _deterministic(self.seed)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x))
        return np.asarray(torch.softmax(logits, dim=-1).numpy(), dtype=np.float64)

    def predict_prepared(self, x: NDArray[np.float32]) -> NDArray[np.int64]:
        return np.asarray(self.predict_proba_prepared(x).argmax(axis=1), dtype=np.int64)

    def predict(self, values: NDArray[np.float64], params: SensorParameters) -> NDArray[np.int64]:
        return self.predict_prepared(prepare(values, params))
