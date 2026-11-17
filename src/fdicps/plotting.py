"""Matplotlib style and the recurring chart of the project: anomaly signatures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fdicps.dual_anomaly_data_generation.scenarios import Scenario
from fdicps.dual_anomaly_data_generation.sensors import CHANNEL_NAMES

NORMAL_COLOR = "#555555"
FAULT_COLOR = "#2f6db5"
FDI_COLOR = "#c8452c"
CLASS_COLORS = {"normal": NORMAL_COLOR, "fault": FAULT_COLOR, "fdi": FDI_COLOR}
SERIES_COLORS = ("#2f6db5", "#e08a1e", "#3a8f5c")


def use_style() -> None:
    """A clean, consistent style for every figure in the repository (same as CPS-01/02)."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def channel_errors(scenario: Scenario, names: Sequence[str]) -> np.ndarray:
    """Each named channel minus the true value it measures, shape (T, len(names))."""
    cols = []
    for name in names:
        unit, quantity, axis = name.split(".")
        a = 0 if axis == "x" else 1
        truth = {"pos": scenario.truth.p, "vel": scenario.truth.v, "acc": scenario.truth.a}
        cols.append(scenario.values[:, CHANNEL_NAMES.index(name)] - truth[quantity][:, a])
    return np.column_stack(cols)


def signature_grid(
    scenarios: Sequence[Scenario], names: Sequence[str], titles: Sequence[str], ylabel: str
) -> Figure:
    """One panel per scenario: the named channels' errors against the truth over time."""
    n = len(scenarios)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.9 * rows), sharex=True, squeeze=False)
    for ax, s, title in zip(axes.flat, scenarios, titles, strict=False):
        err = channel_errors(s, names)
        for j, name in enumerate(names):
            ax.plot(s.truth.t, err[:, j], lw=0.8, color=SERIES_COLORS[j % 3], label=name)
        if s.onset >= 0:
            ax.axvline(s.onset, color=CLASS_COLORS[s.label], ls="--", lw=1)
        ax.set_title(title, fontsize=9, color=CLASS_COLORS[s.label], loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles, labels, loc="upper center", ncol=len(names), fontsize=8, frameon=False)
    return fig
