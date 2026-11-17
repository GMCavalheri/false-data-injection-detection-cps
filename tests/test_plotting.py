"""Smoke tests: the figures build without error and show what they claim."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fdicps import plotting  # noqa: E402
from fdicps.dual_anomaly_data_generation.scenarios import make_scenario  # noqa: E402


def test_channel_errors_are_zero_mean_noise_for_a_normal_scenario() -> None:
    s = make_scenario(3, "normal")
    err = plotting.channel_errors(s, ["gnss_1.pos.h", "baro.pos.h"])
    assert err.shape == (s.truth.t.size, 2)
    assert np.all(np.abs(err.mean(axis=0)) < 2.0)


def test_signature_grid_has_one_panel_per_scenario() -> None:
    plotting.use_style()
    scen = [make_scenario(1, "normal"), make_scenario(1, "fdi", level=1), make_scenario(1, "fault")]
    fig = plotting.signature_grid(scen, ["gnss_1.pos.x"], ["a", "b", "c"], "m")
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert len(visible[1].lines) == 2  # the channel and the onset marker
    plt.close(fig)
