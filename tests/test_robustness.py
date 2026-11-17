"""Phase 4: the robustness analysis rests on closed forms. A statically consistent attack is
detected at exactly the false-alarm rate; any other is detected at the noncentral
chi-square rate; detectability never rises with stealth; the scenario monitor's false-alarm
rate is exact; the attack sets are paired and do what they claim."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from fdicps.adversarial_robustness_analysis.attacks import (
    CONSISTENCY_GRID,
    consistency_variant,
    exact_mimic,
    near_mimic,
    receiver_position_channels,
    spoofer_variant,
)
from fdicps.adversarial_robustness_analysis.monitors import (
    per_test_level,
    snapshot_alarm,
    snapshot_scenario_detection,
)
from fdicps.cross_sensor_consistency_model.navigation import snapshot_models
from fdicps.cross_sensor_consistency_model.wls import WLS
from fdicps.dual_anomaly_data_generation.fdi import injection
from fdicps.dual_anomaly_data_generation.parameters import (
    load_anomaly_ranges,
    load_sensor_parameters,
)
from fdicps.dual_anomaly_data_generation.scenarios import make_scenario, sample_fdi
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS

P = load_sensor_parameters()
RANGES = load_anomaly_ranges()
MODELS = snapshot_models(P)


def _mc_detection(model: WLS, a: np.ndarray, alpha: float, n: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, model.m)) * model.sigma + a
    return int((model.statistic(z) > model.threshold(alpha)).sum()), n


@pytest.mark.parametrize("axis", [0, 1])
def test_wls_statistic_is_chi_square_with_m_minus_n_dof(axis: int) -> None:
    idx, model = MODELS[axis]
    assert model.dof == (2 if axis == 0 else 3)
    k, n = _mc_detection(model, np.zeros(model.m), 0.05, 40_000, axis)
    lo, hi = stats.binomtest(k, n).proportion_ci(0.999)
    assert lo < 0.05 < hi


def test_static_consistent_attack_is_detected_at_the_false_alarm_rate() -> None:
    # Liu, Ning & Reiter (2009): a = H c gives lambda = 0, so P(detect) = alpha for any c.
    idx, model = MODELS[1]
    a = model.H @ np.array([500.0, 0.0])
    assert model.noncentrality(a) == pytest.approx(0.0, abs=1e-18)
    assert model.detection_probability(a, 0.01) == pytest.approx(0.01)
    k, n = _mc_detection(model, a, 0.01, 40_000, 7)
    lo, hi = stats.binomtest(k, n).proportion_ci(0.999)
    assert lo < 0.01 < hi


def test_single_channel_attack_matches_the_noncentral_chi_square() -> None:
    idx, model = MODELS[1]
    a = np.zeros(model.m)
    a[0] = 18.0  # 18 m on one receiver's altitude
    p = float(model.detection_probability(a, 0.01))
    assert 0.2 < p < 0.95  # an informative regime, not a saturated one
    k, n = _mc_detection(model, a, 0.01, 40_000, 8)
    lo, hi = stats.binomtest(k, n).proportion_ci(0.999)
    assert lo < p < hi


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), s1=st.floats(0, 1), s2=st.floats(0, 1))
def test_detectability_never_rises_with_stealth(seed: int, s1: float, s2: float) -> None:
    # noncentrality of a(s) is (1 - s)^2 times that of the single-channel attack (Lesson 1),
    # so it is non-increasing in s at every sample
    lo_s, hi_s = sorted((s1, s2))
    base = sample_fdi(np.random.default_rng(seed), P, RANGES, 1)
    idx, model = MODELS[base.axis]
    lam = [
        model.noncentrality(injection(replace(base, consistency=s), P, P.n_steps)[:, idx])
        for s in (lo_s, hi_s)
    ]
    assert np.all(lam[1] <= lam[0] * (1 + 1e-9) + 1e-12)
    np.testing.assert_allclose(
        lam[1] * (1 - lo_s) ** 2, lam[0] * (1 - hi_s) ** 2, rtol=1e-9, atol=1e-9
    )


def test_per_test_level_gives_an_exact_scenario_rate() -> None:
    alpha = per_test_level(0.01, 600)
    assert 1 - (1 - alpha) ** 600 == pytest.approx(0.01, rel=1e-12)


def test_snapshot_alarm_false_alarm_rate_on_normal_scenarios() -> None:
    values = np.stack([make_scenario(s, "normal").values for s in range(400)])
    k = int(snapshot_alarm(values, P, 0.05).sum())
    lo, hi = stats.binomtest(k, 400).proportion_ci(0.999)
    assert lo < 0.05 < hi


def test_scenario_detection_closed_form() -> None:
    alpha = 0.001
    assert snapshot_scenario_detection(np.zeros(10), np.full(10, 2), alpha) == pytest.approx(
        1 - (1 - alpha) ** 10
    )
    one = snapshot_scenario_detection(np.array([20.0]), np.array([3]), alpha)
    assert one == pytest.approx(stats.ncx2.sf(stats.chi2.isf(alpha, 3), 3, 20.0))


def test_consistency_variants_are_paired_and_keep_the_target_effect() -> None:
    a, b = consistency_variant(5, "offset", 0.0), consistency_variant(5, "offset", 1.0)
    assert a.anomaly.target == b.anomaly.target and a.anomaly.onset == b.anomaly.onset
    np.testing.assert_allclose(a.values[:, a.anomaly.target], b.values[:, b.anomaly.target])
    assert (
        CONSISTENCY_GRID[0] == 0.0 and CONSISTENCY_GRID[-1] == 1.0 and len(CONSISTENCY_GRID) == 11
    )
    assert consistency_variant(5, "drag", 1.0).anomaly.pattern == "drag"


def test_near_mimic_is_a_fault_sized_step_on_one_receiver() -> None:
    s = near_mimic(11)
    assert s.anomaly.ramp == 0.0 and s.anomaly.consistency == 0.0
    assert s.anomaly.target in receiver_position_channels()
    assert RANGES["gnss_bias"].low <= abs(s.anomaly.magnitude) <= RANGES["gnss_bias"].high


def test_exact_mimic_is_a_receiver_bias_labelled_fdi() -> None:
    s = exact_mimic(12)
    clean = make_scenario(12, "normal").values
    changed = np.flatnonzero(np.any(s.values != clean, axis=0))
    units = {CHANNELS[i].unit for i in changed}
    assert s.label == "fdi" and len(units) == 1 and units <= {"gnss_1", "gnss_2"}
    assert {CHANNELS[i].quantity for i in changed} == {"pos"}


def test_spoofer_reach() -> None:
    rf, bus = spoofer_variant(13, "receivers"), spoofer_variant(13, "bus")
    assert not rf.anomaly.include_ins and not rf.anomaly.include_baro
    assert bus.anomaly.include_ins and bus.anomaly.include_baro
