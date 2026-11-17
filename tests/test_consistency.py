"""Phase 5: the model-based consistency checks against their closed forms. The Kalman filter's
covariance converges to the DARE solution, its NIS is chi-square, a dynamically consistent
attack leaves its innovations exactly unchanged, and the IEEE 14-bus DC model reproduces the
stealthy-attack theorem and the noncentral detection rate."""

import numpy as np
import pytest
from scipy import stats

from fdicps.adversarial_robustness_analysis.attacks import spoofer_variant
from fdicps.cross_sensor_consistency_model import ieee14
from fdicps.cross_sensor_consistency_model.detection import (
    consistency_feature_names,
    consistency_features,
    cusum,
    cusum_score,
    filter_axes,
    nis_alarm,
    window_alarm,
    window_sums,
)
from fdicps.cross_sensor_consistency_model.kalman import (
    axis_model,
    gains,
    run_filter,
    steady_state_prior,
)
from fdicps.dual_anomaly_data_generation.fdi import FalseDataInjection, injection
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import generate_dataset, make_scenario
from fdicps.dual_anomaly_data_generation.sensors import CHANNEL_NAMES
from fdicps.physical_vs_fdi_classification.residuals import REFERENCE

P = load_sensor_parameters()


@pytest.fixture(scope="module")
def normals() -> np.ndarray:
    d = generate_dataset(range(900))
    return d.values[d.label == 0]


@pytest.mark.parametrize("axis", [0, 1])
def test_riccati_recursion_converges_to_the_dare_solution(axis: int) -> None:
    # The prior covariance recursion's fixed point is the discrete algebraic Riccati equation's
    # solution (SciPy's solve_discrete_are). The bias mode has almost no process noise, so it
    # converges slowly: 20 000 steps bring every element within 1e-9 relative.
    model = axis_model(P, axis)
    dare = steady_state_prior(model)
    P_end = gains(model, 20_000, P.ins_bias_sigma).P_prior[-1]
    np.testing.assert_allclose(P_end, dare, rtol=1e-9)


@pytest.mark.parametrize("axis", [0, 1])
def test_nis_is_chi_square_with_m_dof(normals, axis: int) -> None:
    model = axis_model(P, axis)
    nis = run_filter(normals, P, model).nis[:, REFERENCE:]
    # mean m; the mean of ~72 000 values has standard error sqrt(2m / n) < 0.012
    assert nis.mean() == pytest.approx(model.m, abs=0.05)
    k, n = int((nis > stats.chi2.isf(0.01, model.m)).sum()), nis.size
    lo, hi = stats.binomtest(k, n).proportion_ci(0.999)
    assert lo < 0.01 < hi


def test_consistent_injection_leaves_innovations_unchanged() -> None:
    # The filter is linear, so the innovation change equals the filter's response to the
    # injection alone. A bus-level L3 drag-off is a trajectory of the filter's own model (with
    # the matching acceleration in the INS), so that response is zero.
    clean = make_scenario(4, "normal").values[None]
    fdi = FalseDataInjection(3, "drag", 1, 150, 0.08, 40.0, target=6)
    attacked = clean + injection(fdi, P, P.n_steps)[None]
    model = axis_model(P, 1)
    a, b = run_filter(attacked, P, model), run_filter(clean, P, model)
    np.testing.assert_allclose(a.innovation, b.innovation, atol=1e-8)
    # ... and the estimate is carried away by exactly the injected position offset
    dp = injection(fdi, P, P.n_steps)[:, CHANNEL_NAMES.index("gnss_1.pos.h")]
    np.testing.assert_allclose(a.state[0, :, 0] - b.state[0, :, 0], dp, atol=1e-6)


def test_receivers_only_spoof_breaks_the_inertial_consistency() -> None:
    s = spoofer_variant(2_000_001, "receivers")
    runs = filter_axes(s.values[None], P)
    assert nis_alarm(runs, 0.01)[0] or window_alarm(runs, 0.01)[0]


def test_monitor_false_alarm_rates_on_normals(normals) -> None:
    runs = filter_axes(normals, P)
    for alarm in (nis_alarm(runs, 0.05), window_alarm(runs, 0.05)):
        k, n = int(alarm.sum()), alarm.size
        lo, hi = stats.binomtest(k, n).proportion_ci(0.999)
        assert lo < 0.05 < hi


def test_window_sums_and_cusum() -> None:
    nis = np.zeros((1, REFERENCE + 45))
    nis[0, REFERENCE:] = 1.0
    np.testing.assert_array_equal(window_sums(nis, 20), [[20.0, 20.0]])
    nis[0, REFERENCE + 10 :] = 10.0  # m = 2, drift 0.5 m: excess 10 - 3 = 7 per sample
    g = cusum(nis, 2)
    assert g[0, 9] == 0.0 and g[0, 10] == pytest.approx(7.0) and g[0, -1] == pytest.approx(7 * 35)


def test_consistency_features_shape_and_scores(normals) -> None:
    X = consistency_features(normals[:5], P)
    assert X.shape == (5, len(consistency_feature_names())) == (5, 14)
    assert np.all(np.isfinite(X))
    assert cusum_score(filter_axes(normals[:5], P)).shape == (5,)


def test_filter_without_baro() -> None:
    model = axis_model(P, 1, include=[CHANNEL_NAMES.index("gnss_1.pos.h")])
    assert model.m == 1
    runs = filter_axes(
        make_scenario(1, "normal").values[None], P, exclude=[CHANNEL_NAMES.index("baro.pos.h")]
    )
    assert runs.dofs == {0: 4, 1: 4}


# ---------- IEEE 14-bus ----------


def test_ieee14_model_dimensions_and_observability() -> None:
    H = ieee14.measurement_matrix()
    assert H.shape == (34, 13)
    assert np.linalg.matrix_rank(H) == 13
    assert ieee14.model().dof == 21
    # injections are the sums of the flows leaving each bus: Kirchhoff, so they sum to zero
    np.testing.assert_allclose(H[20:].sum(axis=0), 0.0, atol=1e-9)


def test_ieee14_branch_data_matches_matpower_case14() -> None:
    br = ieee14.branches()
    assert len(br) == 20
    assert br[0] == (1, 2, 0.05917, 1.0)  # MATPOWER case14, first branch
    assert br[7] == (4, 7, 0.20912, 0.978)  # a transformer branch with its tap ratio


def test_ieee14_stealthy_attack_leaves_the_residual_unchanged() -> None:
    # Liu, Ning & Reiter (2009), on the system their paper studied
    model = ieee14.model()
    rng = np.random.default_rng(0)
    z = rng.normal(size=model.m) * model.sigma
    c = rng.normal(scale=0.2, size=model.n)
    np.testing.assert_allclose(model.residual(z + model.H @ c), model.residual(z), atol=1e-10)
    np.testing.assert_allclose(model.estimate(z + model.H @ c) - model.estimate(z), c, atol=1e-10)


def test_ieee14_sparse_attack_detected_at_the_noncentral_rate() -> None:
    model = ieee14.model()
    a = np.zeros(model.m)
    a[[3, 12, 25]] = [0.03, -0.025, 0.035]  # three meters, a few per-unit hundredths
    p = float(model.detection_probability(a, 0.01))
    assert 0.2 < p < 0.95
    rng = np.random.default_rng(1)
    z = rng.normal(size=(40_000, model.m)) * model.sigma + a
    k = int((model.statistic(z) > model.threshold(0.01)).sum())
    lo, hi = stats.binomtest(k, 40_000).proportion_ci(0.999)
    assert lo < p < hi


def test_ieee14_meters_needed_to_move_one_state() -> None:
    need = ieee14.meters_to_move_one_state()
    assert len(need) == 13
    # bus 8 hangs off bus 7 alone: its flow and the two end injections
    assert need[8 - 2] == 3
    assert min(need) == 3
