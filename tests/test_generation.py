"""Phase 1: the generator obeys its own physics, and each anomaly has the signature its
definition says (closed forms, exact to rounding, or Monte Carlo inside a confidence
interval)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from fdicps.dual_anomaly_data_generation.faults import PhysicalFault, apply_fault
from fdicps.dual_anomaly_data_generation.fdi import (
    FalseDataInjection,
    affected_channels,
    cps02_leaf,
    injection,
)
from fdicps.dual_anomaly_data_generation.mission import (
    MAX_ACCELERATION,
    integrate,
    mission_profile,
    transition,
)
from fdicps.dual_anomaly_data_generation.parameters import (
    load_anomaly_ranges,
    load_sensor_parameters,
    parameter_bases,
)
from fdicps.dual_anomaly_data_generation.scenarios import (
    family,
    generate_dataset,
    make_scenario,
    sample_fdi,
)
from fdicps.dual_anomaly_data_generation.sensors import (
    CHANNEL_NAMES,
    CHANNELS,
    clean_signal,
    measure,
    measurement_model,
    nominal_sigma,
)

P = load_sensor_parameters()
RANGES = load_anomaly_ranges()


def test_every_parameter_states_its_basis() -> None:
    bases = parameter_bases()
    assert bases and all(b.startswith(("assumption", '"assumption')) or b for b in bases.values())
    assert all(b.strip() for b in bases.values())


def test_integrate_equals_the_state_space_recursion() -> None:
    # x[k+1] = F x[k] + G a[k] is the exact zero-order-hold solution; integrate() is a
    # vectorised version of it and must agree to rounding.
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    p, v = integrate(10.0, 3.0, a, 0.5)
    F, G = transition(0.5)
    x = np.array([10.0, 3.0])
    for k in range(199):
        x = F @ x + G * a[k]
        assert x == pytest.approx([p[k + 1], v[k + 1]], rel=1e-12, abs=1e-9)


def test_truth_satisfies_the_kinematics_and_its_envelope() -> None:
    F, G = transition(P.dt)
    for seed in range(20):
        truth = mission_profile(np.random.default_rng(seed), P)
        for axis in range(2):
            x = np.column_stack([truth.p[:, axis], truth.v[:, axis]])
            pred = x[:-1] @ F.T + np.outer(truth.a[:-1, axis], G)
            np.testing.assert_allclose(pred, x[1:], rtol=1e-12, atol=1e-8)
        # the manoeuvre targets bound the velocities (raised-cosine pulses never overshoot)
        assert np.all((truth.v[:, 0] >= 180 - 1e-9) & (truth.v[:, 0] <= 250 + 1e-9))
        assert np.all(np.abs(truth.v[:, 1]) <= 15 + 1e-9)
        assert np.all(truth.p[:, 1] > 0)
        assert np.abs(truth.a).max() <= MAX_ACCELERATION + 1e-12


def test_sensor_noise_has_the_stated_standard_deviation() -> None:
    # sample variance of n iid N(0, sigma^2): (n-1) s^2 / sigma^2 ~ chi2(n-1); two-sided 99.9 %
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(40):
        truth = mission_profile(rng, P)
        tel = measure(truth, P, rng)
        errs.append(tel.values - clean_signal(truth, tel.ins_bias))
    e = np.concatenate(errs)
    n = e.shape[0]
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], n - 1) / (n - 1)
    ratio = e.var(axis=0, ddof=1) / nominal_sigma(P) ** 2
    assert np.all((ratio > lo) & (ratio < hi)), dict(zip(CHANNEL_NAMES, ratio, strict=True))


def test_measurement_model_rows() -> None:
    idx, H = measurement_model(1)
    assert [CHANNELS[i].name for i in idx] == [
        "gnss_1.pos.h",
        "gnss_1.vel.h",
        "gnss_2.pos.h",
        "gnss_2.vel.h",
        "baro.pos.h",
    ]
    np.testing.assert_array_equal(H, [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    assert len(measurement_model(0)[0]) == 4


# ---------- physical faults ----------

BASE = make_scenario(7, "normal")


def _fault(kind, unit="gnss_1", magnitude=1.0, **kw):
    f = PhysicalFault(unit, kind, 150, magnitude, (1.0, -1.0), **kw)
    return f, apply_fault(BASE.values, f, P, np.random.default_rng(0))


def test_bias_is_an_exact_step_on_the_receiver_positions_only() -> None:
    f, out = _fault("bias", magnitude=40.0)
    d = out - BASE.values
    assert np.all(d[:150] == 0)
    np.testing.assert_allclose(d[150:, CHANNEL_NAMES.index("gnss_1.pos.x")], 40.0)
    np.testing.assert_allclose(d[150:, CHANNEL_NAMES.index("gnss_1.pos.h")], -40.0)
    assert np.count_nonzero(np.any(d != 0, axis=0)) == 2  # velocity untouched


def test_drift_slope_equals_the_rate() -> None:
    f, out = _fault("drift", unit="baro", magnitude=0.2)
    d = (out - BASE.values)[:, CHANNEL_NAMES.index("baro.pos.h")]
    # baro is on the vertical axis, where this fault's sign is -1; differences of ~7 km
    # altitudes carry ~1e-12 m of rounding
    np.testing.assert_allclose(np.diff(d[150:]), -0.2 * P.dt, atol=1e-9)


def test_stuck_unit_has_zero_increments() -> None:
    f, out = _fault("stuck", unit="ins_2")
    for name in ("ins_2.acc.x", "ins_2.acc.h"):
        col = out[149:, CHANNEL_NAMES.index(name)]
        assert np.all(np.diff(col) == 0.0)


def test_noise_fault_multiplies_the_standard_deviation() -> None:
    # extra noise of std sigma*sqrt(k^2-1) makes the total std k*sigma (independent Gaussians)
    k, j = 5.0, CHANNEL_NAMES.index("gnss_2.vel.x")
    errs = []
    for seed in range(30):
        s = make_scenario(seed, "normal")
        f = PhysicalFault("gnss_2", "noise", 0, k)
        out = apply_fault(s.values, f, P, np.random.default_rng(seed + 100))
        errs.append(out[:, j] - s.truth.v[:, 0])
    e = np.concatenate(errs)
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], e.size - 1) / (e.size - 1)
    assert lo < e.var(ddof=1) / (k * P.gnss_vel_sigma) ** 2 < hi


def test_common_mode_is_shared_and_saturates() -> None:
    f, out = _fault("common_mode", unit="gnss", magnitude=0.25, cap=12.0)
    d = out - BASE.values
    g1 = d[:, CHANNEL_NAMES.index("gnss_1.pos.x")]
    g2 = d[:, CHANNEL_NAMES.index("gnss_2.pos.x")]
    np.testing.assert_array_equal(g1, g2)
    assert g1.max() == pytest.approx(12.0)
    assert g1[150 + 10] == pytest.approx(2.5)


# ---------- false data injection ----------


def _wls_residual_projection(axis: int) -> tuple[list[int], np.ndarray]:
    idx, H = measurement_model(axis)
    W = np.diag(1.0 / nominal_sigma(P)[idx] ** 2)
    K = H @ np.linalg.solve(H.T @ W @ H, H.T @ W)
    return idx, np.eye(len(idx)) - K


@pytest.mark.parametrize("axis", [0, 1])
def test_l2_injection_lies_in_the_range_of_h(axis: int) -> None:
    # Liu, Ning & Reiter (2009): a = H c leaves every WLS residual unchanged.
    rng = np.random.default_rng(axis)
    fdi = sample_fdi(rng, P, RANGES, 2)
    fdi = FalseDataInjection(
        **{**fdi.__dict__, "axis": axis, "target": measurement_model(axis)[0][0]}
    )
    a = injection(fdi, P, P.n_steps)
    idx, S = _wls_residual_projection(axis)
    assert np.abs(a[:, idx]).max() > 10
    np.testing.assert_allclose(a[:, idx] @ S.T, 0.0, atol=1e-12)


def test_l1_injection_touches_one_channel_and_is_seen_by_the_residual() -> None:
    fdi = sample_fdi(np.random.default_rng(3), P, RANGES, 1)
    a = injection(fdi, P, P.n_steps)
    assert affected_channels(fdi) == [fdi.target]
    assert np.flatnonzero(np.any(a != 0, axis=0)).tolist() == [fdi.target]
    idx, S = _wls_residual_projection(fdi.axis)
    assert np.abs(a[:, idx] @ S.T).max() > 1.0


def test_l3_drag_obeys_the_aircraft_kinematics_exactly() -> None:
    fdi = FalseDataInjection(3, "drag", 0, 120, 0.05, 40.0, target=0)
    a = injection(fdi, P, P.n_steps)
    F, G = transition(P.dt)
    dp = a[:, CHANNEL_NAMES.index("gnss_1.pos.x")]
    dv = a[:, CHANNEL_NAMES.index("gnss_2.vel.x")]
    da = a[:, CHANNEL_NAMES.index("ins_1.acc.x")]
    x = np.column_stack([dp, dv])
    np.testing.assert_allclose(x[:-1] @ F.T + np.outer(da[:-1], G), x[1:], atol=1e-12)
    assert da.max() == pytest.approx(0.05)
    assert dv[-1] == pytest.approx(0.05 * 40.0)  # velocity offset = da x duration
    assert cps02_leaf(fdi) == "inject_false_nav_messages"


def test_gnss_spoof_variant_leaves_ins_and_baro_untouched() -> None:
    fdi = FalseDataInjection(
        3, "drag", 1, 120, 0.05, 40.0, target=6, include_baro=False, include_ins=False
    )
    touched = {CHANNELS[i].unit for i in affected_channels(fdi)}
    assert touched == {"gnss_1", "gnss_2"}
    assert cps02_leaf(fdi) == "gnss_spoof"


def test_l0_replacement_is_noise_free() -> None:
    s = make_scenario(11, "fdi", level=0)
    j, k0 = s.anomaly.target, s.anomaly.onset
    second = np.diff(s.values[k0:, j], 2)
    assert np.abs(second).max() < 1e-6  # a straight line (+ a constant jump) has none


@settings(max_examples=40, deadline=None)
@given(s=st.floats(0.0, 1.0), seed=st.integers(0, 10_000))
def test_consistency_keeps_the_target_effect_and_shrinks_the_residual(s: float, seed: int) -> None:
    # a(s) = s A + (1 - s) A_target: the target channel's injection does not depend on s, and
    # the WLS residual of a(s) is exactly (1 - s) times that of the single-channel attack.
    base = sample_fdi(np.random.default_rng(seed), P, RANGES, 1)
    one = injection(base, P, P.n_steps)
    blend = injection(FalseDataInjection(**{**base.__dict__, "consistency": s}), P, P.n_steps)
    np.testing.assert_allclose(blend[:, base.target], one[:, base.target], atol=1e-12)
    idx, S = _wls_residual_projection(base.axis)
    np.testing.assert_allclose(blend[:, idx] @ S.T, (1 - s) * (one[:, idx] @ S.T), atol=1e-9)


def test_invalid_fdi_is_rejected() -> None:
    with pytest.raises(ValueError):
        FalseDataInjection(1, "offset", 0, 100, 50.0, 10.0, target=0, consistency=1.5)
    with pytest.raises(ValueError):
        FalseDataInjection(1, "offset", 0, 100, 50.0, 10.0, target=6)  # target on the other axis
    with pytest.raises(ValueError):
        sample_fdi(np.random.default_rng(0), P, RANGES, 4)


# ---------- scenarios and datasets ----------


def test_same_seed_same_clean_telemetry_whatever_the_anomaly() -> None:
    normal = make_scenario(21, "normal")
    for label, level in (("fault", None), ("fdi", 1), ("fdi", 3)):
        s = make_scenario(21, label, level=level)
        changed = np.flatnonzero(np.any(s.values != normal.values, axis=1))
        assert changed.size and changed.min() >= s.onset
        np.testing.assert_array_equal(s.truth.p, normal.truth.p)


def test_generation_is_deterministic() -> None:
    a, b = generate_dataset(range(30)), generate_dataset(range(30))
    np.testing.assert_array_equal(a.values, b.values)
    assert a.subtype == b.subtype


def test_dataset_is_balanced_and_labelled() -> None:
    d = generate_dataset(range(120))
    assert np.bincount(d.label).tolist() == [40, 40, 40]
    assert sorted(set(d.level[d.label == 2].tolist())) == [0, 1, 2, 3]
    assert np.all(d.level[d.label != 2] == -1)
    assert np.all((d.onset[d.label != 0] >= P.onset_min) & (d.onset[d.label != 0] <= P.onset_max))
    assert d.values.shape == (120, P.n_steps, len(CHANNELS))


def test_family_groups_units_and_axes() -> None:
    assert family("gnss_1 bias") == "gnss bias"
    assert family("baro stuck") == "baro stuck"
    assert family("gnss common_mode") == "gnss common_mode"
    assert family("L2 offset h") == "L2 offset"
    assert family("normal") == "normal"
