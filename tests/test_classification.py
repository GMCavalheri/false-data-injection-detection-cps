"""Phase 3: the physics residuals are exact where the physics is exact, standard normal where
only noise remains, and the classifiers are deterministic and learn something real."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from fdicps.dual_anomaly_data_generation.faults import PhysicalFault
from fdicps.dual_anomaly_data_generation.fdi import FalseDataInjection, apply_fdi
from fdicps.dual_anomaly_data_generation.mission import mission_profile
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import generate_dataset, make_scenario
from fdicps.dual_anomaly_data_generation.sensors import CHANNELS, clean_signal
from fdicps.experiment import TEST_SEEDS, TRAIN_SEEDS
from fdicps.physical_vs_fdi_classification.classifier import PhysicsClassifier
from fdicps.physical_vs_fdi_classification.features import feature_names, physics_features
from fdicps.physical_vs_fdi_classification.residuals import (
    REFERENCE,
    kinematic_family,
    residual_sequences,
    window_variance,
    window_z,
)
from fdicps.physical_vs_fdi_classification.sequence_model import SequenceClassifier, prepare

P = load_sensor_parameters()


def noise_free(seed: int, bias: float = 0.004) -> np.ndarray:
    truth = mission_profile(np.random.default_rng(seed), P)
    return clean_signal(truth, np.full((2, 2), bias))


def test_kinematic_residuals_vanish_on_noise_free_telemetry() -> None:
    # The zero-order-hold relation is exact (Lesson 1), and a constant INS bias is removed by
    # the reference centring, so every kinematic residual is zero to rounding.
    values = noise_free(3)[None]
    for name, terms in kinematic_family(values, P).items():
        z = window_z(terms, 10)
        assert np.abs(z).max() < 1e-6, name


def test_l3_breaks_no_relation_and_l2_breaks_receiver_kinematics() -> None:
    clean = noise_free(4)[None]
    drag = FalseDataInjection(3, "drag", 1, 150, 0.08, 40.0, target=6)
    offset = FalseDataInjection(2, "offset", 1, 150, 80.0, 15.0, target=6)
    for fdi, broken in ((drag, False), (offset, True)):
        values = apply_fdi(clean[0], fdi, P)[None]
        peak = max(np.abs(window_z(t, 10)).max() for t in kinematic_family(values, P).values())
        assert (peak > 10) == broken, (fdi.subtype, peak)


def test_window_variance_closed_form() -> None:
    terms = kinematic_family(noise_free(0)[None], P)["kin gnss_1 x"]
    f, g, R = terms.var_fixed, terms.var_per_sample, REFERENCE
    assert window_variance(terms, 1) == pytest.approx(f + g + (f + R * g) / R**2)
    assert f == pytest.approx(2 * P.gnss_pos_sigma_x**2)


@pytest.mark.parametrize("n", [1, 10, 60])
def test_healthy_window_z_is_standard_normal(n: int) -> None:
    d = generate_dataset(range(300))
    values = d.values[d.label == 0]
    for name, terms in kinematic_family(values, P).items():
        z = window_z(terms, n)[..., REFERENCE:]
        # windows overlap, so the effective sample size is ~ samples / n; 0.06 is > 4 standard
        # errors of the std estimate for the smallest effective size here (100 x 180 / 60)
        assert abs(z.std() - 1.0) < 0.06, (name, z.std())
        assert abs(z.mean()) < 0.1, (name, z.mean())


def test_features_are_zero_where_physics_holds_exactly() -> None:
    X = physics_features(noise_free(5)[None], P)
    names = feature_names()
    assert X.shape == (1, len(names)) == (1, 52)
    exact = [i for i, n in enumerate(names) if n.startswith(("kin", "vel-ins", "disagree", "both"))]
    assert np.abs(X[0, exact]).max() < 1e-6


def test_noise_structure_features_see_frozen_and_noise_free_outputs() -> None:
    names = feature_names()
    stuck = make_scenario(9, "fault", anomaly=PhysicalFault("ins_1", "stuck", 120, 0.0))
    normal = make_scenario(9, "normal")
    l0 = make_scenario(9, "fdi", level=0)
    X = physics_features(np.stack([normal.values, stuck.values, l0.values]), P)
    frozen = names.index("frozen ins_1")
    assert X[0, frozen] < 0.02 < 0.5 < X[1, frozen]  # 180 of 300 samples frozen
    unit = l0.anomaly.target  # a position channel; its unit's smoothest window is noise-free
    smooth = names.index(f"smooth {CHANNELS[unit].unit}")
    # healthy: the smallest of ~1000 windowed spreads (30 correlated samples each) sits well
    # below 1 but far above the noise-free value
    assert X[2, smooth] < 0.01 < 0.25 < X[0, smooth]


def test_residual_sequences_shape_and_names() -> None:
    seq, names = residual_sequences(np.stack([make_scenario(1, "normal").values]), P)
    assert seq.shape == (1, P.n_steps - 2 - 10 + 1, len(names)) == (1, 289, 27)


def test_train_and_test_seeds_are_disjoint() -> None:
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS)


@pytest.fixture(scope="module")
def data():
    return generate_dataset(range(600)), generate_dataset(range(70_000, 70_300))


def test_physics_classifier_is_deterministic_and_beats_chance(data) -> None:
    train, test = data
    a = PhysicsClassifier(max_iter=100).fit(train.values, train.label, P)
    b = PhysicsClassifier(max_iter=100).fit(train.values, train.label, P)
    ya, yb = a.predict(test.values, P), b.predict(test.values, P)
    np.testing.assert_array_equal(ya, yb)
    dummy = DummyClassifier(strategy="stratified", random_state=0).fit(
        np.zeros((len(train.label), 1)), train.label
    )
    chance = f1_score(test.label, dummy.predict(np.zeros((len(test.label), 1))), average="macro")
    assert f1_score(test.label, ya, average="macro") > chance + 0.3


def test_sequence_model_is_deterministic(data) -> None:
    train, test = data
    x, xt = prepare(train.values[:120], P), prepare(test.values[:30], P)
    a = SequenceClassifier(epochs=2).fit_prepared(x, train.label[:120])
    b = SequenceClassifier(epochs=2).fit_prepared(x, train.label[:120])
    assert a.loss_history == b.loss_history
    np.testing.assert_array_equal(a.predict_proba_prepared(xt), b.predict_proba_prepared(xt))
    assert a.predict(test.values[:3], P).shape == (3,)


def test_models_need_fit() -> None:
    with pytest.raises(AssertionError):
        PhysicsClassifier().predict_features(np.zeros((1, 52)))
    with pytest.raises(AssertionError):
        SequenceClassifier().predict_prepared(np.zeros((1, 27, 289), dtype=np.float32))
