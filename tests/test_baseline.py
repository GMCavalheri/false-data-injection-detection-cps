"""Phase 2: the rule-based checks match their closed-form false-alarm rates, and the
unsupervised detector is deterministic and calibrated on normal data."""

import numpy as np
import pytest
from scipy import stats

from fdicps.baseline_anomaly_detection.detector import BaselineDetector
from fdicps.baseline_anomaly_detection.features import feature_names, generic_features
from fdicps.baseline_anomaly_detection.rate_check import (
    envelope_violations,
    rate_scores,
    rule_score,
    single_test_false_alarm,
)
from fdicps.dual_anomaly_data_generation.mission import Truth, integrate
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import generate_dataset
from fdicps.dual_anomaly_data_generation.sensors import CHANNEL_NAMES, measure

P = load_sensor_parameters()


def straight_and_level(n: int = 300) -> Truth:
    """Constant velocity, zero acceleration: every true second difference is exactly zero."""
    a = np.zeros((n, 2))
    p, v = integrate(np.array([0.0, 8000.0]), np.array([220.0, 0.0]), a, P.dt)
    return Truth(t=np.arange(n) * P.dt, p=p, v=v, a=a)


def test_single_test_false_alarm_closed_form() -> None:
    assert single_test_false_alarm(3.0) == pytest.approx(0.0026997960632601866, rel=1e-12)


def test_rate_score_of_a_noise_free_straight_line_is_zero() -> None:
    # a straight line (any slope, any offset, on every channel) has zero second difference
    line = np.outer(np.arange(50.0), np.linspace(1, 13, 13)) + 7.0
    assert rate_scores(line, P).max() < 1e-12


def test_rate_check_false_alarm_matches_the_normal_tail() -> None:
    # Under straight-and-level flight d2 is pure noise with variance 6 sigma^2. Every third
    # d2 shares no sample with the next, so those tests are independent Bernoulli(p0) trials.
    kappa = 2.5
    p0 = single_test_false_alarm(kappa)
    truth = straight_and_level()
    rng = np.random.default_rng(42)
    hits = []
    for _ in range(60):
        scores = rate_scores(measure(truth, P, rng).values, P)[::3]
        hits.append(scores > kappa)
    h = np.concatenate(hits)
    n, k = h.size, int(h.sum())
    lo, hi = stats.binomtest(k, n).proportion_ci(0.999)
    assert lo < p0 < hi, (k / n, p0)


def test_envelope_flags_impossible_values() -> None:
    values = np.zeros((1, 10, 13))
    values[..., CHANNEL_NAMES.index("gnss_1.vel.x")] = 200.0
    values[..., CHANNEL_NAMES.index("gnss_2.vel.x")] = 200.0
    values[..., CHANNEL_NAMES.index("gnss_1.pos.h")] = 5000.0
    values[..., CHANNEL_NAMES.index("gnss_2.pos.h")] = 5000.0
    values[..., CHANNEL_NAMES.index("baro.pos.h")] = 5000.0
    assert not envelope_violations(values).any()
    values[0, 4, CHANNEL_NAMES.index("gnss_2.vel.h")] = 80.0
    assert envelope_violations(values).sum() == 1
    assert rule_score(values, P)[0] == np.inf


def test_generic_features_are_unit_scale_on_healthy_level_flight() -> None:
    # every feature is normalised by its healthy noise level: d2 std and pair-difference std
    # should be 1 up to sampling error (n = 298 per scenario, averaged over 40)
    truth = straight_and_level()
    rng = np.random.default_rng(5)
    values = np.stack([measure(truth, P, rng).values for _ in range(40)])
    X = generic_features(values, P)
    names = feature_names()
    assert X.shape == (40, len(names)) == (40, 47)
    d2_std = X[:, [i for i, n in enumerate(names) if n.endswith("d2 std")]].mean(axis=0)
    np.testing.assert_allclose(d2_std, 1.0, atol=0.03)
    # pair differences: the INS pair also carries the two units' constant bias difference,
    # which the mean removes before the std
    pair_std = X[:, [i for i, n in enumerate(names) if n.endswith("diff std")]].mean(axis=0)
    np.testing.assert_allclose(pair_std, 1.0, atol=0.03)


@pytest.fixture(scope="module")
def small() -> tuple:
    train = generate_dataset(range(300))
    test = generate_dataset(range(50_000, 50_150))
    return train, test


def test_detector_is_deterministic_and_calibrated(small) -> None:
    train, test = small
    normal = train.values[train.label == 0]
    a = BaselineDetector(n_estimators=100).fit(normal, P)
    b = BaselineDetector(n_estimators=100).fit(normal, P)
    np.testing.assert_array_equal(a.score(test.values, P), b.score(test.values, P))
    # the threshold is the 99th percentile of the training normals' own scores
    assert a.alarm(normal, P).mean() == pytest.approx(0.01, abs=0.011)
    # it does detect: anomalous scenarios score higher than normal ones on average
    s = a.score(test.values, P)
    assert s[test.label > 0].mean() > s[test.label == 0].mean()


def test_detector_needs_fit() -> None:
    with pytest.raises(AssertionError):
        BaselineDetector().score(np.zeros((1, 300, 13)), P)
