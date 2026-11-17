"""Phase 4: how detection and attribution degrade as the attacker respects more physics."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from fdicps import plotting
from fdicps.adversarial_robustness_analysis.attacks import (
    consistency_sweep,
    exact_mimic,
    near_mimic,
    receiver_bias_fault,
    spoofer_variant,
)
from fdicps.adversarial_robustness_analysis.monitors import (
    SCENARIO_FALSE_ALARM,
    per_test_level,
    snapshot_alarm,
    snapshot_scenario_detection,
)
from fdicps.cross_sensor_consistency_model.navigation import snapshot_models
from fdicps.dual_anomaly_data_generation.fdi import FalseDataInjection, injection
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import (
    Dataset,
    Scenario,
    generate_dataset,
    stack,
)
from fdicps.experiment import ROBUSTNESS_SEEDS, TEST_SEEDS, TRAIN_SEEDS
from fdicps.physical_vs_fdi_classification.classifier import PhysicsClassifier
from fdicps.physical_vs_fdi_classification.features import physics_features

from common import FIGURES, table, write_result

P = load_sensor_parameters()
CPS02_ASSUMED_MISS = 0.1  # cpsrbd v1.0.0, integrity_monitor_miss


def r3(x: float) -> float:
    return round(float(x), 3)


def closed_form_snapshot(scenarios: list[Scenario]) -> float:
    """Mean closed-form probability that the snapshot monitor alarms (additive attacks)."""
    models = snapshot_models(P)
    alpha = per_test_level(SCENARIO_FALSE_ALARM, 2 * P.n_steps)
    out = []
    for s in scenarios:
        assert isinstance(s.anomaly, FalseDataInjection)
        a = injection(s.anomaly, P, P.n_steps)
        lam = np.concatenate([models[ax][1].noncentrality(a[:, models[ax][0]]) for ax in (0, 1)])
        dofs = np.repeat([models[0][1].dof, models[1][1].dof], P.n_steps)
        out.append(snapshot_scenario_detection(lam, dofs, alpha))
    return float(np.mean(out))


def shares(clf: PhysicsClassifier, X: np.ndarray) -> np.ndarray:
    """Fraction predicted normal, fault, FDI."""
    return np.bincount(clf.predict_features(X), minlength=3) / len(X)


def main() -> None:
    defender = PhysicsClassifier().fit(*_xy(generate_dataset(TRAIN_SEEDS, fdi_levels=(0, 1))))
    full = PhysicsClassifier().fit(*_xy(generate_dataset(TRAIN_SEEDS)))

    sweep_rows, curves = [], {}
    for pattern, label in (("offset", "L1 → L2 (static)"), ("drag", "drag-off → L3 (dynamic)")):
        sweep = consistency_sweep(ROBUSTNESS_SEEDS, pattern)
        curve = []
        for s, d in sweep.items():
            X = physics_features(d.values, P)
            a, f = shares(defender, X), shares(full, X)
            snap = snapshot_alarm(d.values, P).mean()
            closed = closed_form_snapshot(d.scenarios)
            curve.append((s, a[2], 1 - a[0], f[2], snap, closed))
            sweep_rows.append(
                [label, s, r3(a[2]), r3(1 - a[0]), r3(f[2]), r3(1 - f[0]), r3(snap), r3(closed)]
            )
        curves[label] = np.array(curve)
    sweep_table = table(
        [
            "attack path",
            "stealth s",
            "→ FDI (trained L0–L1)",
            "flagged (trained L0–L1)",
            "→ FDI (trained L0–L3)",
            "flagged (trained L0–L3)",
            "snapshot alarm, measured",
            "snapshot alarm, closed form",
        ],
        sweep_rows,
    )
    breakdown = table(
        ["attack path", "smallest s with P(→ FDI) < 0.5, classifier trained on L0–L1"],
        [
            [label, next((r3(c[0]) for c in curve if c[1] < 0.5), "none")]
            for label, curve in curves.items()
        ],
    )

    seeds = list(ROBUSTNESS_SEEDS)
    # real faults from a disjoint seed range: the comparison is between distributions
    real_bias = stack([receiver_bias_fault(seed + 1000) for seed in seeds])
    groups: list[tuple[str, Callable[[int], Scenario] | Dataset]] = [
        ("real receiver bias fault", real_bias),
        ("near mimic: fault-sized step on one channel", near_mimic),
        ("exact mimic: a receiver bias reproduced by the attacker", exact_mimic),
    ]
    mimic_rows = []
    for name, src in groups:
        d = src if isinstance(src, Dataset) else stack([src(seed) for seed in seeds])
        X = physics_features(d.values, P)
        sh = shares(full, X)
        mimic_rows.append([name, r3(sh[0]), r3(sh[1]), r3(sh[2])])
    mimic = table(["scenarios (300 each)", "→ normal", "→ fault", "→ FDI"], mimic_rows)

    test = generate_dataset(TEST_SEEDS)
    Xte = physics_features(test.values, P)
    pred = full.predict_features(Xte)
    snap = snapshot_alarm(test.values, P)
    miss_rows = []
    for level in (0, 1, 2, 3):
        idx = (test.label == 2) & (test.level == level)
        miss_rows.append(
            [
                f"L{level} (test set)",
                r3(1 - snap[idx].mean()),
                r3((pred[idx] == 0).mean()),
                r3((pred[idx] != 2).mean()),
            ]
        )
    for reach, name in (
        ("receivers", "L3, receivers only (gnss_spoof)"),
        ("bus", "L3, whole bus (bus injection)"),
    ):
        d = stack([spoofer_variant(seed, reach) for seed in seeds])
        p = full.predict_features(physics_features(d.values, P))
        sn = snapshot_alarm(d.values, P)
        miss_rows.append([name, r3(1 - sn.mean()), r3((p == 0).mean()), r3((p != 2).mean())])
    miss = table(
        [
            "attack",
            "P(miss), snapshot monitor",
            "P(miss), classifier (called normal)",
            "P(not attributed to FDI), classifier",
        ],
        miss_rows,
    )
    write_result(
        "robustness",
        "Phase 4: adversarial robustness",
        [
            (
                "Stealth sweep (300 paired attacks per point; the target channel's injection "
                "is the same at every s)",
                sweep_table,
            ),
            ("Breakdown point", breakdown),
            ("Fault mimicry (classifier trained on L0–L3)", mimic),
            (
                f"Integrity-monitor miss probability (CPS-02 assumed {CPS02_ASSUMED_MISS})",
                miss,
            ),
        ],
    )

    plotting.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (label, c) in zip(axes, curves.items(), strict=True):
        ax.plot(c[:, 0], c[:, 1], "o-", color=plotting.FDI_COLOR, label="→ FDI, trained L0–L1")
        ax.plot(
            c[:, 0],
            c[:, 3],
            "s--",
            color=plotting.FDI_COLOR,
            alpha=0.6,
            label="→ FDI, trained L0–L3",
        )
        ax.plot(c[:, 0], c[:, 2], "o-", color=plotting.FAULT_COLOR, label="flagged, trained L0–L1")
        ax.plot(
            c[:, 0], c[:, 4], "^", color=plotting.NORMAL_COLOR, label="snapshot monitor, measured"
        )
        ax.plot(
            c[:, 0],
            c[:, 5],
            "-",
            color=plotting.NORMAL_COLOR,
            lw=1,
            label="snapshot monitor, closed form",
        )
        ax.set(title=label, xlabel="stealth s (0 = one channel, 1 = fully consistent)")
    axes[0].set_ylabel("share of attacks")
    axes[1].legend(fontsize=7, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness_sweep.png")
    plt.close(fig)


def _xy(d: Dataset) -> tuple[np.ndarray, np.ndarray, object]:
    return d.values, d.label, P


if __name__ == "__main__":
    main()
