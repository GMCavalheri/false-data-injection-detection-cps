"""Phase 5: the model-based consistency layer (WLS snapshot check, Kalman filter NIS tests,
CUSUM), the value of a dissimilar sensor, the classifier with consistency features, and the
IEEE 14-bus anchor."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fdicps import plotting
from fdicps.adversarial_robustness_analysis.attacks import consistency_sweep, spoofer_variant
from fdicps.adversarial_robustness_analysis.monitors import snapshot_alarm
from fdicps.cross_sensor_consistency_model import ieee14
from fdicps.cross_sensor_consistency_model.detection import (
    consistency_features,
    cusum_score,
    filter_axes,
    nis_alarm,
    window_alarm,
)
from fdicps.cross_sensor_consistency_model.kalman import axis_model, gains, steady_state_prior
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import Dataset, generate_dataset, stack
from fdicps.experiment import ROBUSTNESS_SEEDS, TEST_SEEDS, TRAIN_SEEDS
from fdicps.physical_vs_fdi_classification.classifier import PhysicsClassifier
from fdicps.physical_vs_fdi_classification.features import physics_features
from fdicps.physical_vs_fdi_classification.residuals import REFERENCE

from common import FIGURES, table, write_result

P = load_sensor_parameters()
FAR = 0.01
REACHES = ("receivers", "bus", "bus, air data protected")


def r3(x: float) -> float:
    return round(float(x), 3)


class Monitors:
    def __init__(self, train: Dataset) -> None:
        normal = filter_axes(train.values[train.label == 0], P)
        self.cusum_threshold = float(np.quantile(cusum_score(normal), 1 - FAR))

    def alarms(self, values: np.ndarray) -> dict[str, np.ndarray]:
        runs = filter_axes(values, P)
        return {
            "snapshot WLS": snapshot_alarm(values, P, FAR),
            "KF NIS, per sample": nis_alarm(runs, FAR),
            "KF NIS, 20-s blocks": window_alarm(runs, FAR),
            "KF CUSUM": cusum_score(runs) > self.cusum_threshold,
        }


def filter_validation(test: Dataset) -> str:
    rows = []
    runs = filter_axes(test.values[test.label == 0], P)
    for axis, name in ((0, "along-track"), (1, "vertical")):
        model = axis_model(P, axis)
        dare = steady_state_prior(model)
        gap = (
            np.abs(gains(model, 20_000, P.ins_bias_sigma).P_prior[-1] - dare).max()
            / np.abs(dare).max()
        )
        sd = np.sqrt(np.diag(dare))
        rows.append(
            [
                name,
                model.m,
                r3(runs.runs[axis].nis[:, REFERENCE:].mean()),
                r3(sd[0]),
                r3(sd[1]),
                f"{sd[2]:.2e}",
                f"{gap:.1e}",
            ]
        )
    return table(
        [
            "axis",
            "m",
            "mean NIS, test normals",
            "steady prior σ p (m)",
            "σ v (m/s)",
            "σ b (m/s²)",
            "gap to DARE after 20 000 steps",
        ],
        rows,
    )


def main() -> None:
    train, test = generate_dataset(TRAIN_SEEDS), generate_dataset(TEST_SEEDS)
    mon = Monitors(train)
    names = ["snapshot WLS", "KF NIS, per sample", "KF NIS, 20-s blocks", "KF CUSUM"]

    rows, variant_alarms = [], {}
    a = mon.alarms(test.values)
    groups = [("normal (false alarms)", test.label == 0), ("physical faults", test.label == 1)]
    groups += [(f"L{lv}", (test.label == 2) & (test.level == lv)) for lv in range(4)]
    for g, idx in groups:
        rows.append([g, int(idx.sum()), *(r3(a[n][idx].mean()) for n in names)])
    seeds = list(ROBUSTNESS_SEEDS)
    variants = {reach: stack([spoofer_variant(s, reach) for s in seeds]) for reach in REACHES}
    for reach, d in variants.items():
        va = mon.alarms(d.values)
        variant_alarms[reach] = (va, np.array([s.anomaly.axis for s in d.scenarios]))
        rows.append(
            [f"L3 drag-off, reach: {reach}", len(seeds), *(r3(va[n].mean()) for n in names)]
        )
    detection = table(["scenarios", "count", *names], rows)

    by_axis = []
    for reach in ("bus", "bus, air data protected"):
        va, axis = variant_alarms[reach]
        for ax, label in ((0, "along-track"), (1, "vertical")):
            sel = axis == ax
            by_axis.append(
                [reach, label, int(sel.sum()), r3(va["KF NIS, 20-s blocks"][sel].mean())]
            )
    dissimilar = table(
        ["attacker reach", "attacked axis", "attacks", "detected (KF NIS, 20-s blocks)"], by_axis
    )

    # classifier with and without the consistency features
    Ptr, Pte = physics_features(train.values, P), physics_features(test.values, P)
    Ctr, Cte = consistency_features(train.values, P), consistency_features(test.values, P)
    crude = generate_dataset(TRAIN_SEEDS, fdi_levels=(0, 1))
    Pcr, Ccr = physics_features(crude.values, P), consistency_features(crude.values, P)
    sets = {
        "physics": (Ptr, Pte, Pcr),
        "physics + consistency": (
            np.hstack([Ptr, Ctr]),
            np.hstack([Pte, Cte]),
            np.hstack([Pcr, Ccr]),
        ),
    }
    cls_rows, sweep_rows, spoof_rows, curves = [], [], [], {}
    sweeps = {
        label: consistency_sweep(ROBUSTNESS_SEEDS, pattern)
        for pattern, label in (("offset", "static"), ("drag", "dynamic"))
    }
    sweep_feats = {
        (label, s): (physics_features(d.values, P), consistency_features(d.values, P))
        for label, sw in sweeps.items()
        for s, d in sw.items()
    }
    variant_feats = {
        r: (physics_features(d.values, P), consistency_features(d.values, P))
        for r, d in variants.items()
    }
    for name, (Xtr, Xte, Xcr) in sets.items():
        full = PhysicsClassifier().fit_features(Xtr, train.label)
        defender = PhysicsClassifier().fit_features(Xcr, crude.label)
        y = full.predict_features(Xte)
        cls_rows.append(
            [
                name,
                r3((y == test.label).mean()),
                r3((y[test.label == 1] == 1).mean()),
                *(r3((y[(test.label == 2) & (test.level == lv)] == 2).mean()) for lv in range(4)),
            ]
        )
        for label in sweeps:
            curve = []
            for s in sweeps[label]:
                pf, cf = sweep_feats[(label, s)]
                X = pf if name == "physics" else np.hstack([pf, cf])
                curve.append(
                    (
                        (defender.predict_features(X) == 2).mean(),
                        (full.predict_features(X) == 2).mean(),
                    )
                )
            curve_a = np.array(curve)
            curves[(name, label)] = curve_a
            brk = next(
                (s for s, c in zip(sweeps[label], curve_a[:, 0], strict=True) if c < 0.5), None
            )
            sweep_rows.append(
                [
                    name,
                    label,
                    "none" if brk is None else brk,
                    r3(curve_a[-2, 0]),
                    r3(curve_a[-1, 1]),
                ]
            )
        for reach in REACHES:
            pf, cf = variant_feats[reach]
            X = pf if name == "physics" else np.hstack([pf, cf])
            p = full.predict_features(X)
            spoof_rows.append([name, reach, r3((p == 0).mean()), r3((p == 2).mean())])
    classifier = table(
        ["features", "accuracy", "fault recall", "L0 → FDI", "L1 → FDI", "L2 → FDI", "L3 → FDI"],
        cls_rows,
    )
    sweep = table(
        [
            "features",
            "path",
            "breakdown s (trained L0–L1)",
            "→ FDI at s = 0.9 (trained L0–L1)",
            "→ FDI at s = 1 (trained L0–L3)",
        ],
        sweep_rows,
    )
    spoof = table(["features", "attacker reach", "→ normal (missed)", "→ FDI"], spoof_rows)

    model = ieee14.model()
    rng = np.random.default_rng(14)
    z = rng.normal(size=model.m) * model.sigma
    c = rng.normal(scale=0.2, size=model.n)
    stealth_gap = float(np.abs(model.residual(z + model.H @ c) - model.residual(z)).max())
    ieee_rows = [
        ["meters m (20 branch flows + 14 injections)", model.m],
        ["states n (bus angles, bus 1 reference)", model.n],
        ["rank of H", int(np.linalg.matrix_rank(model.H))],
        ["degrees of freedom of the bad-data test", model.dof],
        ["largest residual change under a = Hc, c ~ N(0, 0.2²) rad", f"{stealth_gap:.1e}"],
    ]
    det_rows = []
    for label, meters, size in (
        ("one meter, 0.05 p.u.", [0], 0.05),
        ("one meter, 0.10 p.u.", [0], 0.10),
        ("three meters, 0.03 p.u. each", [3, 12, 25], 0.03),
        ("a = Hc (stealthy)", None, None),
    ):
        if meters is None:
            a = model.H @ c
        else:
            a = np.zeros(model.m)
            a[meters] = size
        closed = float(model.detection_probability(a, FAR))
        zz = rng.normal(size=(20_000, model.m)) * model.sigma + a
        mc = float((model.statistic(zz) > model.threshold(FAR)).mean())
        det_rows.append([label, r3(float(model.noncentrality(a))), r3(closed), r3(mc)])
    need = ieee14.meters_to_move_one_state()
    meters = table(["bus", *(str(b) for b in range(2, 15))], [["meters to alter", *need]])
    write_result(
        "consistency",
        "Phase 5: cross-sensor consistency models",
        [
            ("Kalman filter validation", filter_validation(test)),
            (
                f"Detection rate by monitor (scenario false-alarm rate set to {FAR}; CUSUM "
                "threshold calibrated on training normals)",
                detection,
            ),
            ("Value of a dissimilar sensor out of the attacker's reach", dissimilar),
            ("Classifier with consistency features (test set)", classifier),
            ("Robustness sweep with consistency features", sweep),
            ("L3 variants through the classifier (trained L0–L3)", spoof),
            ("IEEE 14-bus DC model", table(["quantity", "value"], ieee_rows)),
            (
                "IEEE 14-bus bad-data test (false-alarm rate 0.01): closed form vs 20 000 draws",
                table(
                    [
                        "attack",
                        "noncentrality λ",
                        "P(detect), closed form",
                        "P(detect), Monte Carlo",
                    ],
                    det_rows,
                ),
            ),
            (
                "IEEE 14-bus: meters an attacker must alter to shift one bus angle stealthily",
                meters,
            ),
        ],
    )

    plotting.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    grid = list(sweeps["static"])
    for ax, label in zip(axes, ("static", "dynamic"), strict=True):
        for name, style in (("physics", "o-"), ("physics + consistency", "s--")):
            c = curves[(name, label)]
            ax.plot(grid, c[:, 0], style, color=plotting.FDI_COLOR, label=f"{name}, trained L0–L1")
            ax.plot(
                grid,
                c[:, 1],
                style,
                color=plotting.FAULT_COLOR,
                alpha=0.7,
                label=f"{name}, trained L0–L3",
            )
        ax.set(title=f"{label} path", xlabel="stealth s")
    axes[0].set_ylabel("share attributed to FDI")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "consistency_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
