"""Phase 2: the undifferentiated baseline detects many anomalies but cannot name a cause."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from fdicps import plotting
from fdicps.baseline_anomaly_detection.detector import BaselineDetector
from fdicps.baseline_anomaly_detection.rate_check import (
    KAPPA,
    rate_scores,
    rule_score,
    single_test_false_alarm,
)
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import family, generate_dataset
from fdicps.experiment import TEST_SEEDS, TRAIN_SEEDS

from common import FIGURES, table, write_result

P = load_sensor_parameters()
GROUPS = ["normal", "physical fault", "FDI L0", "FDI L1", "FDI L2", "FDI L3"]


def group_of(label: int, level: int) -> str:
    return ["normal", "physical fault", f"FDI L{level}"][label]


def r3(x: float) -> float:
    return round(float(x), 3)


def main() -> None:
    train, test = generate_dataset(TRAIN_SEEDS), generate_dataset(TEST_SEEDS)
    det = BaselineDetector().fit(train.values[train.label == 0], P)
    s_if = det.score(test.values, P)
    s_rule = rule_score(test.values, P)
    a_if, a_rule = s_if > det.threshold, s_rule > KAPPA
    normal, anomalous = test.label == 0, test.label > 0

    per_test = rate_scores(test.values[normal], P)
    calibration = table(
        ["quantity", "value"],
        [
            ["Isolation Forest threshold (99th percentile of training normals)", r3(det.threshold)],
            ["Isolation Forest false-alarm rate, test normals", r3(a_if[normal].mean())],
            ["rate check threshold κ (noise standard deviations)", KAPPA],
            ["one rate test, closed form 2(1 − Φ(κ))", single_test_false_alarm(KAPPA)],
            [
                "one rate test, measured on test normals (manoeuvres included)",
                float((per_test > KAPPA).mean()),
            ],
            ["rule-based false-alarm rate per scenario, test normals", r3(a_rule[normal].mean())],
        ],
    )

    fam: dict[str, list[int]] = defaultdict(list)
    for i, st in enumerate(test.subtype):
        fam[family(st)].append(i)
    order = ["normal"] + sorted(k for k in fam if k != "normal" and not k.startswith("L"))
    order += sorted(k for k in fam if k.startswith("L"))
    detection = table(
        ["family", "scenarios", "Isolation Forest", "rule-based", "either"],
        [
            [
                k,
                len(fam[k]),
                r3(a_if[fam[k]].mean()),
                r3(a_rule[fam[k]].mean()),
                r3((a_if | a_rule)[fam[k]].mean()),
            ]
            for k in order
        ],
    )
    finite_rule = np.where(np.isinf(s_rule), 1e6, s_rule)
    overall = table(
        [
            "detector",
            "ROC AUC normal vs anomalous",
            "detection rate, faults",
            "detection rate, FDI",
        ],
        [
            [
                name,
                r3(roc_auc_score(anomalous, score)),
                r3(alarm[test.label == 1].mean()),
                r3(alarm[test.label == 2].mean()),
            ]
            for name, score, alarm in (
                ("Isolation Forest", s_if, a_if),
                ("rule-based", finite_rule, a_rule),
            )
        ],
    )

    alarms = a_if & anomalous
    groups = np.array([group_of(lab, lev) for lab, lev in zip(test.label, test.level, strict=True)])
    routed = table(
        ["true cause of an Isolation Forest alarm", "alarms", "share of alarms"],
        [
            [
                g,
                int((alarms & (groups == g)).sum()),
                r3((alarms & (groups == g)).sum() / alarms.sum()),
            ]
            for g in GROUPS[1:]
        ],
    )
    cause_auc = roc_auc_score(test.label[anomalous] == 2, s_if[anomalous])
    cause_auc_alarmed = roc_auc_score(test.label[alarms] == 2, s_if[alarms])
    cause = table(
        ["question", "value"],
        [
            ["ROC AUC of the anomaly score for 'attack, not fault', all anomalies", r3(cause_auc)],
            ["the same, among alarms only", r3(cause_auc_alarmed)],
            [
                "share of FDI scenarios that raise no alarm at all",
                r3(1 - a_if[test.label == 2].mean()),
            ],
        ],
    )
    write_result(
        "baseline",
        "Phase 2: undifferentiated baseline anomaly detection",
        [
            ("Calibration", calibration),
            ("Overall", overall),
            ("Detection rate by anomaly family (test set)", detection),
            ("What an alarm does not tell you", routed + "\n" + cause),
        ],
    )

    plotting.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    data = [s_if[groups == g] for g in GROUPS]
    colors = [plotting.NORMAL_COLOR, plotting.FAULT_COLOR] + [plotting.FDI_COLOR] * 4
    bp = axes[0].boxplot(data, tick_labels=GROUPS, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors, strict=True):
        patch.set(facecolor=c, alpha=0.35, edgecolor=c)
    axes[0].axhline(det.threshold, color="k", ls="--", lw=1, label="alarm threshold (1 % FAR)")
    axes[0].set_ylabel("Isolation Forest anomaly score")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].tick_params(axis="x", labelsize=8)
    for name, score, c in (
        ("Isolation Forest", s_if, plotting.FAULT_COLOR),
        ("rule-based", finite_rule, plotting.NORMAL_COLOR),
    ):
        fpr, tpr, _ = roc_curve(anomalous, score)
        axes[1].plot(fpr, tpr, color=c, label=name)
    fpr, tpr, _ = roc_curve(test.label[anomalous] == 2, s_if[anomalous])
    axes[1].plot(fpr, tpr, color=plotting.FDI_COLOR, ls=":", label="score as 'attack?' test")
    axes[1].plot([0, 1], [0, 1], color="0.7", lw=0.8)
    axes[1].set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_scores.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
