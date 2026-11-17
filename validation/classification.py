"""Phase 3: physical fault or false data? The physics-informed classifiers on held-out data.

Gradient-boosting results are exact (deterministic everywhere) and go to
``classification.md``. The CNN's go to ``sequence_model.md``: PyTorch's CPU kernels can
differ by instruction set between machines, so the reproduce job compares that file within
a tolerance instead of byte for byte.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from fdicps import plotting
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters
from fdicps.dual_anomaly_data_generation.scenarios import LABELS, family, generate_dataset
from fdicps.experiment import TEST_SEEDS, TRAIN_SEEDS
from fdicps.physical_vs_fdi_classification.classifier import PhysicsClassifier
from fdicps.physical_vs_fdi_classification.features import feature_names, physics_features
from fdicps.physical_vs_fdi_classification.sequence_model import SequenceClassifier, prepare

from common import FIGURES, table, write_result

P = load_sensor_parameters()
CNN_CONFIGS = ((15, "15 epochs"), (25, "25 epochs"))


def r3(x: float) -> float:
    return round(float(x), 3)


def per_class(y: np.ndarray, yhat: np.ndarray) -> str:
    p, r, f, n = precision_recall_fscore_support(y, yhat, labels=[0, 1, 2], zero_division=0)
    rows = [[LABELS[i], int(n[i]), r3(p[i]), r3(r[i]), r3(f[i])] for i in range(3)]
    rows.append(["macro average", int(n.sum()), r3(p.mean()), r3(r.mean()), r3(f.mean())])
    return table(["class", "scenarios", "precision", "recall", "F1"], rows)


def by_family(subtypes: list[str], yhat: np.ndarray) -> str:
    fam: dict[str, list[int]] = defaultdict(list)
    for i, st in enumerate(subtypes):
        fam[family(st)].append(i)
    order = ["normal"] + sorted(k for k in fam if k != "normal" and not k.startswith("L"))
    order += sorted(k for k in fam if k.startswith("L"))
    rows = []
    for k in order:
        share = np.bincount(yhat[fam[k]], minlength=3) / len(fam[k])
        rows.append([k, len(fam[k]), *(r3(s) for s in share)])
    return table(["family", "scenarios", "→ normal", "→ fault", "→ FDI"], rows)


def confusion(y: np.ndarray, yhat: np.ndarray) -> str:
    cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
    return table(["true \\ predicted", *LABELS], [[LABELS[i], *cm[i].tolist()] for i in range(3)])


def main() -> None:
    train, test = generate_dataset(TRAIN_SEEDS), generate_dataset(TEST_SEEDS)
    Xtr, Xte = physics_features(train.values, P), physics_features(test.values, P)
    gbm = PhysicsClassifier().fit_features(Xtr, train.label)
    yhat = gbm.predict_features(Xte)

    anomalous = test.label > 0
    flagged = anomalous & (yhat > 0)
    attribution = table(
        ["question (anomalous test scenarios)", "value"],
        [
            ["flagged as anomalous (fault or FDI)", r3(flagged.sum() / anomalous.sum())],
            ["cause correct among flagged", r3((yhat[flagged] == test.label[flagged]).mean())],
            ["faults called FDI", int(((test.label == 1) & (yhat == 2)).sum())],
            ["FDI called faults", int(((test.label == 2) & (yhat == 1)).sum())],
        ],
    )

    imp = permutation_importance(
        gbm.model, Xte, test.label, scoring="f1_macro", n_repeats=5, random_state=0
    )
    names = feature_names()
    top = np.argsort(-imp.importances_mean)[:12]
    importance = table(
        ["feature", "macro-F1 drop when shuffled (mean of 5)"],
        [[names[i], r3(imp.importances_mean[i])] for i in top],
    )
    write_result(
        "classification",
        "Phase 3: physical-vs-FDI classification (gradient boosting on physics features)",
        [
            ("Per class, test set", per_class(test.label, yhat)),
            ("Confusion matrix", confusion(test.label, yhat)),
            ("Predicted class by anomaly family", by_family(test.subtype, yhat)),
            ("Attribution", attribution),
            ("Most important features (permutation importance, test set)", importance),
        ],
    )

    xtr, xte = prepare(train.values, P), prepare(test.values, P)
    rows, cnn_preds = [], {}
    for epochs, label in CNN_CONFIGS:
        cnn = SequenceClassifier(epochs=epochs).fit_prepared(xtr, train.label)
        yc = cnn.predict_prepared(xte)
        cnn_preds[label] = yc
        rec = precision_recall_fscore_support(test.label, yc, labels=[0, 1, 2], zero_division=0)[1]
        rows.append(
            [
                f"1-D CNN, {label}",
                r3(f1_score(test.label, yc, average="macro")),
                *(r3(r) for r in rec),
            ]
        )
    rec = precision_recall_fscore_support(test.label, yhat, labels=[0, 1, 2], zero_division=0)[1]
    rows.insert(
        0,
        [
            "gradient boosting, physics features",
            r3(f1_score(test.label, yhat, average="macro")),
            *(r3(r) for r in rec),
        ],
    )
    write_result(
        "sequence_model",
        "Phase 3: the 1-D CNN on residual sequences, against gradient boosting",
        [
            (
                "Test set (compared within a tolerance by the reproduce job)",
                table(["model", "macro F1", "recall normal", "recall fault", "recall FDI"], rows),
            ),
            (
                "Predicted class by anomaly family, CNN 25 epochs",
                by_family(test.subtype, cnn_preds["25 epochs"]),
            ),
        ],
        script="classification",
    )

    plotting.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    cm = confusion_matrix(test.label, yhat, labels=[0, 1, 2])
    axes[0].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(
                j, i, cm[i, j], ha="center", va="center", color="k" if cm[i, j] < 300 else "w"
            )
    axes[0].set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS)
    axes[0].set(xlabel="predicted", ylabel="true", title="gradient boosting, test set")
    axes[0].grid(False)
    order = top[::-1]
    axes[1].barh([names[i] for i in order], imp.importances_mean[order], color=plotting.FAULT_COLOR)
    axes[1].set(xlabel="macro-F1 drop when shuffled", title="what the classifier relies on")
    axes[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "classification.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
