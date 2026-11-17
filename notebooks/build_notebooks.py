"""Generate and execute the demo notebooks, one per lesson, from the cell sources below.

    uv run python notebooks/build_notebooks.py           # write and execute all notebooks
    uv run python notebooks/build_notebooks.py --check   # execute in memory only (CI)

Keeping the sources here, instead of hand-editing .ipynb JSON, makes the notebooks reviewable
in diffs. Every computation in them is seeded, and they use small data sets so they run in
seconds; the headline numbers live in ``validation/results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """\
import matplotlib.pyplot as plt
import numpy as np

from fdicps import plotting
from fdicps.dual_anomaly_data_generation.parameters import load_sensor_parameters

plotting.use_style()
P = load_sensor_parameters()"""

Cell = tuple[str, str]

NOTEBOOKS: dict[str, list[Cell]] = {
    "01_two_kinds_of_wrong.ipynb": [
        (
            "md",
            """\
# 1 · Two kinds of wrong

The navigation chain's telemetry under a physical fault and under false data injection at four
levels. See [Lesson 1](../docs/lessons/01-two-kinds-of-wrong.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from fdicps.dual_anomaly_data_generation.scenarios import make_scenario

seed = 42
cases = [("normal", None), ("fault", None), ("fdi", 0), ("fdi", 1), ("fdi", 2), ("fdi", 3)]
scenarios = [make_scenario(seed, label, level=level) for label, level in cases]
for s in scenarios:
    print(f"{s.label:6s} {s.subtype:22s} onset {s.onset:4d}")""",
        ),
        (
            "code",
            """\
names = ["gnss_1.pos.h", "gnss_2.pos.h", "baro.pos.h"]
fig = plotting.signature_grid(scenarios, names, [s.subtype for s in scenarios], "error (m)")""",
        ),
        ("md", "An L2 injection lies in the range of $H$: the WLS residual does not move."),
        (
            "code",
            """\
from fdicps.cross_sensor_consistency_model.navigation import snapshot_models
from fdicps.dual_anomaly_data_generation.fdi import injection

l2 = scenarios[4].anomaly
idx, wls = snapshot_models(P)[l2.axis]
a = injection(l2, P, P.n_steps)[:, idx]
print("largest injected value (m):", np.abs(a).max().round(1))
print("largest residual change:", np.abs(a @ wls.S.T).max())""",
        ),
    ],
    "02_the_detector_that_cannot_say_why.ipynb": [
        (
            "md",
            """\
# 2 · The detector that can't say why

An Isolation Forest trained on normal flight: it detects, but its score cannot attribute cause.
See [Lesson 2](../docs/lessons/02-the-detector-that-cannot-say-why.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from sklearn.metrics import roc_auc_score

from fdicps.baseline_anomaly_detection.detector import BaselineDetector
from fdicps.dual_anomaly_data_generation.scenarios import generate_dataset

train, test = generate_dataset(range(600)), generate_dataset(range(10_000, 10_300))
det = BaselineDetector(n_estimators=150).fit(train.values[train.label == 0], P)
score = det.score(test.values, P)
anomalous = test.label > 0
print("AUC normal vs anomalous:", round(roc_auc_score(anomalous, score), 3))
print("AUC of the score for 'attack, not fault':", round(roc_auc_score(test.label[anomalous] == 2, score[anomalous]), 3))""",
        ),
        (
            "code",
            """\
groups = ["normal", "fault"] + [f"L{k}" for k in range(4)]
data = [score[test.label == 0], score[test.label == 1]] + [
    score[(test.label == 2) & (test.level == k)] for k in range(4)
]
fig, ax = plt.subplots(figsize=(7, 3))
ax.boxplot(data, tick_labels=groups, showfliers=False)
ax.axhline(det.threshold, color="k", ls="--", lw=1)
ax.set_ylabel("anomaly score");""",
        ),
    ],
    "03_reading_physics_in_an_anomaly.ipynb": [
        (
            "md",
            """\
# 3 · Reading physics in an anomaly

Physics residuals, the gradient-boosting classifier, and a tiny run of the sequence model.
See [Lesson 3](../docs/lessons/03-reading-physics-in-an-anomaly.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from fdicps.dual_anomaly_data_generation.scenarios import make_scenario
from fdicps.physical_vs_fdi_classification.residuals import REFERENCE, kinematic_family, window_z

fig, ax = plt.subplots(figsize=(8, 3))
for label, level in (("normal", None), ("fdi", 2)):
    s = make_scenario(1_000_007, label, level=level)
    for name, terms in kinematic_family(s.values[None], P).items():
        if name.startswith("kin gnss_1"):
            z = window_z(terms, 10)[0]
            ax.plot(z, lw=0.8, label=f"{s.subtype}: {name}")
ax.axvspan(0, REFERENCE, color="0.9")
ax.set(xlabel="time (s)", ylabel="z (noise standard deviations)")
ax.legend(fontsize=7, frameon=False);""",
        ),
        (
            "code",
            """\
from sklearn.metrics import confusion_matrix

from fdicps.dual_anomaly_data_generation.scenarios import generate_dataset
from fdicps.physical_vs_fdi_classification.classifier import PhysicsClassifier

train, test = generate_dataset(range(900)), generate_dataset(range(20_000, 20_300))
clf = PhysicsClassifier(max_iter=150).fit(train.values, train.label, P)
pred = clf.predict(test.values, P)
print(confusion_matrix(test.label, pred))
for k in range(4):
    sel = (test.label == 2) & (test.level == k)
    print(f"L{k}: called FDI {np.mean(pred[sel] == 2):.2f}")""",
        ),
        (
            "code",
            """\
from fdicps.physical_vs_fdi_classification.sequence_model import SequenceClassifier

cnn = SequenceClassifier(epochs=3).fit(train.values[:300], train.label[:300], P)
print("training loss by epoch:", [round(x, 3) for x in cnn.loss_history])
print("accuracy on the 300 test scenarios:", np.mean(cnn.predict(test.values, P) == test.label).round(3))""",
        ),
    ],
    "04_when_the_attacker_knows_the_physics.ipynb": [
        (
            "md",
            """\
# 4 · When the attacker knows the physics

The noncentrality law $\\lambda(s) = (1-s)^2 \\lambda(0)$ and the snapshot monitor along the stealth dial,
closed form against simulation. See [Lesson 4](../docs/lessons/04-when-the-attacker-knows-the-physics.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from fdicps.adversarial_robustness_analysis.attacks import consistency_variant
from fdicps.adversarial_robustness_analysis.monitors import snapshot_alarm

grid = np.linspace(0, 1, 11)
seeds = range(3_000_000, 3_000_080)
measured = [snapshot_alarm(np.stack([consistency_variant(s, "offset", x).values for s in seeds]), P).mean() for x in grid]
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(grid, measured, "o-", color=plotting.FDI_COLOR)
ax.set(xlabel="stealth s", ylabel="snapshot monitor alarm rate");""",
        ),
        (
            "code",
            """\
from fdicps.cross_sensor_consistency_model.navigation import snapshot_models

idx, wls = snapshot_models(P)[1]
a_one = np.array([100.0, 0, 0, 0, 0])
a_all = wls.H @ np.array([100.0, 0.0])
for s in (0.0, 0.5, 0.9, 1.0):
    lam = float(wls.noncentrality(s * a_all + (1 - s) * a_one))
    print(f"s = {s:.1f}: lambda = {lam:7.2f} = {(1 - s) ** 2:.2f} x lambda(0)")""",
        ),
    ],
    "05_state_estimation_as_a_lie_detector.ipynb": [
        (
            "md",
            """\
# 5 · State estimation as a lie detector

The Kalman filter's NIS against its chi-square law, three attacks through the filter, and the IEEE
14-bus model. See [Lesson 5](../docs/lessons/05-state-estimation-as-a-lie-detector.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from scipy import stats

from fdicps.cross_sensor_consistency_model.kalman import axis_model, run_filter
from fdicps.dual_anomaly_data_generation.scenarios import generate_dataset

normal = generate_dataset(range(150))
values = normal.values[normal.label == 0]
model = axis_model(P, 1)
nis = run_filter(values, P, model).nis[:, 60:].ravel()
x = np.linspace(0, 25, 200)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(nis, bins=60, density=True, color="0.7")
ax.plot(x, stats.chi2.pdf(x, model.m), color=plotting.FAULT_COLOR, label=f"chi-square, m = {model.m}")
ax.legend(frameon=False)
print("mean NIS:", nis.mean().round(3))""",
        ),
        (
            "code",
            """\
from fdicps.adversarial_robustness_analysis.attacks import spoofer_variant
from fdicps.cross_sensor_consistency_model.detection import filter_axes, window_alarm

for reach in ("receivers", "bus", "bus, air data protected"):
    d = np.stack([spoofer_variant(s, reach).values for s in range(4_000_000, 4_000_060)])
    print(f"{reach:26s} detected {window_alarm(filter_axes(d, P), 0.01).mean():.2f}")""",
        ),
        (
            "code",
            """\
from fdicps.cross_sensor_consistency_model import ieee14

model14 = ieee14.model()
rng = np.random.default_rng(0)
z = rng.normal(size=model14.m) * model14.sigma
c = rng.normal(scale=0.2, size=model14.n)
print("residual change under a = Hc:", np.abs(model14.residual(z + model14.H @ c) - model14.residual(z)).max())
print("meters to alter, buses 2-14:", ieee14.meters_to_move_one_state())""",
        ),
    ],
    "06_testing_on_someone_elses_plant.ipynb": [
        (
            "md",
            """\
# 6 · Testing on someone else's plant

The data-driven consistency checks on a toy plant (the HAI results need the dataset; see
`validation/results/hai_transfer.md`). See [Lesson 6](../docs/lessons/06-testing-on-someone-elses-plant.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from fdicps.transfer.hai import DataDrivenConsistency

rng = np.random.default_rng(0)
mix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.5, 2.0]])


def plant(n):
    latent = np.cumsum(rng.normal(size=(n, 2)) * 0.1, axis=0)
    return latent @ mix.T + rng.normal(size=(n, 5)) * 0.05


model = DataDrivenConsistency().fit(plant(4000))
test = plant(600)
test[300:400, 2] += 3.0
stats_ = model.statistics(test)
fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
for ax, (name, s) in zip(axes, stats_.items()):
    ax.semilogy(s, color="0.3")
    ax.axhline(model.thresholds[name], color="k", ls="--", lw=0.8)
    ax.axvspan(300, 400, color=plotting.FDI_COLOR, alpha=0.3)
    ax.set_title(name, fontsize=9, loc="left")""",
        ),
    ],
    "07_what_is_and_isnt_new.ipynb": [
        (
            "md",
            """\
# 7 · What is (and isn't) new

The meter count of Lesson 5 is only an upper bound on the security index: the three-bus example of
the exercises, by brute force. See [Lesson 7](../docs/lessons/07-what-is-and-isnt-new.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
import itertools

from fdicps.cross_sensor_consistency_model.wls import WLS

# line network 1-2-3, bus 1 reference; rows: flow 1-2, flow 2-3, injections 1, 2, 3
H = np.array([[-1.0, 0.0], [1.0, -1.0], [-1.0, 0.0], [2.0, -1.0], [-1.0, 1.0]])
print("rank:", np.linalg.matrix_rank(H), " dof:", WLS(H, np.ones(5)).dof)
for j in range(2):
    best = min(
        (np.count_nonzero(np.abs(H @ c) > 1e-12), tuple(int(x) for x in c))
        for c in (np.array(v, dtype=float) for v in itertools.product([-1, 0, 1], repeat=2))
        if c[j] == 1
    )
    print(f"theta_{j + 2}: moved alone {np.count_nonzero(H[:, j])} meters, cheapest {best[0]} with c = {best[1]}")""",
        ),
    ],
}


def build(name: str, cells: list[Cell]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"{Path(name).stem}-{i}"
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        record_timing=False,  # no timestamps: rebuilds are byte-identical
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="execute without writing")
    ap.add_argument("names", nargs="*", help="notebooks to build (default: all)")
    args = ap.parse_args()
    for name in args.names or list(NOTEBOOKS):
        print(f"== {name}", flush=True)
        nb = build(name, NOTEBOOKS[name])
        if not args.check:
            nbformat.write(nb, HERE / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
