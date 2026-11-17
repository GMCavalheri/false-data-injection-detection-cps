"""Phase 1: the parameter tables, the datasets every later phase uses, and what each anomaly
does to the telemetry."""

from __future__ import annotations

import csv
import io
from collections import Counter
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np

from fdicps import plotting
from fdicps.dual_anomaly_data_generation.faults import PhysicalFault, affected_channels
from fdicps.dual_anomaly_data_generation.fdi import FalseDataInjection, cps02_leaf
from fdicps.dual_anomaly_data_generation.fdi import affected_channels as fdi_channels
from fdicps.dual_anomaly_data_generation.scenarios import family, generate_dataset, make_scenario
from fdicps.experiment import TEST_SEEDS, TRAIN_SEEDS

from common import FIGURES, table, write_result

EXAMPLE_SEED = 1_000_007


def _csv(name: str) -> list[dict[str, str]]:
    text = resources.files("fdicps.data").joinpath(name).read_text()
    return list(csv.DictReader(io.StringIO(text)))


def composition(train_subtypes: list[str], test_subtypes: list[str]) -> str:
    tr, te = Counter(map(family, train_subtypes)), Counter(map(family, test_subtypes))
    order = ["normal"] + sorted(k for k in tr if not k.startswith(("L", "normal")))
    order += sorted(k for k in tr if k.startswith("L"))
    rows = [[k, tr[k], te[k]] for k in order]
    rows.append(["total", sum(tr.values()), sum(te.values())])
    return table(["family", "train scenarios", "test scenarios"], rows)


def effects(scenarios: list) -> str:
    """Per family: channels touched and the largest injected change (ground truth)."""
    by: dict[str, list[tuple[int, float, float]]] = {}
    for s in scenarios:
        if s.anomaly is None:
            continue
        clean = make_scenario(s.seed, "normal").values
        d = np.abs(s.values - clean)
        if isinstance(s.anomaly, PhysicalFault):
            idx = affected_channels(s.anomaly)
        else:
            idx = fdi_channels(s.anomaly)
        pos = [i for i in idx if i in (0, 2, 6, 8, 12)]
        by.setdefault(family(s.subtype), []).append(
            (len(idx), float(d[:, pos].max()) if pos else 0.0, float(d[:, idx].max()))
        )
    rows = []
    for k in sorted(by):
        v = np.array(by[k])
        rows.append([k, int(np.median(v[:, 0])), float(np.median(v[:, 1])), float(v[:, 1].max())])
    return table(
        ["family", "channels touched (median)", "max position change, median (m)", "max (m)"],
        rows,
    )


def leaves(scenarios: list) -> str:
    c = Counter(
        cps02_leaf(s.anomaly) for s in scenarios if isinstance(s.anomaly, FalseDataInjection)
    )
    return table(["CPS-02 leaf", "test FDI scenarios"], sorted(c.items()))


def figures() -> None:
    plotting.use_style()
    cases = [
        ("normal", {}, "normal"),
        ("fault", {"anomaly": PhysicalFault("gnss_1", "bias", 150, 40.0, (1, 1))}, "gnss_1 bias"),
        ("fault", {"anomaly": PhysicalFault("gnss_1", "drift", 150, 0.6, (1, 1))}, "gnss_1 drift"),
        ("fault", {"anomaly": PhysicalFault("baro", "stuck", 150, 0.0)}, "baro stuck"),
        ("fault", {"anomaly": PhysicalFault("gnss_2", "noise", 150, 5.0)}, "gnss_2 noise ×5"),
        (
            "fault",
            {"anomaly": PhysicalFault("gnss", "common_mode", 150, 0.2, (1, 1), cap=20.0)},
            "shared receiver error (common mode)",
        ),
        ("fdi", {"anomaly": FalseDataInjection(0, "replace", 1, 150, 80.0, 0.0, 6)}, "L0 naive"),
        (
            "fdi",
            {"anomaly": FalseDataInjection(1, "offset", 1, 150, 80.0, 15.0, 6, 0.0)},
            "L1 plausible, one channel",
        ),
        (
            "fdi",
            {"anomaly": FalseDataInjection(2, "offset", 1, 150, 80.0, 15.0, 6, 1.0)},
            "L2 statically consistent (a = Hc)",
        ),
        (
            "fdi",
            {"anomaly": FalseDataInjection(3, "drag", 1, 150, 0.08, 40.0, 6, 1.0)},
            "L3 dynamically consistent (drag-off)",
        ),
    ]
    scen = [make_scenario(EXAMPLE_SEED, lab, **kw) for lab, kw, _ in cases]
    names = ["gnss_1.pos.h", "gnss_2.pos.h", "baro.pos.h"]
    fig = plotting.signature_grid(scen, names, [c[2] for c in cases], "error (m)")
    fig.savefig(FIGURES / "anomaly_signatures.png")
    plt.close(fig)

    s = scen[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    axes[0].plot(s.truth.t, s.truth.p[:, 1], color=plotting.NORMAL_COLOR)
    axes[0].set(title="altitude (m)", xlabel="time (s)")
    axes[1].plot(s.truth.t, s.truth.v[:, 0], color=plotting.NORMAL_COLOR)
    axes[1].set(title="along-track speed (m/s)", xlabel="time (s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "mission_profile.png")
    plt.close(fig)


def main() -> None:
    train = generate_dataset(TRAIN_SEEDS)
    test = generate_dataset(TEST_SEEDS)
    sensors = table(
        ["parameter", "value", "unit", "basis"],
        [[r["name"], r["value"], r["unit"], r["basis"]] for r in _csv("sensor_parameters.csv")],
    )
    anomalies = table(
        ["parameter", "low", "high", "unit", "basis"],
        [
            [r["name"], r["low"], r["high"], r["unit"], r["basis"]]
            for r in _csv("anomaly_parameters.csv")
        ],
    )
    write_result(
        "generation",
        "Phase 1: dual anomaly data generation",
        [
            ("Sensor parameters", sensors),
            ("Anomaly magnitude ranges (drawn uniformly)", anomalies),
            (
                "Datasets",
                f"Train seeds {TRAIN_SEEDS.start}–{TRAIN_SEEDS.stop - 1}, test seeds "
                f"{TEST_SEEDS.start}–{TEST_SEEDS.stop - 1} (disjoint). One third normal, one third "
                "physical fault, one third FDI cycling through L0–L3.\n\n"
                + composition(train.subtype, test.subtype),
            ),
            ("What each anomaly changes (test set, ground truth)", effects(test.scenarios)),
            ("CPS-02 attack-tree leaves of the test FDI scenarios", leaves(test.scenarios)),
        ],
    )
    figures()


if __name__ == "__main__":
    main()
