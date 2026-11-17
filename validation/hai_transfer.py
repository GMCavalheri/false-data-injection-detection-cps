"""HAI transfer: the data-driven consistency checks on a real ICS testbed's attacks.

Needs the HAI 21.03 files: ``uv run python -m fdicps.transfer.fetch`` first. Not part of the
reproduce job (CI has no copy of the data, which is never committed); run_all.py skips it when
the files are absent.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from fdicps import plotting
from fdicps.transfer import hai
from fdicps.transfer.fetch import TARGET

from common import FIGURES, table, write_result


def r3(x: float) -> float:
    return round(float(x), 3)


def diagnose(
    model: hai.DataDrivenConsistency, train: hai.HaiTable, t: hai.HaiTable, alarm: np.ndarray
) -> str:
    """Which tags drive the longest false-alarm episode, and are they outside training?"""
    episodes = [(s, e) for s, e in hai.events(alarm) if not t.attack[s]]
    s, e = max(episodes, key=lambda se: se[1] - se[0])
    normal = np.arange(s, e)[~t.attack[s:e]]  # the episode's seconds outside any attack
    contrib = model.residual_contributions(t.values)[normal].mean(axis=0)
    kept = [tag for tag, k in zip(train.tags, model.keep, strict=True) if k]
    rows = []
    for i in np.argsort(-contrib)[:4]:
        j = train.tags.index(kept[i])
        lo, hi = train.values[:, j].min(), train.values[:, j].max()
        rows.append([kept[i], r3(contrib[i]), r3(t.values[s:e, j].mean()), f"{lo:.3f} to {hi:.3f}"])
    first = hai.events(t.attack)[0][0]
    head = (
        f"Seconds {s} to {e} ({(e - s) / 60:.0f} min); the first attack begins at second "
        f"{first}, so the episode starts in normal operation and runs into it.\n\n"
    )
    return head + table(
        [
            "tag",
            "mean squared residual (normal part)",
            "mean value (normal part)",
            "training range",
        ],
        rows,
    )


def main() -> None:
    paths = {name: TARGET / name for name in hai.FILES}
    if not all(p.exists() for p in paths.values()):
        print("HAI files absent; run `uv run python -m fdicps.transfer.fetch`", file=sys.stderr)
        return
    for p in paths.values():
        hai.verify(p)
    train = hai.read_table(paths["train1.csv.gz"])
    model = hai.DataDrivenConsistency().fit(train.values[~train.attack])
    setup = table(
        ["quantity", "value"],
        [
            ["dataset", f"HAI {hai.VERSION[4:]} (icsdataset/hai at {hai.DATASET_COMMIT[:7]})"],
            ["training file (normal operation only)", f"train1: {len(train.attack)} s"],
            ["tags / tags varying in training", f"{len(train.tags)} / {int(model.keep.sum())}"],
            ["principal components kept (99 % of variance)", model.n_components],
            ["moving average on both statistics (s)", hai.SMOOTH],
            ["threshold", "99.9th percentile of the smoothed statistic on training data"],
        ],
    )
    rows, proc_rows, plots, diagnosis = [], [], {}, ""
    for name in ("test1.csv.gz", "test2.csv.gz"):
        t = hai.read_table(paths[name])
        stats = model.statistics(t.values)
        alarms = model.alarms(t.values)
        short = name.split(".")[0]
        plots[short] = (t, stats)
        if short == "test1":
            diagnosis = diagnose(model, train, t, alarms["static (PCA residual)"])
        for check, s in stats.items():
            a = alarms[check]
            rows.append(
                [
                    short,
                    check,
                    len(t.attack),
                    len(hai.events(t.attack)),
                    r3(roc_auc_score(t.attack, s)),
                    r3(hai.event_recall(a, t.attack)),
                    r3(hai.false_alarm_episodes_per_hour(a, t.attack)),
                    r3(a[~t.attack].mean()),
                ]
            )
            for label, lab in sorted(t.process_labels.items()):
                n_ev = len(hai.events(lab))
                if n_ev:
                    proc_rows.append([short, check, label, n_ev, r3(hai.event_recall(a, lab))])
    results = table(
        [
            "file",
            "check",
            "seconds",
            "attack events",
            "ROC AUC (per second)",
            "event recall",
            "false-alarm episodes per normal hour",
            "share of normal seconds in alarm",
        ],
        rows,
    )
    by_process = table(["file", "check", "label", "events", "event recall"], proc_rows)
    write_result(
        "hai_transfer",
        "HAI transfer: data-driven consistency checks on a real ICS testbed",
        [
            ("Setup", setup),
            ("Detection", results),
            ("By attacked process (events of each per-process label)", by_process),
            ("The longest false-alarm episode of test1 (static check)", diagnosis),
            (
                "What does not transfer",
                "HAI labels attacks but not physical faults, so fault-versus-attack "
                "attribution cannot be evaluated on it; only the detection layer is tested.\n",
            ),
        ],
    )

    plotting.use_style()
    t, stats = plots["test1"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    hours = np.arange(len(t.attack)) / 3600
    for ax, (check, s) in zip(axes, stats.items(), strict=True):
        ax.plot(hours, s, lw=0.6, color=plotting.NORMAL_COLOR)
        ax.axhline(model.thresholds[check], color="k", ls="--", lw=0.8)
        for start, end in hai.events(t.attack):
            ax.axvspan(start / 3600, end / 3600, color=plotting.FDI_COLOR, alpha=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{check}, HAI 21.03 test1 (attacks shaded)", fontsize=9, loc="left")
    axes[-1].set_xlabel("time (h)")
    fig.tight_layout()
    fig.savefig(FIGURES / "hai_transfer.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
