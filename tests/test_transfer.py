"""HAI transfer: the loader, the integrity check and the data-driven consistency checks, on a
tiny synthetic file in HAI's format (the real data are never needed by the tests)."""

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from fdicps.transfer import hai


def synthetic(rng: np.random.Generator, n: int, attack: tuple[int, int] | None = None):
    """Five tags driven by two latent processes (so they obey linear relations), one constant
    tag, and an optional attack that breaks one tag's relation to the others."""
    latent = np.cumsum(rng.normal(size=(n, 2)) * 0.1, axis=0)
    mix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
    tags = latent @ mix.T + rng.normal(size=(n, 5)) * 0.05
    label = np.zeros(n, dtype=bool)
    if attack:
        s, e = attack
        tags[s:e, 2] += 3.0
        label[s:e] = True
    tags = np.column_stack([tags, np.full(n, 7.0)])
    return tags, label


def write_hai(path: Path, tags: np.ndarray, label: np.ndarray) -> None:
    names = ["P1_A", "P1_B", "P2_C", "P2_D", "P3_E", "P4_CONST"]
    header = ["time", *names, "attack", "attack_P1", "attack_P2", "attack_P3"]
    lines = [",".join(header)]
    for k in range(len(tags)):
        lab = int(label[k])
        cells = [f"2021-01-01 00:{k // 60:02d}:{k % 60:02d}", *(f"{v:.6f}" for v in tags[k])]
        lines.append(",".join([*cells, str(lab), "0", str(lab), "0"]))
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_git_blob_sha1_matches_git() -> None:
    # `printf 'hello\n' | git hash-object --stdin` = ce013625030ba8dba906f756967f9e9ca394464a
    assert hai.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert hashlib.sha1(b"blob 0\0", usedforsecurity=False).hexdigest() == hai.git_blob_sha1(b"")


def test_verify_rejects_a_wrong_file(tmp_path: Path) -> None:
    path = tmp_path / "test1.csv.gz"
    path.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="does not match"):
        hai.verify(path)


def test_read_table_splits_tags_and_labels(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tags, label = synthetic(rng, 300, attack=(100, 150))
    path = tmp_path / "x.csv.gz"
    write_hai(path, tags, label)
    t = hai.read_table(path)
    assert t.tags == ["P1_A", "P1_B", "P2_C", "P2_D", "P3_E", "P4_CONST"]
    assert t.values.shape == (300, 6)
    assert t.attack.sum() == 50 and set(t.process_labels) == {"attack_P1", "attack_P2", "attack_P3"}
    np.testing.assert_array_equal(t.process_labels["attack_P2"], label)


def test_data_driven_checks_learn_the_relations_and_catch_a_broken_one() -> None:
    rng = np.random.default_rng(1)
    train, _ = synthetic(rng, 4000)
    model = hai.DataDrivenConsistency().fit(train)
    assert not model.keep[-1]  # the constant tag is dropped
    assert model.n_components == 2  # two latent processes explain > 99 % of the variance
    test, label = synthetic(rng, 2000, attack=(1000, 1100))
    alarms = model.alarms(test)
    for name, a in alarms.items():
        assert hai.event_recall(a, label) == 1.0, name
        assert a[:900].mean() < 0.01, name  # quiet before the attack
    # the broken relation is on tag 2 (the third kept tag): it dominates the residual
    contrib = model.residual_contributions(test)[1000:1100].mean(axis=0)
    assert contrib.shape == (5,) and int(np.argmax(contrib)) == 2


def test_moving_average_and_events() -> None:
    np.testing.assert_allclose(hai.moving_average(np.array([1.0, 3.0, 5.0, 7.0]), 2), [1, 2, 4, 6])
    lab = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert hai.events(lab) == [(1, 3), (5, 6)]
    alarm = np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
    assert hai.event_recall(alarm, lab) == 0.5
    assert hai.event_recall(alarm, lab, grace=1) == 0.5
    assert np.isnan(hai.event_recall(alarm, np.zeros(7, dtype=bool)))
    # one alarm episode starts outside an attack (at 4), over 4 normal seconds
    assert hai.false_alarm_episodes_per_hour(alarm, lab) == pytest.approx(3600 / 4)


def test_fetch_urls_are_pinned_to_one_dataset_commit() -> None:
    from fdicps.transfer import fetch

    url = fetch.URL.format(commit=hai.DATASET_COMMIT, version=hai.VERSION, name="test1.csv.gz")
    assert url == (
        "https://raw.githubusercontent.com/icsdataset/hai/"
        "2a814cebc9a66b06c9e5cd545e2d72e65d383737/hai-21.03/test1.csv.gz"
    )
    assert fetch.TARGET.parts[-3:] == ("data", "external", "hai")
    assert sum(size for size, _ in hai.FILES.values()) == 53_566_298
