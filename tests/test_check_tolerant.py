"""The tolerance check used by the reproduce job for the CNN's results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

from check_tolerant import compare  # noqa: E402

TABLE = "| model | F1 |\n|---|---|\n| cnn | {x} |\n"


def test_small_moves_pass_and_large_moves_or_text_changes_fail() -> None:
    assert compare(TABLE.format(x="0.857"), TABLE.format(x="0.85"), 0.03) == []
    assert compare(TABLE.format(x="0.857"), TABLE.format(x="0.8"), 0.03) != []
    assert compare(TABLE.format(x="0.857"), TABLE.replace("cnn", "gbm").format(x="0.857")) != []
    assert compare(TABLE.format(x="1"), TABLE.format(x="1") + "| extra | 2 |\n") != []


def test_row_filter_compares_only_the_selected_rows() -> None:
    old = "| model | F1 |\n|---|---|\n| cnn, seed 0 | 0.80 |\n| cnn, mean | 0.82 |\n"
    new = "| model | F1 |\n|---|---|\n| cnn, seed 0 | 0.70 |\n| cnn, mean | 0.80 |\n"
    assert compare(old, new, 0.05, rows="cnn, mean") == []
    assert compare(old, new, 0.05) != []
