"""The tolerance check used by the reproduce job for the CNN's results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

from check_tolerant import compare  # noqa: E402

TABLE = "| model | F1 |\n|---|---|\n| cnn | {x} |\n"


def test_small_moves_pass_and_large_moves_or_text_changes_fail() -> None:
    assert compare(TABLE.format(x="0.857"), TABLE.format(x="0.85")) == []
    assert compare(TABLE.format(x="0.857"), TABLE.format(x="0.8")) != []
    assert compare(TABLE.format(x="0.857"), TABLE.replace("cnn", "gbm").format(x="0.857")) != []
    assert compare(TABLE.format(x="1"), TABLE.format(x="1") + "| extra | 2 |\n") != []
