"""Compare a regenerated results table with its committed version, number by number.

    uv run python validation/check_tolerant.py validation/results/sequence_model.md

Used for the CNN's results, which can move slightly between machines (PyTorch CPU kernels
differ by instruction set). Every non-numeric cell must match exactly and every number must be
within ``TOLERANCE``: a few test scenarios changing class out of 1500 moves a rate by about
0.002 each, so 0.03 allows small kernel differences but not a changed model or data set.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOLERANCE = 0.03


def cells(text: str) -> list[str]:
    return [
        c.strip()
        for line in text.splitlines()
        if line.startswith("|")
        for c in line.strip("|").split("|")
    ]


def number(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def compare(committed: str, current: str, tolerance: float = TOLERANCE) -> list[str]:
    a, b = cells(committed), cells(current)
    if len(a) != len(b):
        return [f"table shape changed: {len(a)} cells committed, {len(b)} now"]
    problems = []
    for x, y in zip(a, b, strict=True):
        nx, ny = number(x), number(y)
        if nx is None or ny is None:
            if x != y:
                problems.append(f"text cell changed: {x!r} -> {y!r}")
        elif abs(nx - ny) > tolerance:
            problems.append(f"number moved more than {tolerance}: {x} -> {y}")
    return problems


def main() -> int:
    path = Path(sys.argv[1])
    committed = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"], capture_output=True, text=True, check=True
    ).stdout
    problems = compare(committed, path.read_text())
    for p in problems:
        print(p)
    verdict = "OK" if not problems else f"{len(problems)} differences"
    print(f"{path}: {verdict} (tolerance {TOLERANCE})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
