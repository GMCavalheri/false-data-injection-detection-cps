"""Compare a regenerated results table with its committed version, number by number.

    uv run python validation/check_tolerant.py validation/results/sequence_model.md \\
        --rows "1-D CNN, mean" --tolerance 0.1

Used for the CNN's results. PyTorch's CPU kernels differ by instruction set, and training
amplifies those differences, so a single seed's numbers change between machines. Only the rows
whose first cell starts with ``--rows`` are compared (the mean over seeds); every non-numeric
cell of those rows must match exactly and every number must be within the tolerance, which is
set from the spread across seeds reported in the same table.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOLERANCE = 0.1


def cells(text: str, rows: str = "") -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        row = [c.strip() for c in line.strip().strip("|").split("|")]
        if row[0].startswith(rows):
            out += row
    return out


def number(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def compare(
    committed: str, current: str, tolerance: float = TOLERANCE, rows: str = ""
) -> list[str]:
    a, b = cells(committed, rows), cells(current, rows)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--rows", default="", help="compare only rows starting with this text")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{args.path.as_posix()}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    problems = compare(committed, args.path.read_text(), args.tolerance, args.rows)
    if not cells(committed, args.rows):
        problems.append(f"no committed rows start with {args.rows!r}")
    for p in problems:
        print(p)
    verdict = "OK" if not problems else f"{len(problems)} differences"
    print(f"{args.path}: {verdict} (rows {args.rows!r}, tolerance {args.tolerance})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
