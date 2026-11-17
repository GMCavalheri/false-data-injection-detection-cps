"""Rerun every validation script, in phase order, and report the time each took.

    uv run python validation/run_all.py

Everything except the 1-D CNN is exact or seeded and deterministic across machines, so a rerun
reproduces those tables in ``validation/results/`` byte for byte; CI checks that with
``git diff``. The CNN's table (``sequence_model.md``) is compared within a tolerance by
``check_tolerant.py``, because PyTorch's CPU kernels can differ by instruction set. The HAI
transfer needs the dataset (``uv run python -m fdicps.transfer.fetch``) and is skipped, by the
script itself, when the files are absent, as they are in CI. Figures go to ``docs/figures/``
(their pixels can differ slightly between platforms' font setups).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("generation.py", "Phase 1: dual anomaly data generation"),
    ("baseline.py", "Phase 2: baseline anomaly detection"),
    ("classification.py", "Phase 3: physical-vs-FDI classification"),
    ("robustness.py", "Phase 4: adversarial robustness"),
    ("consistency.py", "Phase 5: cross-sensor consistency models"),
    ("hai_transfer.py", "Extension: HAI transfer (skipped without the data)"),
]


def main() -> int:
    for name, title in SCRIPTS:
        print(f"== {title}", flush=True)
        t0 = time.perf_counter()
        done = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE.parent)
        if done.returncode:
            print(f"   {name} failed (exit {done.returncode})", flush=True)
            return done.returncode
        print(f"   done in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
