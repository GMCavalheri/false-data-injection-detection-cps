"""Download the HAI 21.03 files this project uses, pinned to one dataset commit and verified.

Run ``uv run python -m fdicps.transfer.fetch`` (about 54 MB). Files go to ``data/external/hai``
(ignored by git). Nothing is downloaded at import time or by the tests.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from fdicps.transfer.hai import DATASET_COMMIT, FILES, VERSION, verify

TARGET = Path(__file__).resolve().parents[3] / "data" / "external" / "hai"
URL = "https://raw.githubusercontent.com/icsdataset/hai/{commit}/{version}/{name}"


def fetch(target: Path = TARGET) -> list[Path]:  # pragma: no cover - network
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for name in FILES:
        path = target / name
        if not path.exists():
            url = URL.format(commit=DATASET_COMMIT, version=VERSION, name=name)
            print(f"downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URL
        verify(path)
        out.append(path)
    return out


if __name__ == "__main__":  # pragma: no cover
    for p in fetch():
        print(p)
