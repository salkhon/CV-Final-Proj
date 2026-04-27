#!/usr/bin/env python3
"""Download SVHN into data/svhn (relative to project root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchvision import datasets  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data" / "svhn"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test", "extra"):
        datasets.SVHN(str(data_dir), split=split, download=True)
    print(f"SVHN ready under {data_dir}")


if __name__ == "__main__":
    main()
