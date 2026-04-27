#!/usr/bin/env python3
"""Download SVHN into data/svhn (relative to project root)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_logging import configure_logging, get_logger  # noqa: E402
from torchvision import datasets  # noqa: E402

LOG = get_logger("cv_proj.download")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    configure_logging(verbosity=args.verbose)

    data_dir = ROOT / "data" / "svhn"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "test", "extra")
    for i, split in enumerate(
        tqdm(
            splits,
            desc="SVHN download",
            unit="split",
            disable=args.no_progress,
        ),
        1,
    ):
        LOG.info("Step %d/3: split=%r -> %s", i, split, data_dir)
        datasets.SVHN(str(data_dir), split=split, download=True)
    LOG.info("done: SVHN under %s (train, test, extra)", data_dir)


if __name__ == "__main__":
    main()
