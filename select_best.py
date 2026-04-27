#!/usr/bin/env python3
"""Compare metrics_*.json and write results/best_model.txt (custom|vgg16)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_logging import configure_logging, get_logger  # noqa: E402

LOG = get_logger("cv_proj.select_best")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    configure_logging(verbosity=args.verbose)
    root = Path(__file__).resolve().parent
    d = root / args.results_dir
    candidates = {
        "custom": d / "metrics_custom.json",
        "vgg16": d / "metrics_vgg16.json",
    }
    best_name = "custom"
    best_acc = -1.0
    LOG.info("comparing test metrics: %s", {k: str(v) for k, v in candidates.items()})
    for name, path in candidates.items():
        if not path.is_file():
            LOG.warning("skip %s (missing file)", path)
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        acc = float(m.get("test_acc", 0.0))
        LOG.info("  %s: test_acc=%.6f", name, acc)
        if acc > best_acc:
            best_acc = acc
            best_name = name
    out = d / "best_model.txt"
    d.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(best_name)
    LOG.info("winner=%s  test_acc=%.6f  -> %s", best_name, best_acc, out)


if __name__ == "__main__":
    main()
