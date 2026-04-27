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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    d = root / args.results_dir
    candidates = {
        "custom": d / "metrics_custom.json",
        "vgg16": d / "metrics_vgg16.json",
    }
    best_name = "custom"
    best_acc = -1.0
    for name, path in candidates.items():
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        acc = float(m.get("test_acc", 0.0))
        if acc > best_acc:
            best_acc = acc
            best_name = name
    out = d / "best_model.txt"
    d.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(best_name)
    print(f"best={best_name}  test_acc={best_acc:.6f}  -> {out}")


if __name__ == "__main__":
    main()
