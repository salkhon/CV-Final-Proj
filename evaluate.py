#!/usr/bin/env python3
"""Test-set accuracy; writes metrics_*.json and appends comparison_table.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.svhn_11 import NEGATIVE_CLASS, get_svhn_11_datasets  # noqa: E402
from src.infer_utils import forward_batch, load_model_from_checkpoint  # noqa: E402
from src.metrics import accuracy_from_logits  # noqa: E402


def per_class_acc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_vgg: bool,
    num_classes: int = 10,
) -> List[float]:
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Only count digit classes 0-9 in test
            m = y < NEGATIVE_CLASS
            if not m.any():
                continue
            logits = forward_batch(model, x[m], is_vgg)
            pred = logits.argmax(dim=1)
            yy = y[m]
            for c in range(num_classes):
                sub = yy == c
                if not sub.any():
                    continue
                total[c] += int(sub.sum().item())
                correct[c] += (pred[sub] == c).sum().item()
    return [correct[i] / max(total[i], 1) for i in range(num_classes)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", choices=["custom", "vgg"], required=True, help="which run"
    )
    p.add_argument("--data-dir", type=str, default="data/svhn")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--device", type=str, default="auto", help="cuda, cpu, or auto"
    )
    p.add_argument("--neg-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    ck_dir = root / args.checkpoints_dir
    res_dir = root / args.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    model_name = "custom" if args.model == "custom" else "vgg16"
    ck = ck_dir / f"{model_name}_best.pt"
    if not ck.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ck} — train that model first.")

    model, is_vgg, _ = load_model_from_checkpoint(ck, device)
    _, _, test_ds = get_svhn_11_datasets(
        str(data_dir), neg_ratio=args.neg_ratio, seed=args.seed
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_batch(model, x, is_vgg)
            loss = criterion(logits, y)
            b = x.size(0)
            tot_loss += loss.item() * b
            tot_acc += accuracy_from_logits(logits, y) * b
            n += b
    test_loss = tot_loss / max(n, 1)
    test_acc = tot_acc / max(n, 1)
    pca = per_class_acc(model, test_loader, device, is_vgg, num_classes=10)

    metrics = {
        "model": model_name,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_acc_0_9": pca,
    }
    out_json = res_dir / f"metrics_{model_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    table_path = res_dir / "comparison_table.csv"
    row = {
        "model": model_name,
        "test_acc": f"{test_acc:.6f}",
        "test_loss": f"{test_loss:.6f}",
    }
    write_header = not table_path.is_file()
    with open(
        table_path, "a", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_acc", "test_loss"])
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Wrote {out_json}  test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
