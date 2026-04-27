#!/usr/bin/env python3
"""
Train custom CNN or VGG16 head on SVHN-11. Writes history JSON, curves, checkpoint.
All paths relative to project root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.svhn_11 import get_svhn_11_datasets  # noqa: E402
from src.infer_utils import forward_batch  # noqa: E402
from src.metrics import accuracy_from_logits  # noqa: E402
from src.models import CustomCNN, build_vgg16_classifier, vgg_param_groups  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore


def build_model(
    name: str, device: torch.device
) -> Tuple[nn.Module, bool]:
    name = name.lower()
    if name == "custom":
        return CustomCNN().to(device), False
    if name == "vgg":
        m = build_vgg16_classifier(pretrained=True)
        return m.to(device), True
    raise ValueError(f"Unknown model: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_vgg: bool,
    train: bool,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    use_cuda_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_cuda_amp):
                logits = forward_batch(model, x, is_vgg)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = forward_batch(model, x, is_vgg)
                loss = criterion(logits, y)
        b = x.size(0)
        tot_loss += loss.item() * b
        tot_acc += accuracy_from_logits(logits, y) * b
        n += b
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


def plot_curves(
    history: Dict[str, List[float]], out_path: Path, title: str
) -> None:
    if plt is None:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title(title + " loss")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].set_title(title + " acc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", choices=["custom", "vgg"], required=True, help="classifier"
    )
    p.add_argument("--data-dir", type=str, default="data/svhn")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vgg-lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--neg-ratio", type=float, default=0.2)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda, cpu, or auto (use CUDA if available)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    out_dir = root / args.out_dir
    results_dir = root / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = "custom" if args.model == "custom" else "vgg16"
    torch.manual_seed(args.seed)

    train_ds, val_ds, _ = get_svhn_11_datasets(
        str(data_dir), neg_ratio=args.neg_ratio, seed=args.seed
    )
    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model, is_vgg = build_model(args.model, device)
    criterion = nn.CrossEntropyLoss()
    if is_vgg:
        opt = optim.Adam(
            vgg_param_groups(model, lr_head=args.lr, lr_backbone=args.vgg_lr_backbone),
            weight_decay=args.weight_decay,
        )
    else:
        opt = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val = -1.0
    patience_left = args.patience
    best_path = out_dir / f"{model_name}_best.pt"
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, tr_loader, criterion, device, is_vgg, True, opt
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, device, is_vgg, False, None
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(
            f"epoch {epoch:03d}  train {tr_loss:.4f}/{tr_acc:.4f}  "
            f"val {va_loss:.4f}/{va_acc:.4f}"
        )
        if va_acc > best_val + 1e-6:
            best_val = va_acc
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": best_val,
                    "is_vgg": is_vgg,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}")
                break

    hist_path = results_dir / f"history_{model_name}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if plt is not None:
        plot_curves(
            history,
            results_dir / f"curves_{model_name}.png",
            model_name,
        )
    # Save last meta for evaluation
    meta = {
        "model": model_name,
        "checkpoint": str(best_path.relative_to(root)),
        "best_val_acc": best_val,
    }
    with open(results_dir / f"run_meta_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {best_path} best_val_acc={best_val:.4f}")


if __name__ == "__main__":
    main()
