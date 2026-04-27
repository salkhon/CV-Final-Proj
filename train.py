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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.svhn_11 import get_svhn_11_datasets  # noqa: E402
from src.infer_utils import forward_batch  # noqa: E402
from src.metrics import accuracy_from_logits  # noqa: E402
from src.models import CustomCNN, build_vgg16_classifier, vgg_param_groups  # noqa: E402
from src.utils.run_logging import configure_logging, get_logger  # noqa: E402

LOG = get_logger("cv_proj.train")

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
    *,
    show_progress: bool = True,
    pbar_desc: str = "batches",
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
    it = loader
    if show_progress:
        it = tqdm(
            loader,
            desc=pbar_desc,
            leave=False,
            unit="batch",
        )
    for x, y in it:
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
        li = loss.item()
        ai = accuracy_from_logits(logits, y)
        tot_loss += li * b
        tot_acc += ai * b
        n += b
        if show_progress and isinstance(it, tqdm):
            it.set_postfix(loss=f"{li:.4f}", acc=f"{ai:.3f}")
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
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (repeat for debug)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm batch bars (logs only)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest; try 4–8 on Linux/WSL+GPU for faster loading)",
    )
    args = p.parse_args()

    configure_logging(verbosity=args.verbose)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        LOG.info(
            "CUDA device: %s  (%s)  torch=%s cuda=%s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
            torch.__version__,
            torch.version.cuda,
        )
    else:
        LOG.info("device=cpu  torch=%s", torch.__version__)

    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    out_dir = root / args.out_dir
    results_dir = root / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = "custom" if args.model == "custom" else "vgg16"
    torch.manual_seed(args.seed)

    LOG.info(
        "Step 1/4: load SVHN-11 (neg_ratio=%s) from %s",
        args.neg_ratio,
        data_dir,
    )
    train_ds, val_ds, _ = get_svhn_11_datasets(
        str(data_dir), neg_ratio=args.neg_ratio, seed=args.seed
    )
    pin_mem = device.type == "cuda"
    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=args.num_workers > 0,
    )
    LOG.info(
        "Loaded train=%d val=%d batches (bs=%d)",
        len(train_ds),
        len(val_ds),
        args.batch_size,
    )

    LOG.info(
        "Step 2/4: build model=%s  device=%s  vgg_backbone_lr=%s",
        args.model,
        device,
        args.vgg_lr_backbone if args.model == "vgg" else "n/a",
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
    prog = not args.no_progress
    LOG.info(
        "Step 3/4: training up to %d epochs (patience=%d)  -> %s",
        args.epochs,
        args.patience,
        best_path,
    )
    for epoch in range(1, args.epochs + 1):
        LOG.info("--- Epoch %d / %d ---", epoch, args.epochs)
        tr_loss, tr_acc = run_epoch(
            model,
            tr_loader,
            criterion,
            device,
            is_vgg,
            True,
            opt,
            show_progress=prog,
            pbar_desc=f"e{epoch}/{args.epochs} train",
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            is_vgg,
            False,
            None,
            show_progress=prog,
            pbar_desc=f"e{epoch}/{args.epochs} val  ",
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        LOG.info(
            "epoch %03d summary  train loss/acc: %.4f / %.4f  |  val loss/acc: %.4f / %.4f",
            epoch,
            tr_loss,
            tr_acc,
            va_loss,
            va_acc,
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
                LOG.info("early stopping at epoch %d (no val improvement)", epoch)
                break

    LOG.info("Step 4/4: save history, curves, meta")
    hist_path = results_dir / f"history_{model_name}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if plt is not None:
        plot_curves(
            history,
            results_dir / f"curves_{model_name}.png",
            model_name,
        )
    else:
        LOG.warning("matplotlib not available; skip curves_*.png")
    # Save last meta for evaluation
    meta = {
        "model": model_name,
        "checkpoint": str(best_path.relative_to(root)),
        "best_val_acc": best_val,
    }
    with open(results_dir / f"run_meta_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOG.info("done  checkpoint=%s  best_val_acc=%.4f", best_path, best_val)


if __name__ == "__main__":
    main()
