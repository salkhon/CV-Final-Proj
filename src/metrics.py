"""Batch and epoch accuracy helpers."""

from __future__ import annotations

import torch


def accuracy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()
