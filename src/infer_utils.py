"""Load trained classifiers for evaluation and run.py pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import CustomCNN, build_vgg16_classifier


def forward_batch(
    model: nn.Module, x: torch.Tensor, is_vgg: bool
) -> torch.Tensor:
    """Same as training: 32x32 input; VGG is resized to 224 inside."""
    if is_vgg:
        x = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
    return model(x)


def load_model_from_checkpoint(
    ckpt_path: Path, device: torch.device
) -> Tuple[nn.Module, bool, Dict[str, Any]]:
    data = torch.load(ckpt_path, map_location=device)
    is_vgg = data.get("is_vgg", False)
    if is_vgg:
        model: nn.Module = build_vgg16_classifier(pretrained=False)
    else:
        model = CustomCNN()
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()
    return model, is_vgg, data


def logits_to_digit(
    logits: torch.Tensor, reject_class: int = 10, min_conf: float = 0.5
) -> Optional[Tuple[int, float]]:
    """
    If argmax is reject or max prob < min_conf, return None.
    `logits` is shape (1, C) or (C,); returns digit in 0-9 and confidence.
    """
    if logits.dim() == 1:
        lg = logits.unsqueeze(0)
    else:
        lg = logits
    p = F.softmax(lg, dim=-1)[0]
    conf, idx = p.max(0)
    c = int(idx.item())
    conf_f = conf.item()
    if c == reject_class or conf_f < min_conf:
        return None
    return c, conf_f
