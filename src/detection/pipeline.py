"""
Full inference: OpenCV read -> preprocessed proposals -> batched CNN -> ordered string.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.data.svhn_11 import NEGATIVE_CLASS
from src.detection.roi import Box, preprocess_for_proposals, proposal_boxes
from src.infer_utils import load_model_from_checkpoint, logits_to_digit


def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """Apply same denoise + CLAHE as used for region proposals, return BGR uint8."""
    out, _g, _y = preprocess_for_proposals(bgr)
    return out


def _crop_to_tensor32(bgr: np.ndarray, box: Box) -> torch.Tensor:
    x, y, w, h = box
    h0, w0 = bgr.shape[:2]
    x = max(0, min(x, w0 - 1))
    y = max(0, min(y, h0 - 1))
    w = max(1, min(w, w0 - x))
    h = max(1, min(h, h0 - y))
    patch = bgr[y : y + h, x : x + w]
    if patch.size == 0:
        return torch.zeros(1, 3, 32, 32)
    s = max(w, h)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    off_x = (s - w) // 2
    off_y = (s - h) // 2
    canvas[off_y : off_y + h, off_x : off_x + w] = patch
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
    )
    return t


def _sort_left_to_right(boxes: List[Box], labels: List[int]) -> str:
    if not boxes:
        return ""
    items = list(
        zip(
            [b[0] + 0.5 * b[2] for b in boxes],
            labels,
        )
    )
    items.sort(key=lambda t: t[0])
    return "".join(str(d) for _xc, d in items)


@torch.no_grad()
def detect_digit_sequence(
    bgr: np.ndarray,
    model: nn.Module,
    is_vgg: bool,
    device: torch.device,
    min_conf: float = 0.45,
    max_proposals: int = 64,
) -> Tuple[str, List[Box], List[Tuple[int, float]]]:
    bgr2 = preprocess_image(bgr)
    boxes = proposal_boxes(bgr2)[:max_proposals]
    if not boxes:
        return "", [], []
    batch = torch.cat(
        [_crop_to_tensor32(bgr2, b) for b in boxes], dim=0
    )
    logits: torch.Tensor
    if batch.size(0) > 32:
        parts = []
        for i in range(0, batch.size(0), 32):
            ch = batch[i : i + 32]
            if is_vgg:
                t = torch.nn.functional.interpolate(
                    ch, size=(224, 224), mode="bilinear", align_corners=False
                )
            else:
                t = ch
            parts.append(model(t.to(device)))
        logits = torch.cat(parts, dim=0)
    else:
        if is_vgg:
            t = torch.nn.functional.interpolate(
                batch, size=(224, 224), mode="bilinear", align_corners=False
            )
        else:
            t = batch
        logits = model(t.to(device))
    out_labels: List[Tuple[int, float]] = []
    kept_boxes: List[Box] = []
    for i, box in enumerate(boxes):
        ld = logits[i : i + 1]
        r = logits_to_digit(
            ld, reject_class=NEGATIVE_CLASS, min_conf=min_conf
        )
        if r is not None:
            d, conf = r
            out_labels.append((d, conf))
            kept_boxes.append(box)
    digits = [d for d, _c in out_labels]
    s = _sort_left_to_right(kept_boxes, digits)
    return s, kept_boxes, out_labels


def run_on_image_path(
    path: Path,
    ckpt: Path,
    device: torch.device,
    min_conf: float = 0.45,
) -> Tuple[str, List[Box], List[Tuple[int, float]]]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    model, is_vgg, _ = load_model_from_checkpoint(ckpt, device)
    return detect_digit_sequence(
        bgr, model, is_vgg, device, min_conf=min_conf
    )
