#!/usr/bin/env python3
"""
Grader entry point: run full detection + recognition pipeline and write
graded_images/1.png ... graded_images/5.png. Single file; all paths relative to repo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.pipeline import (  # noqa: E402
    detect_digit_sequence,
    preprocess_image,
)
from src.infer_utils import load_model_from_checkpoint  # noqa: E402


def _resolve_checkpoint() -> Path:
    best = ROOT / "results" / "best_model.txt"
    choices = [("custom", ROOT / "checkpoints" / "custom_best.pt"), ("vgg16", ROOT / "checkpoints" / "vgg16_best.pt")]
    if best.is_file():
        name = best.read_text(encoding="utf-8").strip().lower()
        for n, p in choices:
            if n == name and p.is_file():
                return p
    for _n, p in choices:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No checkpoint found. Train with train.py and evaluate, or use select_best."
    )


def _draw(
    bgr: np.ndarray,
    pred: str,
    boxes: list,
) -> np.ndarray:
    vis = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
    msg = f"seq={pred!r}" if pred else "seq= <none>"
    cv2.putText(
        vis,
        msg,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 200, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    return vis


def _fallback_synthetic() -> List[np.ndarray]:
    """If demo assets are missing, simple synthetic 5 images (diverse)."""
    out: List[np.ndarray] = []
    for n, s in enumerate(["42", "7", "305", "9", "18"], start=1):
        w, h = 400, 180
        bgr = np.full((h, w, 3), 32, np.uint8)
        # lighting gradient
        for x in range(w):
            bgr[:, x, :] = 32 + (x * 50) // w
        ang = (n - 1) * 5
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(ang - 10), 0.7 + 0.08 * n)
        t = 2 if n == 1 else 1
        cv2.putText(
            bgr,
            s,
            (20 + 30 * t, 120),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 + 0.12 * n,
            (200, 220, 255),
            3,
        )
        bgr = cv2.warpAffine(
            bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40)
        )
        out.append(bgr)
    return out


def main() -> None:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    ck = _resolve_checkpoint()
    model, is_vgg, _ = load_model_from_checkpoint(ck, dev)
    out_dir = ROOT / "graded_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = ROOT / "assets" / "demo_inputs"
    for i in range(1, 6):
        p = demo_dir / f"{i}.png"
        bgr: np.ndarray
        if p.is_file():
            bgr = cv2.imread(str(p))
            if bgr is None:
                syns = _fallback_synthetic()
                bgr = syns[i - 1] if i - 1 < len(syns) else np.zeros((120, 300, 3), np.uint8)
        else:
            syns = _fallback_synthetic()
            bgr = syns[i - 1] if i - 1 < len(syns) else np.zeros((120, 300, 3), np.uint8)
        bgr2 = np.asarray(bgr, dtype=np.uint8)
        pred, boxes, _info = detect_digit_sequence(
            bgr2, model, is_vgg, dev, min_conf=0.4, max_proposals=80
        )
        vis = _draw(preprocess_image(bgr2) if bgr2.size else bgr2, pred, boxes)
        dst = out_dir / f"{i}.png"
        if not cv2.imwrite(str(dst), vis):
            raise OSError(f"Failed to write {dst}")


if __name__ == "__main__":
    main()
