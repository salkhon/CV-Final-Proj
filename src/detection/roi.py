"""
Region proposals: image pyramid, MSER, optional sliding window, NMS.
Separate from the CNN (assignment requirement: no one-stage detector).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h in original image (integer pixels)


def preprocess_for_proposals(
    bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Denoise, CLAHE on luminance. Returns (bgr_out, gray, clahe_gray).
    """
    bilat = cv2.bilateralFilter(bgr, 5, 50, 50)
    ycrcb = cv2.cvtColor(bilat, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y2 = clahe.apply(y)
    merge = cv2.merge([y2, cr, cb])
    out = cv2.cvtColor(merge, cv2.COLOR_YCrCb2BGR)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return out, gray, y2


def _map_from_level(
    x: int, y: int, w: int, h: int, level_w: int, level_h: int, base_w: int, base_h: int
) -> Box:
    sx = base_w / max(level_w, 1)
    sy = base_h / max(level_h, 1)
    x0 = int(round(x * sx))
    y0 = int(round(y * sy))
    w0 = int(max(4, round(w * sx)))
    h0 = int(max(4, round(h * sy)))
    x0 = max(0, min(x0, max(0, base_w - 1)))
    y0 = max(0, min(y0, max(0, base_h - 1)))
    w0 = min(w0, base_w - x0)
    h0 = min(h0, base_h - y0)
    return (x0, y0, max(1, w0), max(1, h0))


def _mser_on_gray(gray: np.ndarray) -> List[Box]:
    mser = cv2.MSER_create()
    try:
        mser.setMinArea(30)  # type: ignore[attr-defined]
        mser.setMaxArea(2_000)  # type: ignore[attr-defined]
    except (AttributeError, cv2.error):
        pass
    out: List[Box] = []
    try:
        ret = mser.detectRegions(gray)
    except cv2.error:
        return out
    if not ret:
        return out
    regions = ret[0] if isinstance(ret, tuple) else ret
    for p in regions:
        pts = np.asarray(p, dtype=np.int32)
        if pts.size < 6:
            continue
        if pts.ndim == 2 and pts.shape[1] == 2:
            x, y, w, h = cv2.boundingRect(pts)
        else:
            x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
        if w <= 0 or h <= 0:
            continue
        ar = w / max(h, 1)
        if ar < 0.12 or ar > 1.1:
            continue
        a = w * h
        if a < 20 or a > 8_000:
            continue
        out.append((x, y, w, h))
    return out


def _sliding_windows(gray: np.ndarray, win: int, stride: int) -> List[Box]:
    h, w = gray.shape[:2]
    out: List[Box] = []
    if win > min(h, w):
        return out
    for yy in range(0, h - win + 1, stride):
        for xx in range(0, w - win + 1, stride):
            patch = gray[yy : yy + win, xx : xx + win]
            if float(patch.std()) < 2.0:
                continue
            out.append((xx, yy, win, win))
    return out


def nms(
    boxes: List[Box], scores: List[float], iou_th: float = 0.35
) -> List[int]:
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 0] + b[:, 2]
    y2 = b[:, 1] + b[:, 3]
    order = s.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        o = order[1:]
        ix1 = np.maximum(x1[i], x1[o])
        iy1 = np.maximum(y1[i], y1[o])
        ix2 = np.minimum(x2[i], x2[o])
        iy2 = np.minimum(y2[i], y2[o])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_o = (x2[o] - x1[o]) * (y2[o] - y1[o])
        union = area_i + area_o - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_th)[0]
        order = o[inds] if inds.size else order[:0]
    return keep


def non_maximum_suppression(
    boxes: List[Box], iou_th: float = 0.35
) -> List[Box]:
    if not boxes:
        return []
    areas = [w * h for (_x, _y, w, h) in boxes]
    order = nms(
        list(boxes), [float(a) for a in areas], iou_th=iou_th
    )
    return [boxes[i] for i in order]


def proposal_boxes(
    bgr: np.ndarray,
    pyramid_scales: Tuple[float, ...] = (1.0, 0.75, 0.5),
    use_sliding: bool = True,
    max_sliding: int = 40,
) -> List[Box]:
    """
    Multi-scale MSER (and optional sliding windows on full-res) -> NMS.
    """
    base_h, base_w = bgr.shape[:2]
    _, _gray, clahe_y = preprocess_for_proposals(bgr)
    all_boxes: List[Box] = []
    for sc in pyramid_scales:
        if sc >= 0.99:
            nw, nh = base_w, base_h
            small = bgr
            gy = clahe_y
        else:
            nw = max(8, int(base_w * sc))
            nh = max(8, int(base_h * sc))
            small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            gy = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for bx, by, bw, bh in _mser_on_gray(gy):
            all_boxes.append(
                _map_from_level(bx, by, bw, bh, nw, nh, base_w, base_h)
            )
    if use_sliding:
        sw: List[Box] = []
        g = clahe_y
        for win, stride in ((24, 10), (32, 12)):
            sw.extend(
                _sliding_windows(g, win=win, stride=stride)[: max_sliding // 2]
            )
        for bx, by, bw, bh in sw[:max_sliding]:
            all_boxes.append((bx, by, bw, bh))
    if not all_boxes:
        return []
    merged = non_maximum_suppression(all_boxes, iou_th=0.3)
    return merged
