#!/usr/bin/env python3
"""Create assets/demo_inputs/1.png–5.png for varied scale, rotation, position, lighting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _light_gradient(w: int, h: int, base: int) -> Image.Image:
    im = Image.new("RGB", (w, h))
    px = im.load()
    for x in range(w):
        v = int(base + 40 * x / w) % 256
        for y in range(h):
            px[x, y] = (v, (v + 20) % 256, (v + 10) % 256)
    return im


def _make(i: int, w: int, h: int) -> Image.Image:
    if i == 0:
        im = _light_gradient(w, h, 60)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
        except OSError:
            font = ImageFont.load_default()
        t = "42"
        dr.text((w // 2 - 20, h // 2 - 20), t, fill=(255, 200, 30), font=font)
    elif i == 1:
        im = _light_gradient(w, h, 80)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except OSError:
            font = ImageFont.load_default()
        dr.text((40, 50), "7", fill=(20, 255, 200), font=font)
        im = im.rotate(12, resample=Image.BICUBIC, expand=False, fillcolor=(50, 50, 50))
    elif i == 2:
        im = _light_gradient(w, h, 30)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except OSError:
            font = ImageFont.load_default()
        dr.text((w - 160, 20), "123", fill=(200, 200, 255), font=font)
    elif i == 3:
        im = _light_gradient(w, h, 15)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except OSError:
            font = ImageFont.load_default()
        dr.text((50, 80), "5", fill=(250, 250, 250), font=font)
    else:
        im = _light_gradient(w, h, 50)
        a = np.asarray(im, dtype=np.float32)
        g = np.random.default_rng(42 + i)
        noise = (g.random(a.shape) * 40.0 - 20.0).astype(np.float32)
        a = np.clip(a + noise, 0, 255).astype(np.uint8)
        im = Image.fromarray(a)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except OSError:
            font = ImageFont.load_default()
        dr.text((50, 80), "99", fill=(180, 255, 180), font=font)
    return im


def main() -> None:
    out = ROOT / "assets" / "demo_inputs"
    out.mkdir(parents=True, exist_ok=True)
    w, h = 480, 200
    for i in range(1, 6):
        im = _make(i - 1, w, h)
        path = out / f"{i}.png"
        im.save(str(path), "PNG")
    print(f"Wrote 1..5 to {out}")


if __name__ == "__main__":
    main()
