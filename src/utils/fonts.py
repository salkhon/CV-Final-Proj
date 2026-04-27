"""Cross-platform font paths for PIL (macOS, Linux/WSL, Windows)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Common locations; first match wins for ImageFont.truetype(..., size)
_CANDIDATES: List[str] = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu/WSL
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",  # Windows (native or WSL interop)
    str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"),
]


def truetype_font(size: int):
    """
    Return a PIL ImageFont for drawing text, or default bitmap font if none found.
    """
    from PIL import ImageFont

    for path in _CANDIDATES:
        if not path:
            continue
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()
