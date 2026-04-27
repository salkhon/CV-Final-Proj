"""Project root and relative path helpers. No absolute paths in user config."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_ROOT: Optional[Path] = None


def project_root() -> Path:
    """Return repository root (parent of `src/` or cwd if run as script)."""
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    here = Path(__file__).resolve()
    for p in (here.parent.parent, *here.parents):
        if (p / "src").is_dir() and (p / "run.py").is_file():
            _ROOT = p
            return _ROOT
    _ROOT = Path.cwd()
    return _ROOT


def rel_to_root(*parts: str) -> Path:
    return project_root().joinpath(*parts)
