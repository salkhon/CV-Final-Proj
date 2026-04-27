"""Consistent logging setup for train/eval/download/run scripts."""

from __future__ import annotations

import logging
import sys
_configured: bool = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(verbosity: int = 0) -> None:
    """
    `verbosity`: 0 = INFO, 1+ = DEBUG. Safe to call more than once.
    """
    global _configured
    level = logging.DEBUG if verbosity else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if not _configured:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(h)
        _configured = True
    else:
        logging.getLogger().setLevel(level)
        for h in logging.getLogger().handlers:
            h.setLevel(level)
