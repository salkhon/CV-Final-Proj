"""
Detect PyTorch builds that cannot run kernels on the current GPU (e.g. RTX 50-series
sm_120 with an old torch+cu117 wheel that only ships sm_86 and below).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def raise_if_cuda_kernels_missing(device: torch.device) -> None:
    """
    Run a tiny conv on `device`. If it fails with "no kernel image", raise with fix hints.
    """
    if device.type != "cuda":
        return
    try:
        m = nn.Conv2d(3, 1, 3, bias=False).to(device)
        x = torch.randn(1, 3, 32, 32, device=device)
        _ = m(x)
        torch.cuda.synchronize()
    except RuntimeError as e:
        err = str(e).lower()
        if "no kernel image" in err or "kernel image" in err:
            cap = torch.cuda.get_device_capability(0)
            raise RuntimeError(
                "This PyTorch wheel was not built for your GPU architecture "
                f"(capability {cap}). RTX 50-series (Blackwell, sm_120) needs "
                "PyTorch 2.7+ with CUDA 12.8 wheels, not torch 2.0+cu11.\n\n"
                "Fix (inside your conda env):\n"
                "  bash scripts/upgrade_pytorch_cuda128.sh\n\n"
                "Or manually:\n"
                "  pip install --upgrade torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu128\n\n"
                f"Original error: {e}"
            ) from e
        raise
