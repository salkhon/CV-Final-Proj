#!/usr/bin/env bash
# Reinstall PyTorch/torchvision with CUDA 12.8 (supports NVIDIA Blackwell / sm_120, e.g. RTX 5080).
# Run inside conda env: conda activate cv_proj && bash scripts/upgrade_pytorch_cuda128.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
echo "Upgrading torch + torchvision from PyTorch cu128 index (for RTX 50-series / sm_120)..."
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
if torch.cuda.is_available():
    x = torch.randn(1, 3, 32, 32, device="cuda")
    w = torch.nn.Conv2d(3, 1, 3, bias=False).cuda().weight
    y = torch.nn.functional.conv2d(x, w)
    torch.cuda.synchronize()
    print("GPU smoke OK:", y.shape, torch.cuda.get_device_name(0))
else:
    print("CUDA not available in this env (OK for CPU-only machines).")
PY
echo "Done."
