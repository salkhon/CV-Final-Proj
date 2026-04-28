# CV Final Project: Digit detection and recognition

This repository implements a **two-stage** pipeline: **traditional region proposals** (MSER, image pyramid, optional sliding window, NMS) followed by a **PyTorch CNN** (custom from scratch or VGG-16 with ImageNet pre-training) with **11 classes** (digits 0–9 plus a *non-digit* class). It satisfies the course requirements in `problem.txt` (no one-stage object detectors; relative paths; single `run.py` for grading).

## Environment (Conda)

```bash
conda env create -f cv_proj.yml
conda activate cv_proj
```

**Why the file changed from the course handout:** the TA `python=3.8.8` + old `pytorch` conda builds do not exist for **Apple Silicon (osx-arm64)**, and `tensorflow-gpu` is not used by this project (the assignment allows PyTorch only). [`cv_proj.yml`](cv_proj.yml) uses **Python 3.10**, **conda-forge** for scientific libs, and **PyTorch/torchvision via pip** so the same file can install on macOS arm64 and Linux. The code uses standard PyTorch APIs compatible with the pinned pip versions.

If `conda env create` still fails, try: `conda install -c conda-forge mamba` then `mamba env create -f cv_proj.yml`.

**NumPy 2.x vs PyTorch:** the pinned `torch` / `torchvision` pip packages expect **NumPy 1.x**. If you see `Numpy is not available` or `_ARRAY_API not found`, your conda pulled NumPy 2. Fix the active env with:

`conda install -c conda-forge "numpy>=1.21,<2" --force-reinstall`

Then continue (`bash scripts/run_all_experiments.sh` or `python train.py ...`).

**Progress:** training, evaluation, data download, and `run.py` use **timestamped logging** and **tqdm** progress bars. Pass `-v` / `--verbose` on any of those Python scripts for more detail, or `--no-progress` to disable tqdm (logs only). The shell script `run_all_experiments.sh` prints numbered steps and sets `PYTHONUNBUFFERED=1` so log lines show up immediately.

**Platform note:** all paths are built with `pathlib` relative to the project root. Use `cpu` on machines without a GPU: `--device cpu`.

### Windows WSL + NVIDIA GPU (e.g. RTX 5080)

The code is **Linux-oriented** (bash, `pathlib`); **WSL2 with Ubuntu** is the right place to run it, not legacy Windows `cmd` paths.

1. **Clone the repo inside the Linux filesystem** (e.g. `~/projects/...`), not only under `/mnt/c/...`, so I/O and PyTorch are not unnecessarily slow.
2. **NVIDIA driver on Windows** + WSL CUDA support: after `conda activate cv_proj`, check the GPU with:
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`
3. **RTX 50-series (Blackwell, e.g. 5080)** use **CUDA capability sm_120**. Older wheels such as `torch 2.0+cu117` only ship kernels up to about **sm_86**, which causes `no kernel image is available for execution on the device`. After `conda env create`, **always** install the CUDA 12.8 PyTorch build on Linux+NVIDIA:
   `bash scripts/upgrade_pytorch_cuda128.sh`
   (or `pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128`). Then confirm with `python -c "import torch; print(torch.__version__, torch.version.cuda)"` — you want a **cu128** build. Training/eval/`run.py` call a small GPU self-test and print this hint if kernels are missing. Keep **NumPy below 2.x** unless you verify your stack supports NumPy 2.
4. **Faster dataloading:** pass `--num-workers 6` (or similar) to `train.py` / `evaluate.py`, or run the shell script with:
   `NUM_WORKERS=6 DEVICE=cuda bash scripts/run_all_experiments.sh`
   Training already uses **mixed precision on CUDA**, `pin_memory=True` when on GPU, and `non_blocking` transfers.
5. **Demo image fonts:** [`src/utils/fonts.py`](src/utils/fonts.py) tries DejaVu/Liberation on Linux, Arial on Windows, and Helvetica on macOS so `scripts/generate_demo_inputs.py` works everywhere.

## One-command experiment pipeline

From the project root (after `conda activate cv_proj`):

```bash
bash scripts/run_all_experiments.sh
```

Optional environment variables:

- `TRAIN_EPOCHS` (default `25`) – reduce for a quick test (e.g. `3`).
- `DEVICE` (default `auto`) – `auto`, `cuda`, or `cpu`.
- `NUM_WORKERS` (default `0`) – DataLoader workers; use `4`–`8` on WSL/Linux with a GPU for faster epoch iteration.

Steps inside the script:

1. `python scripts/download_svhn.py` – SVHN to `data/svhn/`
2. `python train.py --model custom ...` and `python train.py --model vgg ...`
3. `python evaluate.py` for each model
4. `python select_best.py` – writes `results/best_model.txt` (`custom` or `vgg16`)

**Report artifacts:** for each run, the repo writes:

- `results/history_{custom,vgg16}.json` – per-epoch train/val loss and accuracy  
- `results/curves_{custom,vgg16}.png` – training/validation curves  
- `checkpoints/{custom,vgg16}_best.pt` – best checkpoints (by validation accuracy, early stopping)  
- `results/metrics_{custom,vgg16}.json` – test loss/accuracy; `results/comparison_table.csv`  
- `results/best_model.txt` – which model the inference script should load  

To re-build the comparison table from scratch, delete `results/comparison_table.csv` before re-running `evaluate.py`.

**Hyperparameters (for the write-up):** Cross-entropy loss, Adam, early stopping on **validation accuracy** with `--patience` (default 5), batch size 128, learning rate `1e-3` (custom) or `1e-3` on the new head and `1e-4` on the VGG feature backbone (`--vgg-lr-backbone`).

## Grader: single entry point

The assignment requires a single file **`run.py`** that writes **`graded_images/1.png` … `graded_images/5.png`**.

1. (Optional) Generate demo source images: `python scripts/generate_demo_inputs.py` → `assets/demo_inputs/1.png`…`5.png` (diverse scale, rotation, position, lighting, noise).
2. Train and select the best model (see above) so `checkpoints/*_best.pt` and `results/best_model.txt` exist.
3. Run: `python run.py`

`run.py` loads the best checkpoint, runs the full proposal + classifier pipeline, and saves annotated results under `graded_images/`.

## Project layout (short)

| Path | Role |
|------|------|
| `src/data/svhn_11.py` | SVHN + synthetic negatives, train/val split |
| `src/models/` | `CustomCNN`, VGG-16 with 11-class head |
| `src/detection/` | Preprocessing, pyramid, MSER, sliding window, NMS, pipeline |
| `train.py` | Training + curves |
| `evaluate.py` | Test metrics |
| `select_best.py` | Writes `results/best_model.txt` |
| `run.py` | Grader entry |
| `scripts/upgrade_pytorch_cuda128.sh` | Reinstall torch/torchvision for **RTX 50-series** (CUDA 12.8 / sm_120) |
| `src/utils/cuda_kernel_check.py` | Fails fast with install hint if GPU kernels are missing |

## Report checklist (you generate the PDF)

- Methods: two-stage (MSER/pyramid/sliding window vs YOLO), loss and optimizer choices, why early stopping, what changed on VGG-16.  
- Experiments: use `results/curves_*.png`, `results/metrics_*.json`, and `comparison_table.csv`.  
- Show success and **failure** cases (extra figures in the report) and why some scenes fail.  

## Rules recap

- **Relative paths only** in submitted code.  
- **No YOLO/SSD** as the main detector; proposals are **separate** from the CNN.  
- **You must train**; submission is not a pre-trained off-the-shelf sequence reader.  
- CNN implementation uses **PyTorch** as in `cv_proj.yml`.  
