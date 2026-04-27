#!/usr/bin/env bash
# End-to-end: download SVHN, train both classifiers, evaluate, select best model.
# Run from repo root: bash scripts/run_all_experiments.sh
# Expect long run time on CPU; set TRAIN_EPOCHS e.g. 3 for a dry run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/download_svhn.py
: "${TRAIN_EPOCHS:=25}"
: "${DEVICE:=auto}"
echo "=== train custom (from scratch) ==="
python3 train.py --model custom --data-dir data/svhn --epochs "$TRAIN_EPOCHS" --device "$DEVICE"
echo "=== train VGG16 (fine-tune) ==="
python3 train.py --model vgg --data-dir data/svhn --epochs "$TRAIN_EPOCHS" --device "$DEVICE" --vgg-lr-backbone 1e-4 --lr 1e-3
echo "=== evaluate on test set ==="
python3 evaluate.py --model custom --data-dir data/svhn
python3 evaluate.py --model vgg --data-dir data/svhn
echo "=== select best (writes results/best_model.txt) ==="
python3 select_best.py
echo "Done. Grader: python3 run.py  (writes graded_images/1.png ... 5.png)"
