#!/usr/bin/env bash
# End-to-end: download SVHN, train both classifiers, evaluate, select best model.
# Run from repo root: bash scripts/run_all_experiments.sh
# Expect long run time on CPU; set TRAIN_EPOCHS e.g. 3 for a dry run.
# On Linux/WSL+GPU, try: NUM_WORKERS=6 DEVICE=cuda bash scripts/run_all_experiments.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "Step 1/6: download SVHN"
python3 scripts/download_svhn.py
: "${TRAIN_EPOCHS:=25}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=0}"
log "Step 2/6: train custom CNN (from scratch)"
python3 train.py --model custom --data-dir data/svhn --epochs "$TRAIN_EPOCHS" --device "$DEVICE" --num-workers "$NUM_WORKERS"
log "Step 3/6: fine-tune VGG16"
python3 train.py --model vgg --data-dir data/svhn --epochs "$TRAIN_EPOCHS" --device "$DEVICE" --vgg-lr-backbone 1e-4 --lr 1e-3 --num-workers "$NUM_WORKERS"
log "Step 4/6: evaluate both models on test set"
python3 evaluate.py --model custom --data-dir data/svhn --num-workers "$NUM_WORKERS"
python3 evaluate.py --model vgg --data-dir data/svhn --num-workers "$NUM_WORKERS"
log "Step 5/6: select best model -> results/best_model.txt"
python3 select_best.py
log "Step 6/6: done. Optional grader: python3 run.py  (graded_images/1.png ... 5.png)"
