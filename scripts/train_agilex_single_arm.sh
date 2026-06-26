#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/hyn/agilex

MODEL_ROOT="/mnt/data/hyn/model"
CONFIG_NAME="pi05_agilex_right_finetune"
EXP_NAME="stack_bowl_0617_right_4gpu"

mkdir -p \
  "${MODEL_ROOT}/openpi_cache" \
  "${MODEL_ROOT}/agilex_assets" \
  "${MODEL_ROOT}/agilex_checkpoints"

export OPENPI_DATA_HOME="${MODEL_ROOT}"
export WANDB_MODE=online
export WANDB_API_KEY="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"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if [[ "${WANDB_API_KEY}" == "PASTE_YOUR_WANDB_API_KEY_HERE" ]]; then
  echo "Please set WANDB_API_KEY before running, or replace it in this script."
  exit 1
fi

echo "[1/2] Computing normalization stats for ${CONFIG_NAME}"
uv run scripts/compute_agilex_right_norm_stats.py

echo "[2/2] Starting 4-GPU training: ${EXP_NAME}"
uv run scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --fsdp-devices=4 \
  --overwrite
