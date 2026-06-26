#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${AGILEX_TRAIN_IN_TMUX:-}" && -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed or not in PATH."
    exit 1
  fi

  SCRIPT_PATH="$(readlink -f "$0")"
  SESSION_NAME="hyn"
  WINDOW_NAME="arrange_flower_train"
  TRAIN_CMD="AGILEX_TRAIN_IN_TMUX=1 bash ${SCRIPT_PATH}; exec bash"

  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux new-window -t "${SESSION_NAME}" -n "${WINDOW_NAME}" "${TRAIN_CMD}"
    tmux attach-session -t "${SESSION_NAME}"
  else
    tmux new-session -s "${SESSION_NAME}" -n "${WINDOW_NAME}" "${TRAIN_CMD}"
  fi
  exit 0
fi

cd /mnt/data/hyn/agilex

MODEL_ROOT="/mnt/data/hyn/model"
CONFIG_NAME="pi05_agilex_bimanual_arrange_flower_finetune"
EXP_NAME="arrange_flower_0625_0626_bimanual_4gpu"

mkdir -p \
  "${MODEL_ROOT}/openpi_cache" \
  "${MODEL_ROOT}/agilex_assets" \
  "${MODEL_ROOT}/agilex_checkpoints"

export OPENPI_DATA_HOME="${MODEL_ROOT}"
export WANDB_MODE=online
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if [[ -z "${WANDB_API_KEY}" || "${WANDB_API_KEY}" == "PASTE_YOUR_WANDB_API_KEY_HERE" ]]; then
  echo "Please set WANDB_API_KEY before running, or replace it in this script."
  exit 1
fi

echo "[1/2] Computing normalization stats for ${CONFIG_NAME}"
uv run scripts/compute_agilex_bimanual_arrange_flower_norm_stats.py

echo "[2/2] Starting 4-GPU training: ${EXP_NAME}"
uv run scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --fsdp-devices=4 \
  --overwrite
