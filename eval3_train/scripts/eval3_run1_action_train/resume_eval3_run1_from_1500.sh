#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source .venv/bin/activate

export HF_HOME="${HF_HOME:-/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf-home/hub}"
export EVAL3_VLM_LORA_PATH="${EVAL3_VLM_LORA_PATH:-${ROOT_DIR}/checkpoint/eval3_phase2_TOY}"
export EVAL3_EXPECTED_VLM_BASE="${EVAL3_EXPECTED_VLM_BASE:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
export WANDB_PROJECT="robot-learning-group47-eval3"
export WANDB_NAME="eval3_run1_action_expert_frozen_vlm_lora_lr3e5"

CONFIG_PATH="outputs/train/eval3_run1_action_expert_frozen_vlm_lora/checkpoints/001500/pretrained_model/train_config.json"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "ERROR: resume config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

python scripts/eval3_run1_action_train/run1_train_launcher.py \
  --resume=true \
  --config_path="${CONFIG_PATH}"
