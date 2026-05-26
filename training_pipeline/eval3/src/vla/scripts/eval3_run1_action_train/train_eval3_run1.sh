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

OUTPUT_DIR="outputs/train/eval3_run1_action_expert_frozen_vlm_lora"

if [ -d "${OUTPUT_DIR}" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists. Refusing to overwrite Run 1 outputs." >&2
  echo "Move/delete it explicitly or run with a different output_dir." >&2
  exit 1
fi

python scripts/eval3_run1_action_train/sanity_check_run1.py
python scripts/eval3_run1_action_train/check_trainable_params.py

python scripts/eval3_run1_action_train/run1_train_launcher.py \
  --dataset.repo_id=robot-learning-group47/eval3_90_permutation \
  --dataset.root=data/eval3_90_permutation \
  --dataset.video_backend=pyav \
  --policy.type=smolvla \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=-1 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.use_peft=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.n_obs_steps=1 \
  --policy.tokenizer_max_length=48 \
  --policy.empty_cameras=2 \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=3e-5 \
  --batch_size=16 \
  --steps=8000 \
  --save_freq=500 \
  --eval_freq=0 \
  --log_freq=50 \
  --num_workers=4 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${WANDB_NAME}" \
  --wandb.enable=true \
  --wandb.project="${WANDB_PROJECT}" \
  --policy.push_to_hub=true \
  --policy.repo_id=robot-learning-group47/eval3_run1_action_expert_frozen_vlm_lora_smolvla
