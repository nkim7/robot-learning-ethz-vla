#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source .venv/bin/activate

export HF_HOME="${HF_HOME:-/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf-home/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export EVAL3_VLM_LORA_PATH="${EVAL3_VLM_LORA_PATH:-${ROOT_DIR}/checkpoint/eval3_phase2_TOY}"
export EVAL3_EXPECTED_VLM_BASE="${EVAL3_EXPECTED_VLM_BASE:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
export WANDB_PROJECT="robot-learning-group47-eval3"
export WANDB_NAME="eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k_lr3e5"

OUTPUT_DIR="outputs/train/eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k"

if [ -d "${OUTPUT_DIR}" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists. Refusing to overwrite Run 2 outputs." >&2
  echo "Move/delete it explicitly or run with a different output_dir." >&2
  exit 1
fi

python scripts/eval3_run2_lmr210_action_train/sanity_check_run2.py
python scripts/eval3_run2_lmr210_action_train/check_trainable_params_run2.py

python scripts/eval3_run2_lmr210_action_train/run2_train_launcher.py \
  --dataset.repo_id=robot-learning-group47/eval3_90_plus_lmr_210 \
  --dataset.root=data/eval3_90_plus_lmr_210 \
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
  --steps=30000 \
  --save_freq=2000 \
  --eval_freq=0 \
  --log_freq=50 \
  --num_workers=4 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${WANDB_NAME}" \
  --wandb.enable=true \
  --wandb.project="${WANDB_PROJECT}" \
  --policy.push_to_hub=false \
  --policy.repo_id=robot-learning-group47/eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_smolvla
