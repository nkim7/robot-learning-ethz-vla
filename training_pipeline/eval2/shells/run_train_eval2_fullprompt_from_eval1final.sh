#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-robot-learning-group47/eval2_180_full_promptaug_clean256_v1}"
DATASET_ROOT="${DATASET_ROOT:-/data/lerobot_cache/${DATASET}}"
BASE_POLICY="${BASE_POLICY:-robot-learning-group47/eval1_clean_256_chunk50_lightingonly_smolvla_v1}"
RUN_NAME="${RUN_NAME:-eval2_fullprompt_from_eval1final_smolvla_v1}"
OUT="${OUT:-/data/lerobot_outputs/train/${RUN_NAME}}"
LOG="${LOG:-/data/lerobot_outputs/${RUN_NAME}.log}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

RENAME_MAP="${RENAME_MAP:-{\"observation.images.front\":\"observation.images.camera1\"}}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-2}"
RENAME_ARGS=()
if [[ -n "${RENAME_MAP}" ]]; then
  RENAME_ARGS=(--rename_map="${RENAME_MAP}")
fi

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DATASET_ROOT}"

export HF_HOME="${HF_HOME:-/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_home/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot_cache}"

echo "============================================================"
echo "  Eval 2 Track 1: full-prompt from Eval1 final"
echo "  Dataset:     ${DATASET}"
echo "  Dataset root:${DATASET_ROOT}"
echo "  Base policy: ${BASE_POLICY}"
echo "  Output:      ${OUT}"
echo "  Push to hub: ${PUSH_TO_HUB}"
echo "============================================================"

lerobot-train \
  --dataset.repo_id="${DATASET}" \
  --dataset.revision=main \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=torchcodec \
  \
  --policy.path="${BASE_POLICY}" \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --policy.repo_id="robot-learning-group47/${RUN_NAME}" \
  \
  "${RENAME_ARGS[@]}" \
  \
  --batch_size=64 \
  --num_workers=12 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --use_policy_training_preset=false \
  \
  --steps=6000 \
  --save_freq=500 \
  --log_freq=100 \
  --eval_freq=6000 \
  \
  --optimizer.type=adamw \
  --optimizer.lr=2e-5 \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10 \
  --optimizer.betas='[0.9,0.95]' \
  --optimizer.eps=1e-8 \
  \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=6000 \
  --scheduler.peak_lr=2e-5 \
  --scheduler.decay_lr=2.5e-6 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}" \
  2>&1 | tee -a "${LOG}"
