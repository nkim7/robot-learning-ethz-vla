#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-robot-learning-group47/eval1_position}"
DATASET_ROOT="${DATASET_ROOT:-/home/shadeform/lerobot_cache/robot-learning-group47/eval1_position}"
BASE_POLICY="${BASE_POLICY:-lerobot/smolvla_base}"
RUN_NAME="${RUN_NAME:-eval1_position}"
OUT="${OUT:-/home/shadeform/outputs/${RUN_NAME}}"
LOG="${LOG:-/home/shadeform/${RUN_NAME}.log}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LR="${LR:-2e-5}"
RESUME="${RESUME:-false}"

RENAME_MAP='{"observation.images.front":"observation.images.camera1"}'
EMPTY_CAMERAS=2

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DATASET_ROOT}"

export HF_HOME="${HF_HOME:-/home/shadeform/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/shadeform/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/shadeform/hf_home/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/shadeform/lerobot_cache}"

echo "============================================================"
echo "  Eval 1 retrain: fixed banana + fixed bowl, position labels"
echo "  Dataset:     ${DATASET}"
echo "  DatasetRoot: ${DATASET_ROOT}"
echo "  Base policy: ${BASE_POLICY}"
echo "  Run:         ${RUN_NAME}"
echo "  Steps:       ${STEPS}"
echo "  Output:      ${OUT}"
echo "  Push to hub: ${PUSH_TO_HUB}"
echo "  WandB:       enabled, project=lerobot"
echo "============================================================"

/home/shadeform/lerobot/.venv/bin/lerobot-train \
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
  --rename_map="${RENAME_MAP}" \
  \
  --batch_size=64 \
  --num_workers=12 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --use_policy_training_preset=false \
  \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=100 \
  --eval_freq="${STEPS}" \
  \
  --optimizer.type=adamw \
  --optimizer.lr="${LR}" \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10 \
  --optimizer.betas='[0.9,0.95]' \
  --optimizer.eps=1e-8 \
  \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps="${STEPS}" \
  --scheduler.peak_lr="${LR}" \
  --scheduler.decay_lr=2.5e-6 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}" \
  --resume="${RESUME}" \
  2>&1 | tee -a "${LOG}"
