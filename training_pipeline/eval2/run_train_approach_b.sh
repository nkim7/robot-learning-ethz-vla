#!/usr/bin/env bash
# Approach B: smolvlm2-500m-vqa-lora-merged (frozen VLM) + train action expert
# Dataset: eval2_compositional_augmented — 108 original eps × ~7 prompts ≈ 756 virtual episodes.
#          Each episode is duplicated once per valid compositional prompt (ordinal, relative,
#          negation, direct color) so the model sees the same trajectory under every valid
#          language description.
# At inference: feed ANY eval-2-style prompt directly — no separate resolver needed.
# Output: robot-learning-group47/smolvla-eval2-approach-b
set -euo pipefail

DATASET="${DATASET:-robot-learning-group47/eval2_compositional_augmented}"
VLM_MODEL="${VLM_MODEL:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
PRETRAINED_POLICY="${PRETRAINED_POLICY:-robot-learning-group47/eval1_final_final}"
RUN_NAME="${RUN_NAME:-smolvla-eval2-approach-b}"
OUT="${OUT:-/data/lerobot_outputs/train/${RUN_NAME}}"
LOG="${LOG:-/data/lerobot_outputs/${RUN_NAME}.log}"
DATASET_ROOT="${DATASET_ROOT:-/data/lerobot_cache/${DATASET}}"

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
echo "  Approach B: VQA VLM + eval2_compositional_augmented action expert training"
echo "  Dataset:   ${DATASET}"
echo "  VLM:       ${VLM_MODEL}"
echo "  Output:    ${OUT}"
echo "  Log:       ${LOG}"
echo "============================================================"

lerobot-train \
  --dataset.repo_id="${DATASET}" \
  --dataset.revision=main \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}' \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=torchcodec \
  \
  --policy.type=smolvla \
  --policy.vlm_model_name="${VLM_MODEL}" \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --policy.push_to_hub=true \
  --policy.repo_id="robot-learning-group47/${RUN_NAME}" \
  \
  "${RENAME_ARGS[@]}" \
  \
  --batch_size=32 \
  --num_workers=12 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=200 \
  --eval_freq=20000 \
  \
  --optimizer.type=adamw \
  --optimizer.lr=1e-4 \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10 \
  --optimizer.betas='[0.9,0.95]' \
  --optimizer.eps=1e-8 \
  \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=30000 \
  --scheduler.peak_lr=1e-4 \
  --scheduler.decay_lr=2.5e-6 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}" \
  2>&1 | tee -a "${LOG}"
