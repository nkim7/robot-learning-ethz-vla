#!/usr/bin/env bash
# ============================================================
# Frozen VQA Expert Adaptation — clean baseline
#
# Design:
#   - Init:      merged checkpoint (VQA backbone + Eval1 expert)
#   - Backbone:  fully frozen  (train_expert_only=true)
#   - Text LM:   frozen        (included in train_expert_only)
#   - Vision:    frozen        (freeze_vision_encoder=true, belt+suspenders)
#   - Expert:    Eval1 warm-start, trainable at 5e-6
#   - Augment:   brightness / contrast / sharpness ONLY (no hue / saturation)
#
# Goal: test whether frozen VQA hidden states are already
#       usable by the action expert, avoiding any forgetting.
#
# Probe plan (at 2k checkpoints):
#   1. Color first — if this fails, stop and revisit
#   2. Ordinal, Relative, Negation
#   3. Same-scene-different-prompt divergence
#
# Usage (on GPU server):
#   bash run_train_frozen_vqa_expert.sh
#   INIT_CHECKPOINT=/custom/path bash run_train_frozen_vqa_expert.sh
# ============================================================
set -euo pipefail

INIT_CHECKPOINT="${INIT_CHECKPOINT:-/data/lerobot_outputs/train/smolvla-eval2-pathC-init}"
DATASET="${DATASET:-robot-learning-group47/eval2_compositional_augmented}"
RUN_NAME="${RUN_NAME:-smolvla-eval2-frozen-vqa-expert}"
OUT="${OUT:-/data/lerobot_outputs/train/${RUN_NAME}}"
LOG="${LOG:-/data/lerobot_outputs/${RUN_NAME}.log}"
DATASET_ROOT="${DATASET_ROOT:-/data/lerobot_cache/${DATASET}}"

# Same rename map as Path C — dataset stores 'front', policy expects 'camera1'
RENAME_MAP='{"observation.images.front":"observation.images.camera1"}'
EMPTY_CAMERAS=2

export HF_HOME="${HF_HOME:-/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_home/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot_cache}"

if [[ ! -f "${INIT_CHECKPOINT}/model.safetensors" ]]; then
  echo "ERROR: Init checkpoint not found at ${INIT_CHECKPOINT}"
  echo "       This should be the same merged VQA+Eval1 checkpoint used for Path C."
  echo "       Run: python merge_pathC.py  (or copy from Path C init)"
  exit 1
fi

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DATASET_ROOT}"

echo "============================================================"
echo "  Frozen VQA Expert Adaptation"
echo "  Init:       ${INIT_CHECKPOINT}"
echo "  Dataset:    ${DATASET}"
echo "  Output:     ${OUT}"
echo "  Log:        ${LOG}"
echo "  Backbone:   FROZEN (train_expert_only=true)"
echo "  Expert LR:  5e-6"
echo "  Steps:      40 000  (~7.9 epochs)"
echo "============================================================"

lerobot-train \
  --policy.path="${INIT_CHECKPOINT}" \
  \
  --policy.device=cuda \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --policy.push_to_hub=true \
  --policy.repo_id="robot-learning-group47/${RUN_NAME}" \
  \
  --dataset.repo_id="${DATASET}" \
  --dataset.revision=main \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}' \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=torchcodec \
  --rename_map="${RENAME_MAP}" \
  \
  --batch_size=32 \
  --num_workers=12 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  \
  --steps=40000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=40000 \
  \
  --optimizer.type=adamw \
  --optimizer.lr=5e-6 \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10 \
  --optimizer.betas='[0.9,0.95]' \
  --optimizer.eps=1e-8 \
  \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=40000 \
  --scheduler.peak_lr=5e-6 \
  --scheduler.decay_lr=5e-8 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}" \
  2>&1 | tee -a "${LOG}"

echo ""
echo "Training complete."
echo "Probe checkpoints at: ${OUT}/checkpoints/"
echo ""
echo "Color check (run at step 2000 first):"
echo "  python probe_vqa_backbone.py \\"
echo "    --model ${OUT}/checkpoints/002000/pretrained_model \\"
echo "    --per_perm 2 --verbose"
