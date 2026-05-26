#!/usr/bin/env bash
# ============================================================
# Stage 1 — Position-Only Motor Bootstrap
#
# Goal: teach the action expert to pick banana and place it
#       into left / middle / right bowl, using the strict10
#       VQA backbone as a frozen language encoder.
#
# Design:
#   Backbone:  robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged
#              FULLY FROZEN (train_expert_only=true + freeze_vision_encoder=true)
#   Expert:    randomly initialised (load_vlm_weights=true → VLM loaded; expert random)
#   Dataset:   robot-learning-group47/eval2_108_position_only
#              (108 episodes, 3 prompts: left / middle / right)
#   LR:        1e-4 peak  →  cosine decay  →  2.5e-6 (500 warmup / 8000 decay)
#   Steps:     8 000
#   Save:      every 1 000 steps
#
# Success gate (run manually after training):
#   1. Grabs banana
#   2. Finishes within 20 s
#   3. Final placement direction changes correctly for left / middle / right prompts
#
# Usage:
#   On GPU server:
#     tmux new -s stage1
#     bash ~/run_stage1_position_bootstrap.sh 2>&1 | tee ~/outputs/stage1.log
# ============================================================
set -euo pipefail

VLM="robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
DATASET="robot-learning-group47/eval2_108_position_only"
RUN_NAME="eval2-stage1-position-bootstrap"
OUT="/home/shadeform/outputs/eval2_stage1_position_bootstrap"
DATASET_ROOT="/home/shadeform/lerobot_cache/${DATASET}"
LOG="${OUT}.log"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"

mkdir -p "${DATASET_ROOT}" "$(dirname "${LOG}")"

echo "============================================================"
echo "  Stage 1 — Position Bootstrap"
echo "  VLM:      ${VLM} (FROZEN)"
echo "  Dataset:  ${DATASET}"
echo "  Output:   ${OUT}"
echo "  Steps:    15 000  |  LR peak: 1e-4  |  warmup: 1 000"
echo "============================================================"
echo ""

/home/shadeform/lerobot/.venv/bin/lerobot-train \
  \
  --policy.type=smolvla \
  --policy.vlm_model_name="${VLM}" \
  --policy.load_vlm_weights=true \
  \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.num_steps=10 \
  --policy.use_cache=true \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras=2 \
  --policy.tokenizer_max_length=48 \
  --policy.num_vlm_layers=16 \
  --policy.self_attn_every_n_layers=2 \
  --policy.expert_width_multiplier=0.75 \
  --policy.attention_mode=cross_attn \
  --policy.prefix_length=-1 \
  --policy.pad_language_to=longest \
  --policy.min_period=0.004 \
  --policy.max_period=4.0 \
  --policy.add_image_special_tokens=false \
  \
  --policy.push_to_hub=true \
  --policy.repo_id="robot-learning-group47/smolvla-eval2-stage1-position-bootstrap" \
  \
  --dataset.repo_id="${DATASET}" \
  --dataset.revision=main \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}' \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=torchcodec \
  \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  \
  --batch_size=32 \
  --num_workers=12 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  \
  --use_policy_training_preset=false \
  \
  --steps=15000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=15000 \
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
  --scheduler.num_decay_steps=15000 \
  --scheduler.peak_lr=1e-4 \
  --scheduler.decay_lr=2.5e-6 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}"

echo ""
echo "[$(date)] Stage 1 training complete."
echo "Best checkpoint: ${OUT}/checkpoints/last/pretrained_model"
