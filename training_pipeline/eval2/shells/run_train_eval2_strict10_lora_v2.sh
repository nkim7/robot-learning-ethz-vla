#!/usr/bin/env bash
# ============================================================
# Revised Run 2 — eval2_strict10_lora_expert_full_v2
#
# Purpose:
#   Frozen-backbone Run 2 collapsed (color pan <1°, ordinal <1°).
#   This run opens the VLM text q/v layers with LoRA r=16 so the
#   backbone and expert can co-adapt under ordinary BC.
#
# Design:
#   Backbone:   strict10 VQA backbone (base weights FROZEN)
#               text q/v LoRA r=16 (TRAINABLE)
#   Expert:     eval1_final_final (warm-start, TRAINABLE)
#   Projections: state/action/time_mlp (TRAINABLE)
#   Vision enc: FROZEN
#   Loss:       L = L_action  (no aux target-slot loss)
#   LR:         5e-6 peak → 5e-8 decay
#   Dataset:    eval2_fixed_bowl_fixed_banana_108
#
# Diagnostic protocol:
#   Probe at every 2k checkpoint (2k/4k/6k/8k/10k).
#   DO NOT continue blindly to 40k.
#   Real success = clearly different action chunks per prompt.
#
#   Baselines (frozen Run 2 — collapsed):
#     2k: color pan ~0.60°, ordinal ~0.38°
#     4k: color pan ~0.77°, ordinal ~0.07°
#     6k: color pan ~0.46°, ordinal ~0.67°
#
#   Thresholds:
#     0–2°:  collapsed (no better than frozen)
#     3–6°:  weak signal
#     8–12°: worth watching
#     15°+:  robot-test candidate (only if direction correct)
#
# Usage:
#   bash run_train_eval2_strict10_lora_v2.sh
# ============================================================
set -euo pipefail

INIT_CHECKPOINT="${INIT_CHECKPOINT:-/data/lerobot_outputs/train/smolvla-eval2-strict10frozenbackbone-init_migrated}"
DATASET="${DATASET:-robot-learning-group47/eval2_fixed_bowl_fixed_banana_108}"
RUN_NAME="${RUN_NAME:-eval2_strict10_lora_expert_full_v2}"
OUT="${OUT:-/data/lerobot_outputs/train/${RUN_NAME}}"
LOG="${LOG:-/data/lerobot_outputs/${RUN_NAME}.log}"
DATASET_ROOT="${DATASET_ROOT:-/data/lerobot_cache/${DATASET}}"

RENAME_MAP='{"observation.images.front":"observation.images.camera1"}'
EMPTY_CAMERAS=2

export HF_HOME="${HF_HOME:-/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_home/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot_cache}"

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DATASET_ROOT}"

if [[ ! -f "${INIT_CHECKPOINT}/model.safetensors" ]]; then
  echo "ERROR: Init checkpoint not found at ${INIT_CHECKPOINT}"
  echo "       Run merge_strict10_into_eval1.py first."
  exit 1
fi

echo "============================================================"
echo "  Revised Run 2: eval2_strict10_lora_expert_full_v2"
echo "  Init:       ${INIT_CHECKPOINT}"
echo "  Dataset:    ${DATASET}"
echo "  Output:     ${OUT}"
echo "  Log:        ${LOG}"
echo "  Backbone:   strict10 base FROZEN + text q/v LoRA r=16"
echo "  Expert:     eval1_final_final (warm-start, trainable)"
echo "  LR:         5e-6 peak → 5e-8 decay"
echo "  No aux loss. No grouped sampler. BC only."
echo "  PROBE at every 2k. Stop early if still collapsed at 6k."
echo "============================================================"

/home/shadeform/lerobot/.venv/bin/lerobot-train \
  --policy.path="${INIT_CHECKPOINT}" \
  \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --policy.push_to_hub=false \
  \
  --peft.method_type=lora \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj' \
  --peft.full_training_modules='["lm_expert","state_proj","action_in_proj","action_out_proj","action_time_mlp_in","action_time_mlp_out"]' \
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
  --use_policy_training_preset=false \
  \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=10000 \
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
  --scheduler.num_decay_steps=10000 \
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
echo "[$(date)] Training complete (or stopped at 10k diagnostic)."
echo ""
echo "  Probe all checkpoints:"
echo "  for STEP in 002000 004000 006000 008000 010000; do"
echo "    echo '--- Step' \${STEP} '---'"
echo "    .venv/bin/python probe_action_conditioning.py \\"
echo "      --model ${OUT}/checkpoints/\${STEP}/pretrained_model"
echo "  done"
