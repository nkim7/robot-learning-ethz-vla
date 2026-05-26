#!/usr/bin/env bash
# Path C: VQA backbone (smolvlm2-500m-vqa-lora-merged) +
#         eval1 expert warm-start (eval1_final_final)
#
# Prerequisites:
#   1. Run scripts/merge_vqa_backbone_into_eval1.py --dry-run  → must pass all gates
#   2. Run scripts/merge_vqa_backbone_into_eval1.py            → produces INIT_CHECKPOINT
#   3. Run smoke test (200 steps, see below) and confirm gradients + loss baseline
#
# Smoke test (run this FIRST, ~5 min):
#   lerobot-train \
#     --policy.path="${INIT_CHECKPOINT}" \
#     --policy.train_expert_only=false \
#     --policy.freeze_vision_encoder=true \
#     --dataset.repo_id="${DATASET}" \
#     --dataset.root="${DATASET_ROOT}" \
#     --rename_map="${RENAME_MAP}" \
#     --steps=200 --batch_size=8 \
#     --output_dir=/tmp/pathC_smoke \
#     --wandb.enable=false
#
# After smoke test:
#   uv run python scripts/check_gradients_pathC.py \
#     --checkpoint /tmp/pathC_smoke/checkpoints/last/pretrained_model
#
set -euo pipefail

INIT_CHECKPOINT="${INIT_CHECKPOINT:-/data/lerobot_outputs/train/smolvla-eval2-pathC-init}"
DATASET="${DATASET:-robot-learning-group47/eval2_compositional_augmented}"
RUN_NAME="${RUN_NAME:-smolvla-eval2-pathC}"
OUT="${OUT:-/data/lerobot_outputs/train/${RUN_NAME}}"
LOG="${LOG:-/data/lerobot_outputs/${RUN_NAME}.log}"
DATASET_ROOT="${DATASET_ROOT:-/data/lerobot_cache/${DATASET}}"
RENAME_MAP='{"observation.images.front":"observation.images.camera1"}'
EMPTY_CAMERAS=2

export HF_HOME="${HF_HOME:-/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_home/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot_cache}"

if [[ ! -f "${INIT_CHECKPOINT}/model.safetensors" ]]; then
  echo "ERROR: Merged checkpoint not found at ${INIT_CHECKPOINT}"
  echo "       Run: uv run python scripts/merge_vqa_backbone_into_eval1.py"
  exit 1
fi

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DATASET_ROOT}"

echo "============================================================"
echo "  Path C: VQA backbone + eval1 expert warm-start"
echo "  Init:    ${INIT_CHECKPOINT}"
echo "  Dataset: ${DATASET}"
echo "  Output:  ${OUT}"
echo "  Log:     ${LOG}"
echo "============================================================"

lerobot-train \
  --policy.path="${INIT_CHECKPOINT}" \
  \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
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
  --steps=10000 \
  --save_freq=1000 \
  --log_freq=100 \
  --eval_freq=10000 \
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
  --scheduler.num_decay_steps=15000 \
  --scheduler.peak_lr=2e-5 \
  --scheduler.decay_lr=2.5e-7 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}" \
  2>&1 | tee -a "${LOG}"

echo ""
echo "Training complete. Run compositional probe at step 5000 checkpoint:"
echo "  uv run python run_eval2_resolver_fullrollout.py \\"
echo "    --policy.path=${OUT}/checkpoints/005000/pretrained_model \\"
echo "    --prompt_types=ordinal_right,relative,negation"
