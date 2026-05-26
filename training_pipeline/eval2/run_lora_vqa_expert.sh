#!/usr/bin/env bash
set -euo pipefail

INIT="/home/shadeform/outputs/smolvla-eval2-pathC-init"
DATASET="robot-learning-group47/eval2_compositional_augmented"
RUN_NAME="smolvla-eval2-lora-vqa-expert"
OUT="/home/shadeform/outputs/${RUN_NAME}"
DATASET_ROOT="/home/shadeform/lerobot_cache/${DATASET}"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"

mkdir -p "${DATASET_ROOT}"

/home/shadeform/lerobot/.venv/bin/lerobot-train \
  --policy.path="${INIT}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras=2 \
  --policy.push_to_hub=true \
  --policy.repo_id="robot-learning-group47/${RUN_NAME}" \
  --dataset.repo_id="${DATASET}" \
  --dataset.revision=main \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}' \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --batch_size=32 \
  --num_workers=12 \
  --steps=40000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=40000 \
  --optimizer.type=adamw \
  --optimizer.lr=5e-6 \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10 \
  --optimizer.betas='[0.9,0.95]' \
  --optimizer.eps=1e-8 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=40000 \
  --scheduler.peak_lr=5e-6 \
  --scheduler.decay_lr=5e-8 \
  --peft.method_type=lora \
  --peft.r=16 \
  --peft.target_modules='model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj' \
  --peft.full_training_modules='["lm_expert","state_proj","action_in_proj","action_out_proj","action_time_mlp_in","action_time_mlp_out"]' \
  --use_policy_training_preset=false \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}"
