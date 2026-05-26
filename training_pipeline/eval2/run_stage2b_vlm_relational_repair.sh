#!/usr/bin/env bash
set -euo pipefail

INIT_CKPT=/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model
DATASET=robot-learning-group47/eval2_compositional_augmented
DATASET_ROOT=/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented
OUT_DIR=/home/shadeform/outputs/eval2_stage2b_vlm_relational_repair_from_20k

export HF_HOME=/home/shadeform/hf_home
export HF_HUB_CACHE=/home/shadeform/hf_home/hub
export HF_LEROBOT_HOME=/home/shadeform/lerobot_cache

PYTHONPATH=/home/shadeform/lerobot/src:/home/shadeform \
/home/shadeform/lerobot/.venv/bin/python /home/shadeform/train_eval2_relational_vlm_repair.py \
  --init="${INIT_CKPT}" \
  --dataset="${DATASET}" \
  --dataset-root="${DATASET_ROOT}" \
  --out-dir="${OUT_DIR}" \
  --steps=12000 \
  --batch-size=30 \
  --lr=1e-4 \
  --lambda-slot=0.5 \
  --save-every=2000 \
  --log-every=25 \
  --lora-rank=16 \
  "$@"
