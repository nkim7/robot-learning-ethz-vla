#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/home/shadeform/hf_home
export HF_HUB_CACHE=/home/shadeform/hf_home/hub
export HF_LEROBOT_HOME=/home/shadeform/lerobot_cache
export PYTHONPATH=/home/shadeform/lerobot/src
export TOKENIZERS_PARALLELISM=false

PY=/home/shadeform/lerobot/.venv/bin/python
DATA_ROOT=/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented
VQA_DIR=/data/branchphase_vqa
OUT_DIR=/data/lerobot_outputs/smolvlm2-500m-vqa-branchphase-color-slot

cd /home/shadeform

echo "[$(date)] Building branch-phase VQA dataset..."
if [ ! -f "${VQA_DIR}/branchphase_vqa.jsonl" ]; then
  "${PY}" build_branchphase_vqa_dataset.py \
    --root "${DATA_ROOT}" \
    --out-dir "${VQA_DIR}" \
    --n-real-demos 108
else
  echo "[$(date)] Reusing existing ${VQA_DIR}/branchphase_vqa.jsonl"
fi

echo "[$(date)] Starting branch-phase SmolVLM2 LoRA training..."
"${PY}" train_branchphase_vqa_lora.py \
  --jsonl "${VQA_DIR}/branchphase_vqa.jsonl" \
  --base-model robot-learning-group47/smolvlm2-500m-vqa-lora-merged \
  --output-dir "${OUT_DIR}" \
  --epochs 8 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-4 \
  --lora-rank 16 \
  --eval-every 50 \
  --log-every 1 \
  --target-acc 0.997 \
  --wandb \
  --wandb-project smolvlm2-branchphase-vqa \
  --wandb-name smolvlm2-500m-vqa-branchphase-color-slot

echo "[$(date)] Done."
