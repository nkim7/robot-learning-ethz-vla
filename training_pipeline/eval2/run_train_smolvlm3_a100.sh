#!/usr/bin/env bash
# Train smolvlm3: smolvlm2 + position-output VQA questions.
#
# Identical pipeline to smolvlm2:
#   1. Build VQA v3 dataset (v2 questions + new color_to_pos questions)
#   2. Fine-tune LoRA r=32 on smolvlm2-500m-vqa-lora-merged
#   3. Merge LoRA into base, push to HF as smolvlm3-500m-vqa-lora-merged
#
# Run on A100: bash run_train_smolvlm3_a100.sh
set -euo pipefail

PHOTOS_ROOT="/home/shadeform/augmented/augmented"
VQA_DIR="/home/shadeform/data/eval2_vqa_v3"
VQA_JSONL="${VQA_DIR}/vqa.jsonl"
OUT_DIR="/home/shadeform/models/smolvlm3_vqa_lora"
LOG="/home/shadeform/smolvlm3_vqa_train.log"

PY="/usr/bin/python3"
REPO_ROOT="/home/shadeform"

echo "============================================================"
echo "  smolvlm3: VQA v3 training"
echo "  Photos:  ${PHOTOS_ROOT}"
echo "  VQA out: ${VQA_JSONL}"
echo "  Model:   ${OUT_DIR}"
echo "============================================================"

cd "${REPO_ROOT}"

# ── Step 1: Build VQA v3 dataset ─────────────────────────────────────────────
if [ ! -f "${VQA_JSONL}" ]; then
  echo "[$(date)] Building VQA v3 dataset..."
  "${PY}" scripts/create_eval2_vqa_v3_dataset.py \
    --photos-root "${PHOTOS_ROOT}" \
    --out "${VQA_JSONL}"
else
  echo "[$(date)] Reusing existing ${VQA_JSONL}"
fi

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo "[$(date)] Starting smolvlm3 LoRA training..."
"${PY}" scripts/train_eval2_smolvlm_vqa_v3_lora.py \
  --data          "${VQA_JSONL}" \
  --model         "robot-learning-group47/smolvlm2-500m-vqa-lora-merged" \
  --output-dir    "${OUT_DIR}" \
  --epochs        12 \
  --lr            2e-4 \
  --grad-accum    8 \
  --lora-r        32 \
  --lora-alpha    64 \
  --target-accuracy 0.995 \
  --wandb-enable \
  --wandb-project lerobot \
  --wandb-run-name smolvlm3-vqa-v3-r32 \
  2>&1 | tee -a "${LOG}"

# ── Step 3: Merge and push ────────────────────────────────────────────────────
echo "[$(date)] Merging LoRA and pushing smolvlm3 to HF..."
"${PY}" merge_lora_v3_and_push.py \
  2>&1 | tee -a "${LOG}"

echo "[$(date)] Done. smolvlm3 live at:"
echo "  https://huggingface.co/robot-learning-group47/smolvlm3-500m-vqa-lora-merged"
