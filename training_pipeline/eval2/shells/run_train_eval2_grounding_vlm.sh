#!/usr/bin/env bash
# Train Eval2-Specific Grounding VLM backbone.
# Runs on the shadeform GPU instance.
#
# Usage:
#   bash run_train_eval2_grounding_vlm.sh
#
# Override any var on the command line:
#   MODEL=robot-learning-group47/smolvlm2-500m-vqa-lora-merged \
#   bash run_train_eval2_grounding_vlm.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
# Prefer the already-VQA-tuned merged checkpoint if available:
#   MODEL=robot-learning-group47/smolvlm2-500m-vqa-lora-merged

DATA="${DATA:-data/eval2_grounding_vqa/vqa.jsonl}"
IMAGES_ROOT="${IMAGES_ROOT:-data/eval2_grounding_vqa}"
RUN_NAME="${RUN_NAME:-eval2_grounding_vlm_lora_r16}"
OUT="${OUT:-/data/resolver_outputs/${RUN_NAME}}"
LOG="${LOG:-/data/resolver_outputs/${RUN_NAME}.log}"

EPOCHS="${EPOCHS:-20}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GROUP_SIZE="${GROUP_SIZE:-6}"

export HF_HOME="${HF_HOME:-/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_home/hub}"

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

echo "============================================================"
echo "  Eval2 Grounding VLM — LoRA fine-tune"
echo "  Base model : ${MODEL}"
echo "  Data       : ${DATA}"
echo "  Run name   : ${RUN_NAME}"
echo "  Output     : ${OUT}"
echo "  Epochs     : ${EPOCHS}  LR: ${LR}  LoRA r=${LORA_R}"
echo "============================================================"

# ── Build dataset if needed ───────────────────────────────────────────────────
if [[ ! -f "${DATA}" ]]; then
  echo "[dataset] vqa.jsonl not found — running build_eval2_grounding_dataset.py ..."
  python3 build_eval2_grounding_dataset.py \
    --images-dir permutation_init_position \
    --out-dir "$(dirname "${DATA}")"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
python3 train_eval2_grounding_vlm.py \
  --model        "${MODEL}" \
  --data         "${DATA}" \
  --images-root  "${IMAGES_ROOT}" \
  --output-dir   "${OUT}" \
  --epochs       "${EPOCHS}" \
  --lr           "${LR}" \
  --lora-r       "${LORA_R}" \
  --lora-alpha   "${LORA_ALPHA}" \
  --grad-accum   "${GRAD_ACCUM}" \
  --group-size   "${GROUP_SIZE}" \
  --wandb-enable \
  --wandb-project lerobot \
  --wandb-run-name "${RUN_NAME}" \
  2>&1 | tee -a "${LOG}"
