#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
PYTHONPATH_VAL="/home/shadeform"
TRAIN_SCRIPT="/home/shadeform/train_eval2_relational_clean_balanced.py"
GATE_SCRIPT="/home/shadeform/gate_eval2_relational_clean_balanced.py"

DATASET="robot-learning-group47/eval2_108_position_only"
DATASET_ROOT="/home/shadeform/lerobot_cache/${DATASET}"
SPLIT="/home/shadeform/eval3_repair_work/eval2_clean_fixed_frame_split.json"

STAGE2_INIT="/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
STAGE1_INIT="/home/shadeform/outputs/eval2_stage1_position_bootstrap/checkpoints/015000/pretrained_model"

OUT_A="/home/shadeform/outputs/eval2_stage2_relational_clean_60_20_15_5_fixedframe_balanced_from_stage2_20k"
OUT_B="/home/shadeform/outputs/eval2_stage2_relational_clean_60_20_15_5_fixedframe_balanced_from_stage1_15k"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"

run_one() {
  local NAME="$1"
  local INIT="$2"
  local OUT="$3"
  local TRAIN_LOG="${OUT}.log"
  local GATE_LOG="${OUT}_gates.log"

  mkdir -p "${OUT}"
  echo "============================================================" | tee -a "${TRAIN_LOG}"
  echo "Run ${NAME}" | tee -a "${TRAIN_LOG}"
  echo "Init: ${INIT}" | tee -a "${TRAIN_LOG}"
  echo "Out:  ${OUT}" | tee -a "${TRAIN_LOG}"
  echo "============================================================" | tee -a "${TRAIN_LOG}"

  PYTHONPATH="${PYTHONPATH_VAL}" "${PYTHON}" "${TRAIN_SCRIPT}" \
    --init="${INIT}" \
    --dataset="${DATASET}" \
    --dataset-root="${DATASET_ROOT}" \
    --out-dir="${OUT}" \
    --split-path="${SPLIT}" \
    --steps=5000 \
    --batch-size=36 \
    --lr=5e-5 \
    --lora-rank=16 \
    --save-steps=500,1000,1500,2000,3000,5000 \
    --log-every=25 \
    --wandb \
    --wandb-project=lerobot \
    --wandb-name="${NAME}" \
    2>&1 | tee -a "${TRAIN_LOG}"

  for STEP in 500 1000 1500 2000 3000 5000; do
    local CKPT="${OUT}/checkpoints/$(printf '%06d' "${STEP}")/pretrained_model"
    if [[ ! -d "${CKPT}" ]]; then
      echo "Missing ${CKPT}, skipping gate" | tee -a "${GATE_LOG}"
      continue
    fi
    echo "" | tee -a "${GATE_LOG}"
    echo "============================================================" | tee -a "${GATE_LOG}"
    echo "${NAME} step ${STEP}: expanded held-out gate" | tee -a "${GATE_LOG}"
    echo "============================================================" | tee -a "${GATE_LOG}"
    PYTHONPATH="${PYTHONPATH_VAL}" "${PYTHON}" "${GATE_SCRIPT}" \
      --ckpt="${CKPT}" \
      --base-policy="${INIT}" \
      --dataset="${DATASET}" \
      --dataset-root="${DATASET_ROOT}" \
      --split-path="${SPLIT}" \
      --n-gate-frames=5 \
      --threshold=0.95 \
      2>&1 | tee "${OUT}/gate_$(printf '%06d' "${STEP}").txt" | tee -a "${GATE_LOG}"
  done
}

run_one "eval2-rel-clean-from-stage2-20k" "${STAGE2_INIT}" "${OUT_A}"
run_one "eval2-rel-clean-from-stage1-15k" "${STAGE1_INIT}" "${OUT_B}"
