#!/usr/bin/env bash
# E2E co-training with auxiliary VLM Q&A loss.
# Co-trains cycle 05 action expert + VLM LoRA jointly.
# Auxiliary cross-entropy loss on position Q&A preserves VLM accuracy.
# VLM LoRA LR is kept very low (1e-7) to prevent forgetting.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source .venv/bin/activate

export HF_HOME="${HF_HOME:-/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf-home/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export EVAL3_EXPECTED_VLM_BASE="${EVAL3_EXPECTED_VLM_BASE:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"

START_ACTION_CHECKPOINT="${START_ACTION_CHECKPOINT:-${ROOT_DIR}/outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_action/checkpoints/000500/pretrained_model}"
START_VLM_LORA="${START_VLM_LORA:-${ROOT_DIR}/outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_vlm_lora}"
ACTION_TOY_DATASET_ROOT="${ACTION_TOY_DATASET_ROOT:-${ROOT_DIR}/data/eval3_90_permutation}"
ACTION_LMR_DATASET_ROOT="${ACTION_LMR_DATASET_ROOT:-${ROOT_DIR}/data/eval3_lmr}"

TOY_REPEATS="${TOY_REPEATS:-4}"
LMR_REPEATS=0
STEPS="${STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACTION_LR="${ACTION_LR:-5e-6}"
VLM_LORA_LR="${VLM_LORA_LR:-1e-7}"          # Very low: preserve Q&A while adapting
AUX_VLM_LAMBDA="${AUX_VLM_LAMBDA:-0.5}"      # Weight of Q&A loss relative to action loss
AUX_VLM_BATCH_SIZE="${AUX_VLM_BATCH_SIZE:-2}" # Q&A examples per step (keep small for memory)

PREPARED_DATASET_ROOT="${PREPARED_DATASET_ROOT:-${ROOT_DIR}/data/eval3_e2e_person_only_toy4}"
ROBOT_DATASET_ROOT="${ROBOT_DATASET_ROOT:-${PREPARED_DATASET_ROOT}}"
ROBOT_DATASET_REPO_ID="${ROBOT_DATASET_REPO_ID:-robot-learning-group47/eval3_e2e_person_only_toy4}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/eval3_e2e_aux_vlm_cycle05_${STEPS}steps_vlmlr${VLM_LORA_LR}_lam${AUX_VLM_LAMBDA}}"

WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-group47-eval3}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_MODE="${WANDB_MODE:-online}"

if [ -d "${OUTPUT_DIR}" ] && [ "${E2E_OVERWRITE:-0}" != "1" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists." >&2
  echo "Set E2E_OVERWRITE=1 to replace it." >&2
  exit 1
fi

if [ ! -d "${START_ACTION_CHECKPOINT}" ]; then
  echo "ERROR: Missing start action checkpoint: ${START_ACTION_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -f "${START_VLM_LORA}/adapter_model.safetensors" ]; then
  echo "ERROR: Missing start VLM LoRA: ${START_VLM_LORA}" >&2
  exit 1
fi

echo "===== Preparing person-only dataset (toy x${TOY_REPEATS}, no LMR) ====="
PREPARE_ARGS=(
  --toy-root="${ACTION_TOY_DATASET_ROOT}"
  --lmr-root="${ACTION_LMR_DATASET_ROOT}"
  --output-root="${PREPARED_DATASET_ROOT}"
  --toy-repeats="${TOY_REPEATS}"
  --lmr-repeats="${LMR_REPEATS}"
)
if [ "${E2E_PREPARE_OVERWRITE:-0}" = "1" ]; then
  PREPARE_ARGS+=(--overwrite)
fi
python scripts/eval3_cotrain/prepare_action_dataset.py "${PREPARE_ARGS[@]}"

export EVAL3_VLM_LORA_PATH="${START_VLM_LORA}"
export EVAL3_VLM_LORA_TRAINABLE=1
export EVAL3_VLM_LORA_LR="${VLM_LORA_LR}"
export EVAL3_AUX_VLM_LAMBDA="${AUX_VLM_LAMBDA}"
export EVAL3_AUX_VLM_BATCH_SIZE="${AUX_VLM_BATCH_SIZE}"
export WANDB_NAME="eval3_e2e_aux_vlm_cycle05_${STEPS}steps_vlmlr${VLM_LORA_LR}_lam${AUX_VLM_LAMBDA}"

echo "===== E2E co-training with auxiliary VLM Q&A loss ====="
echo "Action checkpoint : ${START_ACTION_CHECKPOINT}"
echo "VLM LoRA          : ${START_VLM_LORA}"
echo "Dataset           : ${ROBOT_DATASET_ROOT}"
echo "Steps             : ${STEPS}  batch=${BATCH_SIZE}  save_freq=${SAVE_FREQ}"
echo "Action LR         : ${ACTION_LR}  VLM LoRA LR: ${VLM_LORA_LR}"
echo "Aux VLM lambda    : ${AUX_VLM_LAMBDA}  Q&A batch/step: ${AUX_VLM_BATCH_SIZE}"
echo "Output            : ${OUTPUT_DIR}"

python scripts/eval3_e2e_aux_vlm/run3_train_launcher.py \
  --dataset.repo_id="${ROBOT_DATASET_REPO_ID}" \
  --dataset.root="${ROBOT_DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.path="${START_ACTION_CHECKPOINT}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.use_peft=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr="${ACTION_LR}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq=0 \
  --log_freq=25 \
  --num_workers=4 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${WANDB_NAME}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.mode="${WANDB_MODE}" \
  --policy.push_to_hub=false \
  --policy.repo_id=robot-learning-group47/eval3_e2e_aux_vlm_cycle05_smolvla

FINAL_CKPT="${OUTPUT_DIR}/checkpoints/$(printf "%06d" "${STEPS}")/pretrained_model"
if [ ! -d "${FINAL_CKPT}" ]; then
  echo "ERROR: Expected checkpoint was not created: ${FINAL_CKPT}" >&2
  exit 1
fi

cat > "${OUTPUT_DIR}/e2e_aux_vlm_manifest.txt" <<MANIFEST
start_action_checkpoint=${START_ACTION_CHECKPOINT}
start_vlm_lora=${START_VLM_LORA}
prepared_dataset=${ROBOT_DATASET_ROOT}
toy_repeats=${TOY_REPEATS}
lmr_repeats=${LMR_REPEATS}
steps=${STEPS}
action_lr=${ACTION_LR}
vlm_lora_trainable=1
vlm_lora_lr=${VLM_LORA_LR}
aux_vlm_lambda=${AUX_VLM_LAMBDA}
aux_vlm_qa_batch=${AUX_VLM_BATCH_SIZE}
MANIFEST

echo "E2E aux VLM training finished."
echo "Final checkpoint: ${FINAL_CKPT}"
