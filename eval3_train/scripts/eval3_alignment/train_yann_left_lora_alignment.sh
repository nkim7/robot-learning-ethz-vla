#!/usr/bin/env bash
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
BETTER_VLM_LORA="${BETTER_VLM_LORA:-${ROOT_DIR}/checkpoint/eval3_yann_left_repair_adapter}"

REQUESTED_DATASET_ROOT="${ROOT_DIR}/data/eval3_binding_repair_action_90toy4_plus_lmr"
FALLBACK_DATASET_ROOT="${ROOT_DIR}/data/eval3_binding_repair_action_90toy4_plus_lmr1"
ROBOT_DATASET_ROOT="${ROBOT_DATASET_ROOT:-${REQUESTED_DATASET_ROOT}}"
if [ ! -d "${ROBOT_DATASET_ROOT}" ] && [ "${ROBOT_DATASET_ROOT}" = "${REQUESTED_DATASET_ROOT}" ] && [ -d "${FALLBACK_DATASET_ROOT}" ]; then
  echo "WARN: ${REQUESTED_DATASET_ROOT} does not exist; using ${FALLBACK_DATASET_ROOT}" >&2
  ROBOT_DATASET_ROOT="${FALLBACK_DATASET_ROOT}"
fi
ROBOT_DATASET_REPO_ID="${ROBOT_DATASET_REPO_ID:-robot-learning-group47/$(basename "${ROBOT_DATASET_ROOT}")}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/eval3_yann_left_lora_alignment_cycle05_action}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_DIR}_setup}"
ACTION_STEPS="${ACTION_STEPS:-1500}"
ACTION_SAVE_FREQ="${ACTION_SAVE_FREQ:-500}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-8}"
ACTION_LR="${ACTION_LR:-5e-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ALIGNMENT_SKIP_CHECK="${ALIGNMENT_SKIP_CHECK:-0}"
ALIGNMENT_OVERWRITE="${ALIGNMENT_OVERWRITE:-0}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-group47-eval3}"
WANDB_NAME="${WANDB_NAME:-eval3_yann_left_lora_alignment_cycle05_action}"
WANDB_MODE="${WANDB_MODE:-online}"

if [ -d "${OUTPUT_DIR}" ] && [ "${ALIGNMENT_OVERWRITE}" != "1" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists. Refusing to overwrite." >&2
  echo "Set ALIGNMENT_OVERWRITE=1 only if you intentionally want to replace it." >&2
  exit 1
fi

if [ ! -d "${START_ACTION_CHECKPOINT}" ]; then
  echo "ERROR: Missing start action checkpoint: ${START_ACTION_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -f "${BETTER_VLM_LORA}/adapter_model.safetensors" ]; then
  echo "ERROR: Missing VLM LoRA adapter weights: ${BETTER_VLM_LORA}/adapter_model.safetensors" >&2
  exit 1
fi
if [ ! -f "${BETTER_VLM_LORA}/adapter_config.json" ]; then
  echo "ERROR: Missing VLM LoRA adapter config: ${BETTER_VLM_LORA}/adapter_config.json" >&2
  exit 1
fi
if [ ! -d "${ROBOT_DATASET_ROOT}" ]; then
  echo "ERROR: Missing robot action dataset: ${ROBOT_DATASET_ROOT}" >&2
  exit 1
fi

mkdir -p "${SETUP_DIR}"

export EVAL3_VLM_LORA_PATH="${BETTER_VLM_LORA}"
export EVAL3_VLM_LORA_TRAINABLE=0
export WANDB_NAME

cat > "${SETUP_DIR}/alignment_manifest.txt" <<MANIFEST
start_action_checkpoint=${START_ACTION_CHECKPOINT}
better_vlm_lora=${BETTER_VLM_LORA}
robot_dataset=${ROBOT_DATASET_ROOT}
robot_dataset_repo_id=${ROBOT_DATASET_REPO_ID}
output_dir=${OUTPUT_DIR}
action_steps=${ACTION_STEPS}
action_save_freq=${ACTION_SAVE_FREQ}
action_batch_size=${ACTION_BATCH_SIZE}
action_lr=${ACTION_LR}
vlm_lora_frozen=1
base_vlm_frozen=1
vision_encoder_frozen=1
train_action_expert=1
train_state_projection=1
train_action_projection=1
wandb_enable=${WANDB_ENABLE}
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
wandb_mode=${WANDB_MODE}
MANIFEST

if [ "${ALIGNMENT_SKIP_CHECK}" != "1" ]; then
  echo "===== Verifying alignment trainability ====="
  python scripts/eval3_alignment/check_alignment_trainable.py \
    --action-checkpoint="${START_ACTION_CHECKPOINT}" \
    --adapter="${BETTER_VLM_LORA}" \
    --dataset-root="${ROBOT_DATASET_ROOT}" \
    --dataset-repo-id="${ROBOT_DATASET_REPO_ID}" \
    --output="${SETUP_DIR}/trainability_check.json" \
    --device=cuda \
    --lr="${ACTION_LR}"
fi

echo "===== Eval 3 LoRA alignment fine-tune ====="
echo "Action checkpoint: ${START_ACTION_CHECKPOINT}"
echo "Frozen VLM LoRA: ${BETTER_VLM_LORA}"
echo "Dataset: ${ROBOT_DATASET_ROOT}"
echo "W&B: enable=${WANDB_ENABLE} project=${WANDB_PROJECT} name=${WANDB_NAME} mode=${WANDB_MODE}"

python scripts/eval3_run2_lmr210_action_train/run2_train_launcher.py \
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
  --batch_size="${ACTION_BATCH_SIZE}" \
  --steps="${ACTION_STEPS}" \
  --save_freq="${ACTION_SAVE_FREQ}" \
  --eval_freq=0 \
  --log_freq=25 \
  --num_workers="${NUM_WORKERS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${WANDB_NAME}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.mode="${WANDB_MODE}" \
  --policy.push_to_hub=false \
  --policy.repo_id=robot-learning-group47/eval3_yann_left_lora_alignment_cycle05_action_smolvla

ACTION_CKPT="${OUTPUT_DIR}/checkpoints/$(printf "%06d" "${ACTION_STEPS}")/pretrained_model"
if [ ! -d "${ACTION_CKPT}" ]; then
  echo "ERROR: Expected alignment checkpoint was not created: ${ACTION_CKPT}" >&2
  exit 1
fi

cp "${SETUP_DIR}/alignment_manifest.txt" "${OUTPUT_DIR}/alignment_manifest.txt"
if [ -f "${SETUP_DIR}/trainability_check.json" ]; then
  cp "${SETUP_DIR}/trainability_check.json" "${OUTPUT_DIR}/trainability_check.json"
fi

echo "Alignment fine-tune finished."
echo "Final checkpoint: ${ACTION_CKPT}"
