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
START_VLM_LORA="${START_VLM_LORA:-${ROOT_DIR}/outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_vlm_lora}"
GROUNDING_DATA_DIR="${GROUNDING_DATA_DIR:-${ROOT_DIR}/data/eval3_position_qa_3frames}"
ROBOT_DATASET_ROOT="${ROBOT_DATASET_ROOT:-${ROOT_DIR}/data/eval3_binding_repair_action_90toy4_plus_lmr1}"
ROBOT_DATASET_REPO_ID="${ROBOT_DATASET_REPO_ID:-robot-learning-group47/eval3_binding_repair_action_90toy4_plus_lmr1}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/eval3_cycle06_vlm_repair_then_action_frozen_lora}"
VLM_OUT="${VLM_OUT:-${OUTPUT_DIR}/cycle_06_vlm_lora_repair}"
ACTION_OUT="${ACTION_OUT:-${OUTPUT_DIR}/cycle_06_action_binding_frozen_vlm_lora}"

VLM_EPOCHS="${VLM_EPOCHS:-5}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-1}"
VLM_GRAD_ACCUM="${VLM_GRAD_ACCUM:-8}"
VLM_LR="${VLM_LR:-1e-5}"

ACTION_STEPS="${ACTION_STEPS:-1500}"
ACTION_SAVE_FREQ="${ACTION_SAVE_FREQ:-500}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-8}"
ACTION_LR="${ACTION_LR:-5e-6}"
WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-group47-eval3}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

if [ -d "${OUTPUT_DIR}" ] && [ "${CYCLE06_OVERWRITE:-0}" != "1" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists. Refusing to overwrite." >&2
  echo "Set CYCLE06_OVERWRITE=1 only if you intentionally want to replace it." >&2
  exit 1
fi

if [ ! -d "${START_ACTION_CHECKPOINT}" ]; then
  echo "ERROR: Missing start action checkpoint: ${START_ACTION_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -f "${START_VLM_LORA}/adapter_model.safetensors" ]; then
  echo "ERROR: Missing start VLM LoRA adapter: ${START_VLM_LORA}" >&2
  exit 1
fi
if [ ! -f "${GROUNDING_DATA_DIR}/train.jsonl" ] || [ ! -f "${GROUNDING_DATA_DIR}/eval.jsonl" ]; then
  echo "ERROR: Missing position QA train/eval jsonl files in ${GROUNDING_DATA_DIR}" >&2
  exit 1
fi
if [ ! -d "${ROBOT_DATASET_ROOT}" ]; then
  echo "ERROR: Missing robot action dataset: ${ROBOT_DATASET_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "===== Cycle 06: VLM LoRA repair on QA frames, no oversampling ====="
echo "Start VLM LoRA: ${START_VLM_LORA}"
echo "Grounding data: ${GROUNDING_DATA_DIR}"
python scripts/eval3_vlm_lora/finetune_position_lora.py \
  --data-dir="${GROUNDING_DATA_DIR}" \
  --checkpoint="${START_VLM_LORA}" \
  --output-dir="${VLM_OUT}" \
  --epochs="${VLM_EPOCHS}" \
  --batch-size="${VLM_BATCH_SIZE}" \
  --grad-accum="${VLM_GRAD_ACCUM}" \
  --lr="${VLM_LR}"

if [ ! -f "${VLM_OUT}/adapter_model.safetensors" ]; then
  echo "ERROR: Expected repaired VLM LoRA was not created: ${VLM_OUT}/adapter_model.safetensors" >&2
  exit 1
fi

python scripts/eval3_vlm_lora/evaluate_position_jsonl.py \
  --checkpoint="${VLM_OUT}" \
  --jsonl="${GROUNDING_DATA_DIR}/all.jsonl" \
  --output="${VLM_OUT}/position_qa_all_predictions.csv" \
  --dtype=auto

echo "===== Cycle 06: action binding with repaired VLM LoRA frozen ====="
echo "Start action checkpoint: ${START_ACTION_CHECKPOINT}"
echo "Frozen repaired VLM LoRA: ${VLM_OUT}"
export EVAL3_VLM_LORA_PATH="${VLM_OUT}"
export EVAL3_VLM_LORA_TRAINABLE=0
export WANDB_NAME="eval3_cycle06_action_binding_frozen_repaired_vlm_lora"

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
  --num_workers=4 \
  --output_dir="${ACTION_OUT}" \
  --job_name="${WANDB_NAME}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --policy.push_to_hub=false \
  --policy.repo_id=robot-learning-group47/eval3_cycle06_binding_repair_frozen_lora_smolvla

ACTION_CKPT="${ACTION_OUT}/checkpoints/$(printf "%06d" "${ACTION_STEPS}")/pretrained_model"
if [ ! -d "${ACTION_CKPT}" ]; then
  echo "ERROR: Expected action checkpoint was not created: ${ACTION_CKPT}" >&2
  exit 1
fi

cat > "${OUTPUT_DIR}/cycle_06_manifest.txt" <<MANIFEST
start_action_checkpoint=${START_ACTION_CHECKPOINT}
start_vlm_lora=${START_VLM_LORA}
grounding_data=${GROUNDING_DATA_DIR}
robot_dataset=${ROBOT_DATASET_ROOT}
vlm_lora=${VLM_OUT}
vlm_epochs=${VLM_EPOCHS}
vlm_lr=${VLM_LR}
action_checkpoint=${ACTION_CKPT}
action_steps=${ACTION_STEPS}
action_lr=${ACTION_LR}
action_batch_size=${ACTION_BATCH_SIZE}
vlm_lora_frozen_during_action=1
MANIFEST

echo "Cycle 06 finished."
echo "Repaired VLM LoRA: ${VLM_OUT}"
echo "Frozen-LoRA action checkpoint: ${ACTION_CKPT}"
