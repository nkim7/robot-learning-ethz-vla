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

CYCLES="${CYCLES:-5}"
ACTION_STEPS="${ACTION_STEPS:-500}"
VLM_STEPS="${VLM_STEPS:-100}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-16}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-1}"
VLM_GRAD_ACCUM="${VLM_GRAD_ACCUM:-4}"
ACTION_LR="${ACTION_LR:-1e-5}"
VLM_LR="${VLM_LR:-1e-5}"

START_ACTION_CHECKPOINT="${START_ACTION_CHECKPOINT:-${ROOT_DIR}/outputs/train/eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k/checkpoints/014000/pretrained_model}"
START_VLM_LORA="${START_VLM_LORA:-${ROOT_DIR}/checkpoint/eval3_phase2_TOY}"
GROUNDING_DATA_DIR="${GROUNDING_DATA_DIR:-${ROOT_DIR}/data/eval3_position_qa_3frames}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/eval3_cotrain_action_lora_auxvlm_5k}"
ACTION_TOY_DATASET_ROOT="${ACTION_TOY_DATASET_ROOT:-${ROOT_DIR}/data/eval3_90_permutation}"
ACTION_LMR_DATASET_ROOT="${ACTION_LMR_DATASET_ROOT:-${ROOT_DIR}/data/eval3_lmr}"
PREPARED_ACTION_DATASET_ROOT="${PREPARED_ACTION_DATASET_ROOT:-${ROOT_DIR}/data/eval3_cotrain_action_90toy_plus_lmr}"
ROBOT_DATASET_ROOT="${ROBOT_DATASET_ROOT:-${PREPARED_ACTION_DATASET_ROOT}}"
ROBOT_DATASET_REPO_ID="${ROBOT_DATASET_REPO_ID:-robot-learning-group47/eval3_cotrain_action_90toy_plus_lmr}"
WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-group47-eval3}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

export START_ACTION_CHECKPOINT
export START_VLM_LORA
export GROUNDING_DATA_DIR
export ACTION_TOY_DATASET_ROOT
export ACTION_LMR_DATASET_ROOT
export PREPARED_ACTION_DATASET_ROOT
export ROBOT_DATASET_ROOT

if [ -d "${OUTPUT_DIR}" ] && [ "${COTRAIN_OVERWRITE:-0}" != "1" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists. Refusing to overwrite." >&2
  echo "Set COTRAIN_OVERWRITE=1 only if you intentionally want to replace it." >&2
  exit 1
fi

prepare_args=(
  --toy-root="${ACTION_TOY_DATASET_ROOT}"
  --lmr-root="${ACTION_LMR_DATASET_ROOT}"
  --output-root="${PREPARED_ACTION_DATASET_ROOT}"
)
if [ "${COTRAIN_PREPARE_OVERWRITE:-0}" = "1" ]; then
  prepare_args+=(--overwrite)
fi

python scripts/eval3_cotrain/prepare_action_dataset.py "${prepare_args[@]}"
python scripts/eval3_cotrain/check_cotrain_inputs.py

mkdir -p "${OUTPUT_DIR}"

current_action="${START_ACTION_CHECKPOINT}"
current_lora="${START_VLM_LORA}"

for cycle in $(seq 1 "${CYCLES}"); do
  cycle_id="$(printf "%02d" "${cycle}")"
  action_out="${OUTPUT_DIR}/cycle_${cycle_id}_action"
  lora_out="${OUTPUT_DIR}/cycle_${cycle_id}_vlm_lora"
  action_ckpt="${action_out}/checkpoints/$(printf "%06d" "${ACTION_STEPS}")/pretrained_model"

  echo "===== Cycle ${cycle_id}/${CYCLES}: action phase ====="
  echo "Action start checkpoint: ${current_action}"
  echo "Frozen VLM LoRA: ${current_lora}"
  export EVAL3_VLM_LORA_PATH="${current_lora}"
  export WANDB_NAME="eval3_cotrain_cycle_${cycle_id}_action"

  python scripts/eval3_run2_lmr210_action_train/run2_train_launcher.py \
    --dataset.repo_id="${ROBOT_DATASET_REPO_ID}" \
    --dataset.root="${ROBOT_DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --policy.path="${current_action}" \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.use_peft=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.freeze_vision_encoder=true \
    --policy.optimizer_lr="${ACTION_LR}" \
    --batch_size="${ACTION_BATCH_SIZE}" \
    --steps="${ACTION_STEPS}" \
    --save_freq="${ACTION_STEPS}" \
    --eval_freq=0 \
    --log_freq=25 \
    --num_workers=4 \
    --output_dir="${action_out}" \
    --job_name="${WANDB_NAME}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    --policy.push_to_hub=false \
    --policy.repo_id=robot-learning-group47/eval3_cotrain_action_lora_auxvlm_5k_smolvla

  if [ ! -d "${action_ckpt}" ]; then
    echo "ERROR: Expected action checkpoint was not created: ${action_ckpt}" >&2
    exit 1
  fi

  echo "===== Cycle ${cycle_id}/${CYCLES}: VLM LoRA refresh ====="
  echo "VLM LoRA start checkpoint: ${current_lora}"
  python scripts/eval3_vlm_lora/finetune_position_lora.py \
    --data-dir="${GROUNDING_DATA_DIR}" \
    --checkpoint="${current_lora}" \
    --output-dir="${lora_out}" \
    --epochs=20 \
    --batch-size="${VLM_BATCH_SIZE}" \
    --grad-accum="${VLM_GRAD_ACCUM}" \
    --lr="${VLM_LR}" \
    --max-steps="${VLM_STEPS}"

  if [ ! -f "${lora_out}/adapter_model.safetensors" ]; then
    echo "ERROR: Expected VLM LoRA adapter was not created: ${lora_out}/adapter_model.safetensors" >&2
    exit 1
  fi

  python scripts/eval3_cotrain/eval_cotrain_vlm_grounding.py \
    --checkpoint="${lora_out}" \
    --data="${GROUNDING_DATA_DIR}/train.jsonl" \
    --output="${lora_out}/cotrain_predictions.jsonl" \
    --batch-size=8

  cat > "${OUTPUT_DIR}/cycle_${cycle_id}_manifest.txt" <<MANIFEST
cycle=${cycle_id}
action_checkpoint=${action_ckpt}
vlm_lora=${lora_out}
grounding_data=${GROUNDING_DATA_DIR}
MANIFEST

  current_action="${action_ckpt}"
  current_lora="${lora_out}"
done

cat > "${OUTPUT_DIR}/latest_manifest.txt" <<MANIFEST
cycles=${CYCLES}
latest_action_checkpoint=${current_action}
latest_vlm_lora=${current_lora}
grounding_data=${GROUNDING_DATA_DIR}
MANIFEST

echo "Co-training finished."
echo "Latest action checkpoint: ${current_action}"
echo "Latest VLM LoRA: ${current_lora}"
