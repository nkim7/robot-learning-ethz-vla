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
ACTION_TOY_DATASET_ROOT="${ACTION_TOY_DATASET_ROOT:-${ROOT_DIR}/data/eval3_90_permutation}"
ACTION_LMR_DATASET_ROOT="${ACTION_LMR_DATASET_ROOT:-${ROOT_DIR}/data/eval3_lmr}"
PREPARED_ACTION_DATASET_ROOT="${PREPARED_ACTION_DATASET_ROOT:-${ROOT_DIR}/data/eval3_binding_repair_action_90toy4_plus_lmr1}"
ROBOT_DATASET_ROOT="${ROBOT_DATASET_ROOT:-${PREPARED_ACTION_DATASET_ROOT}}"
ROBOT_DATASET_REPO_ID="${ROBOT_DATASET_REPO_ID:-robot-learning-group47/eval3_binding_repair_action_90toy4_plus_lmr1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/eval3_binding_repair_toy4_lmr1_cycle05_1500}"

TOY_REPEATS="${TOY_REPEATS:-4}"
LMR_REPEATS="${LMR_REPEATS:-1}"
STEPS="${STEPS:-1500}"
SAVE_FREQ="${SAVE_FREQ:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACTION_LR="${ACTION_LR:-5e-6}"
VLM_LORA_LR="${VLM_LORA_LR:-1e-6}"
WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-group47-eval3}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

if [ -d "${OUTPUT_DIR}" ] && [ "${BINDING_REPAIR_OVERWRITE:-0}" != "1" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists. Refusing to overwrite." >&2
  echo "Set BINDING_REPAIR_OVERWRITE=1 only if you intentionally want to replace it." >&2
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

prepare_args=(
  --toy-root="${ACTION_TOY_DATASET_ROOT}"
  --lmr-root="${ACTION_LMR_DATASET_ROOT}"
  --output-root="${PREPARED_ACTION_DATASET_ROOT}"
  --toy-repeats="${TOY_REPEATS}"
  --lmr-repeats="${LMR_REPEATS}"
)
if [ "${BINDING_REPAIR_PREPARE_OVERWRITE:-0}" = "1" ]; then
  prepare_args+=(--overwrite)
fi
python scripts/eval3_cotrain/prepare_action_dataset.py "${prepare_args[@]}"

export EVAL3_VLM_LORA_PATH="${START_VLM_LORA}"
export EVAL3_VLM_LORA_TRAINABLE=1
export EVAL3_VLM_LORA_LR="${VLM_LORA_LR}"
export WANDB_NAME="eval3_binding_repair_toy4_lmr1_cycle05_${STEPS}"

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
  --policy.push_to_hub=false \
  --policy.repo_id=robot-learning-group47/eval3_binding_repair_toy4_lmr1_smolvla

cat > "${OUTPUT_DIR}/binding_repair_manifest.txt" <<MANIFEST
start_action_checkpoint=${START_ACTION_CHECKPOINT}
start_vlm_lora=${START_VLM_LORA}
prepared_action_dataset=${ROBOT_DATASET_ROOT}
toy_repeats=${TOY_REPEATS}
lmr_repeats=${LMR_REPEATS}
steps=${STEPS}
action_lr=${ACTION_LR}
vlm_lora_trainable=1
vlm_lora_lr=${VLM_LORA_LR}
MANIFEST

echo "Binding repair finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Latest checkpoint: ${OUTPUT_DIR}/checkpoints/$(printf "%06d" "${STEPS}")/pretrained_model"
