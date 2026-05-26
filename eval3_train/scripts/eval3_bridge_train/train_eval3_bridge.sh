#!/usr/bin/env bash
# Bridge training: align celebrity-prompt action hidden states to spatial-prompt ones.
# VLM LoRA is frozen. Action expert + spatial_head are trainable.
# Two bridge losses: MSE alignment + spatial cross-entropy classification.
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

# ── Checkpoints ──────────────────────────────────────────────────────────────
START_ACTION_CHECKPOINT="${START_ACTION_CHECKPOINT:-${ROOT_DIR}/outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_action/checkpoints/000500/pretrained_model}"
START_VLM_LORA="${START_VLM_LORA:-${ROOT_DIR}/outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_vlm_lora}"

# ── Dataset ───────────────────────────────────────────────────────────────────
ACTION_TOY_DATASET_ROOT="${ACTION_TOY_DATASET_ROOT:-${ROOT_DIR}/data/eval3_90_permutation}"
ACTION_LMR_DATASET_ROOT="${ACTION_LMR_DATASET_ROOT:-${ROOT_DIR}/data/eval3_lmr}"
SIDECAR_PATH="${SIDECAR_PATH:-${ROOT_DIR}/data/eval3_90_permutation_spatial_labels/target_slot_by_episode.json}"

TOY_REPEATS="${TOY_REPEATS:-4}"
LMR_REPEATS=0

# ── Hyperparameters ───────────────────────────────────────────────────────────
STEPS="${STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACTION_LR="${ACTION_LR:-1e-5}"
ALIGN_LAMBDA="${ALIGN_LAMBDA:-0.1}"
SPATIAL_LAMBDA="${SPATIAL_LAMBDA:-1.0}"
SPATIAL_HEAD_LR="${SPATIAL_HEAD_LR:-1e-4}"

# ── Prepared dataset ──────────────────────────────────────────────────────────
PREPARED_DATASET_ROOT="${PREPARED_DATASET_ROOT:-${ROOT_DIR}/data/eval3_bridge_toy${TOY_REPEATS}}"
ROBOT_DATASET_ROOT="${ROBOT_DATASET_ROOT:-${PREPARED_DATASET_ROOT}}"
ROBOT_DATASET_REPO_ID="${ROBOT_DATASET_REPO_ID:-robot-learning-group47/eval3_bridge_toy${TOY_REPEATS}}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/eval3_bridge_cycle05_${STEPS}steps_alm${ALIGN_LAMBDA}_slm${SPATIAL_LAMBDA}}"

WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-group47-eval3}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_MODE="${WANDB_MODE:-online}"

# ── Guards ────────────────────────────────────────────────────────────────────
if [ -d "${OUTPUT_DIR}" ] && [ "${BRIDGE_OVERWRITE:-0}" != "1" ]; then
  echo "ERROR: ${OUTPUT_DIR} already exists." >&2
  echo "Set BRIDGE_OVERWRITE=1 to replace it." >&2
  exit 1
fi

if [ ! -d "${START_ACTION_CHECKPOINT}" ]; then
  echo "ERROR: Missing start action checkpoint: ${START_ACTION_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -f "${START_VLM_LORA}/adapter_model.safetensors" ]; then
  echo "ERROR: Missing VLM LoRA: ${START_VLM_LORA}" >&2
  exit 1
fi

# ── Generate sidecar labels (idempotent) ─────────────────────────────────────
echo "===== Generating spatial slot labels ====="
python scripts/eval3_bridge_train/relabel_eval3_90_target_slot.py \
  --dataset-root="${ACTION_TOY_DATASET_ROOT}" \
  --output-dir="$(dirname "${SIDECAR_PATH}")"

python scripts/eval3_bridge_train/check_eval3_spatial_labels.py \
  --sidecar="${SIDECAR_PATH}"

# ── Prepare action dataset (toy only, no LMR) ────────────────────────────────
echo "===== Preparing dataset (toy x${TOY_REPEATS}, no LMR) ====="
PREPARE_ARGS=(
  --toy-root="${ACTION_TOY_DATASET_ROOT}"
  --lmr-root="${ACTION_LMR_DATASET_ROOT}"
  --output-root="${PREPARED_DATASET_ROOT}"
  --toy-repeats="${TOY_REPEATS}"
  --lmr-repeats="${LMR_REPEATS}"
)
if [ "${BRIDGE_PREPARE_OVERWRITE:-0}" = "1" ]; then
  PREPARE_ARGS+=(--overwrite)
fi
python scripts/eval3_cotrain/prepare_action_dataset.py "${PREPARE_ARGS[@]}"

# ── Export bridge env vars ────────────────────────────────────────────────────
export EVAL3_VLM_LORA_PATH="${START_VLM_LORA}"
export EVAL3_VLM_LORA_TRAINABLE=0
export EVAL3_BRIDGE_SIDECAR_PATH="${SIDECAR_PATH}"
export EVAL3_BRIDGE_ALIGN_LAMBDA="${ALIGN_LAMBDA}"
export EVAL3_BRIDGE_SPATIAL_LAMBDA="${SPATIAL_LAMBDA}"
export EVAL3_BRIDGE_SPATIAL_HEAD_LR="${SPATIAL_HEAD_LR}"
export WANDB_NAME="eval3_bridge_cycle05_${STEPS}steps_alm${ALIGN_LAMBDA}_slm${SPATIAL_LAMBDA}"

echo "===== Bridge training ====="
echo "Action checkpoint : ${START_ACTION_CHECKPOINT}"
echo "VLM LoRA (frozen) : ${START_VLM_LORA}"
echo "Dataset           : ${ROBOT_DATASET_ROOT}"
echo "Steps             : ${STEPS}  batch=${BATCH_SIZE}  save_freq=${SAVE_FREQ}"
echo "Action LR         : ${ACTION_LR}  Spatial head LR: ${SPATIAL_HEAD_LR}"
echo "Align lambda      : ${ALIGN_LAMBDA}  Spatial lambda: ${SPATIAL_LAMBDA}"
echo "Output            : ${OUTPUT_DIR}"

python scripts/eval3_bridge_train/train_eval3_bridge.py \
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
  --policy.repo_id=robot-learning-group47/eval3_bridge_cycle05_smolvla

FINAL_CKPT="${OUTPUT_DIR}/checkpoints/$(printf "%06d" "${STEPS}")/pretrained_model"
if [ ! -d "${FINAL_CKPT}" ]; then
  echo "ERROR: Expected checkpoint not found: ${FINAL_CKPT}" >&2
  exit 1
fi

cat > "${OUTPUT_DIR}/bridge_manifest.txt" <<MANIFEST
start_action_checkpoint=${START_ACTION_CHECKPOINT}
start_vlm_lora=${START_VLM_LORA}
prepared_dataset=${ROBOT_DATASET_ROOT}
toy_repeats=${TOY_REPEATS}
lmr_repeats=${LMR_REPEATS}
steps=${STEPS}
action_lr=${ACTION_LR}
vlm_lora_trainable=0
align_lambda=${ALIGN_LAMBDA}
spatial_lambda=${SPATIAL_LAMBDA}
spatial_head_lr=${SPATIAL_HEAD_LR}
MANIFEST

echo "Bridge training finished."
echo "Final checkpoint: ${FINAL_CKPT}"
