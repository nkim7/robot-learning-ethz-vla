#!/usr/bin/env bash

EVAL3_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL3_ROOT="$(cd "${EVAL3_COMMON_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL3_ROOT}/../.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"

EVAL3_COTRAIN_REPO_ID="${EVAL3_COTRAIN_REPO_ID:-robot-learning-group47/eval3_cotrain}"
EVAL3_CELEB_REPO_ID="${EVAL3_CELEB_REPO_ID:-robot-learning-group47/eval3_phase2_celeb100}"
EVAL3_E2E_REPO_ID="${EVAL3_E2E_REPO_ID:-robot-learning-group47/eval3_bridge_cycle05_smolvla}"
EVAL3_BASE_REPO_ID="${EVAL3_BASE_REPO_ID:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"

EVAL3_COTRAIN_DIR="${EVAL3_COTRAIN_DIR:-${CHECKPOINT_ROOT}/eval3_cotrain}"
EVAL3_CELEB_LORA_DIR="${EVAL3_CELEB_LORA_DIR:-${CHECKPOINT_ROOT}/eval3_phase2_celeb100}"
EVAL3_E2E_POLICY_DIR="${EVAL3_E2E_POLICY_DIR:-${CHECKPOINT_ROOT}/eval3_bridge_cycle05_smolvla}"
EVAL3_BASE_MODEL_DIR="${EVAL3_BASE_MODEL_DIR:-${CHECKPOINT_ROOT}/base_models/HuggingFaceTB__SmolVLM2-2.2B-Instruct}"

EVAL3_DEFAULT_CAMERA_CONFIG='{"front":{"type":"opencv_crop","index_or_path":0,"source_width":640,"source_height":480,"roi_x":215,"roi_y":0,"roi_size":425,"width":256,"height":256,"fps":30,"warmup_s":1}}'

eval3_log() {
  printf '[eval3] %s\n' "$*"
}

eval3_die() {
  printf '[eval3] ERROR: %s\n' "$*" >&2
  exit 1
}

eval3_has_command() {
  command -v "$1" >/dev/null 2>&1
}

eval3_setup_env() {
  if [ ! -x "${PYTHON_BIN}" ]; then
    eval3_log "Creating virtualenv at ${VENV_DIR}"
    if eval3_has_command uv; then
      uv venv --python "${PYTHON_VERSION:-3.12}" "${VENV_DIR}"
    elif eval3_has_command python3.12; then
      python3.12 -m venv "${VENV_DIR}"
    else
      python3 -m venv "${VENV_DIR}"
    fi
  fi

  if [ "${EVAL3_SKIP_INSTALL:-0}" = "1" ]; then
    eval3_log "Skipping dependency install because EVAL3_SKIP_INSTALL=1"
    return
  fi

  eval3_log "Installing LeRobot runtime extras into ${VENV_DIR}"
  if eval3_has_command uv; then
    uv pip install --python "${PYTHON_BIN}" \
      -e "${REPO_ROOT}/lerobot[feetech,smolvla,async,peft]"
  else
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install \
      -e "${REPO_ROOT}/lerobot[feetech,smolvla,async,peft]"
  fi
}

eval3_snapshot_download() {
  local repo_id="$1"
  local local_dir="$2"

  if [ -d "${local_dir}" ] && [ -n "$(ls -A "${local_dir}" 2>/dev/null)" ] && [ "${EVAL3_FORCE_DOWNLOAD:-0}" != "1" ]; then
    eval3_log "Using existing ${repo_id} at ${local_dir}"
    return
  fi

  mkdir -p "${local_dir}"
  eval3_log "Downloading ${repo_id} to ${local_dir}"
  "${PYTHON_BIN}" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' \
    "${repo_id}" "${local_dir}"
}

eval3_download_base_model() {
  if [ "${EVAL3_SKIP_BASE_MODEL_DOWNLOAD:-0}" = "1" ]; then
    eval3_log "Skipping base VLM download because EVAL3_SKIP_BASE_MODEL_DOWNLOAD=1"
    return
  fi
  eval3_snapshot_download "${EVAL3_BASE_REPO_ID}" "${EVAL3_BASE_MODEL_DIR}"
}

eval3_prepare_cotrain_assets() {
  eval3_setup_env
  eval3_snapshot_download "${EVAL3_COTRAIN_REPO_ID}" "${EVAL3_COTRAIN_DIR}"
  eval3_download_base_model
}

eval3_prepare_celeb_assets() {
  eval3_setup_env
  eval3_snapshot_download "${EVAL3_COTRAIN_REPO_ID}" "${EVAL3_COTRAIN_DIR}"
  eval3_snapshot_download "${EVAL3_CELEB_REPO_ID}" "${EVAL3_CELEB_LORA_DIR}"
  eval3_download_base_model
}

eval3_prepare_e2e_assets() {
  eval3_setup_env
  eval3_snapshot_download "${EVAL3_E2E_REPO_ID}" "${EVAL3_E2E_POLICY_DIR}"
  eval3_snapshot_download "${EVAL3_COTRAIN_REPO_ID}" "${EVAL3_COTRAIN_DIR}"
  eval3_download_base_model
}

eval3_first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [ -d "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  printf '%s\n' "$1"
}

eval3_cotrain_action_checkpoint() {
  if [ -n "${EVAL3_ACTION_CHECKPOINT:-}" ]; then
    printf '%s\n' "${EVAL3_ACTION_CHECKPOINT}"
    return
  fi

  eval3_first_existing_dir \
    "${EVAL3_COTRAIN_DIR}/cycle_05_action_checkpoint" \
    "${EVAL3_COTRAIN_DIR}/cycle_05_action/checkpoints/000500/pretrained_model" \
    "${EVAL3_COTRAIN_DIR}"
}

eval3_cotrain_vlm_lora() {
  if [ -n "${EVAL3_COTRAIN_VLM_LORA_DIR:-}" ]; then
    printf '%s\n' "${EVAL3_COTRAIN_VLM_LORA_DIR}"
    return
  fi

  eval3_first_existing_dir \
    "${EVAL3_COTRAIN_DIR}/cycle_05_vlm_lora" \
    "${EVAL3_COTRAIN_DIR}/cycle_05_vlm_lora/checkpoint" \
    "${EVAL3_COTRAIN_DIR}"
}

eval3_require_dir() {
  [ -d "$1" ] || eval3_die "Missing directory: $1"
}

eval3_require_file() {
  [ -f "$1" ] || eval3_die "Missing file: $1"
}

eval3_export_server_env() {
  local vlm_lora_path="$1"

  export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-/tmp/hf-home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets-cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

  export EVAL3_EXPECTED_VLM_BASE="${EVAL3_EXPECTED_VLM_BASE:-${EVAL3_BASE_REPO_ID}}"
  export EVAL3_VLM_BASE_PATH="${EVAL3_VLM_BASE_PATH:-${EVAL3_BASE_MODEL_DIR}}"
  export EVAL3_VLM_LORA_PATH="${vlm_lora_path}"
  export EVAL3_FIRST_FRAME_PROBE="${EVAL3_FIRST_FRAME_PROBE:-1}"
  export EVAL3_INSTRUCTION_REWRITE="${EVAL3_INSTRUCTION_REWRITE:-1}"
  export EVAL3_RUN1_PATCH_ACTIVE="${EVAL3_RUN1_PATCH_ACTIVE:-1}"
  export EVAL3_VLM_LORA_TRAINABLE="${EVAL3_VLM_LORA_TRAINABLE:-0}"
}

eval3_run_preprompt_server() {
  local vlm_lora_path="$1"
  shift

  eval3_require_dir "${vlm_lora_path}"
  eval3_require_file "${vlm_lora_path}/adapter_model.safetensors"
  eval3_export_server_env "${vlm_lora_path}"

  eval3_log "Starting preprompt policy server on ${HOST:-0.0.0.0}:${PORT:-8080}"
  eval3_log "VLM LoRA: ${EVAL3_VLM_LORA_PATH}"
  eval3_log "Base VLM: ${EVAL3_VLM_BASE_PATH}"

  exec "${PYTHON_BIN}" "${REPO_ROOT}/eval3/scripts/eval3_run2_lmr210_action_train/run2_policy_server_launcher.py" \
    --host="${HOST:-0.0.0.0}" \
    --port="${PORT:-8080}" \
    --fps="${FPS:-30}" \
    --inference_latency="${INFERENCE_LATENCY:-0.033}" \
    --obs_queue_timeout="${OBS_QUEUE_TIMEOUT:-1}" \
    "$@"
}

eval3_run_e2e_server() {
  local vlm_lora_path="$1"
  shift

  eval3_require_dir "${vlm_lora_path}"
  eval3_require_file "${vlm_lora_path}/adapter_model.safetensors"
  eval3_export_server_env "${vlm_lora_path}"
  export EVAL3_INSTRUCTION_REWRITE=0

  eval3_log "Starting end-to-end policy server on ${HOST:-0.0.0.0}:${PORT:-8080}"
  eval3_log "E2E server uses celebrity prompts directly; instruction rewrite is disabled."
  eval3_log "VLM LoRA: ${EVAL3_VLM_LORA_PATH}"
  eval3_log "Base VLM: ${EVAL3_VLM_BASE_PATH}"

  exec "${PYTHON_BIN}" "${REPO_ROOT}/eval3/scripts/eval3_bridge_train/bridge_policy_server_launcher.py" \
    --host="${HOST:-0.0.0.0}" \
    --port="${PORT:-8080}" \
    --fps="${FPS:-30}" \
    --inference_latency="${INFERENCE_LATENCY:-0.033}" \
    --obs_queue_timeout="${OBS_QUEUE_TIMEOUT:-1}" \
    "$@"
}

eval3_require_follower_port() {
  if [ -z "${FOLLOWER_PORT:-}" ]; then
    eval3_die "Set FOLLOWER_PORT before running the robot client."
  fi
}

eval3_run_client() {
  local server_policy_path="$1"
  shift

  eval3_setup_env
  eval3_require_follower_port

  local camera_config="${EVAL3_CAMERA_CONFIG:-${EVAL3_DEFAULT_CAMERA_CONFIG}}"
  local task="${TASK:-Place the coke on Taylor Swift.}"
  local server_address="${SERVER_ADDRESS:-216.81.248.32:8080}"

  eval3_log "Starting robot client for ${server_address}"
  eval3_log "Server-side policy path: ${server_policy_path}"
  eval3_log "Task: ${task}"

  local robot_extra_args=()
  if [ -n "${ROBOT_MAX_RELATIVE_TARGET:-}" ]; then
    robot_extra_args+=(--robot.max_relative_target="${ROBOT_MAX_RELATIVE_TARGET}")
  fi

  exec "${PYTHON_BIN}" -m lerobot.async_inference.robot_client \
    --robot.type="${ROBOT_TYPE:-so101_follower}" \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${ROBOT_ID:-None}" \
    --robot.calibration_dir="${ROBOT_CALIBRATION_DIR:-robot/calibration/so101_group47/follower}" \
    --robot.disable_torque_on_disconnect="${ROBOT_DISABLE_TORQUE_ON_DISCONNECT:-false}" \
    --robot.cameras="${camera_config}" \
    "${robot_extra_args[@]}" \
    --server_address="${server_address}" \
    --policy_type="${POLICY_TYPE:-smolvla}" \
    --pretrained_name_or_path="${server_policy_path}" \
    --policy_device="${POLICY_DEVICE:-cuda}" \
    --client_device="${CLIENT_DEVICE:-cpu}" \
    --actions_per_chunk="${ACTIONS_PER_CHUNK:-30}" \
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD:-0.2}" \
    --aggregate_fn_name="${AGGREGATE_FN_NAME:-weighted_average}" \
    --fps="${FPS:-30}" \
    --episode_time_s="${EPISODE_TIME_S:-30}" \
    --reset_time_s="${RESET_TIME_S:-15}" \
    --task="${task}" \
    "$@"
}

eval3_print_paths() {
  printf 'CHECKPOINT_ROOT=%s\n' "${CHECKPOINT_ROOT}"
  printf 'EVAL3_COTRAIN_DIR=%s\n' "${EVAL3_COTRAIN_DIR}"
  printf 'EVAL3_CELEB_LORA_DIR=%s\n' "${EVAL3_CELEB_LORA_DIR}"
  printf 'EVAL3_E2E_POLICY_DIR=%s\n' "${EVAL3_E2E_POLICY_DIR}"
  printf 'EVAL3_BASE_MODEL_DIR=%s\n' "${EVAL3_BASE_MODEL_DIR}"
}
