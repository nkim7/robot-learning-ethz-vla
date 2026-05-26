#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/eval3_runtime_common.sh"

eval3_e2e_vlm_lora() {
  if [ -n "${EVAL3_E2E_VLM_LORA_DIR:-}" ]; then
    printf '%s\n' "${EVAL3_E2E_VLM_LORA_DIR}"
    return
  fi
  eval3_cotrain_vlm_lora
}

usage() {
  cat <<'EOF'
Usage:
  ./eval_3/run_eval_3_e2e.sh setup
  ./eval_3/run_eval_3_e2e.sh server
  ./eval_3/run_eval_3_e2e.sh client
  ./eval_3/run_eval_3_e2e.sh paths

Eval 3 end-to-end baseline uses robot-learning-group47/eval3_bridge_cycle05_smolvla.
It is kept separate from the preprompt/rewrite module and serves celebrity
prompts directly.

Common overrides:
  CHECKPOINT_ROOT=/home/shadeform/checkpoint
  HOST=0.0.0.0 PORT=8080
  SERVER_ADDRESS=216.81.248.32:8080
  SERVER_POLICY_PATH=/home/shadeform/checkpoint/eval3_bridge_cycle05_smolvla
  EVAL3_E2E_VLM_LORA_DIR=/home/shadeform/checkpoint/eval3_cotrain/cycle_05_vlm_lora
  FOLLOWER_PORT=/dev/tty.usbmodem...
  TASK="Place the coke on Taylor Swift."
EOF
}

command_name="${1:-help}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${command_name}" in
  setup)
    eval3_prepare_e2e_assets
    eval3_print_paths
    ;;
  server)
    eval3_prepare_e2e_assets
    eval3_run_e2e_server "$(eval3_e2e_vlm_lora)" "$@"
    ;;
  client)
    eval3_run_client "${SERVER_POLICY_PATH:-${EVAL3_E2E_POLICY_DIR}}" "$@"
    ;;
  paths)
    eval3_print_paths
    printf 'E2E_POLICY=%s\n' "${EVAL3_E2E_POLICY_DIR}"
    printf 'E2E_VLM_LORA=%s\n' "$(eval3_e2e_vlm_lora)"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
