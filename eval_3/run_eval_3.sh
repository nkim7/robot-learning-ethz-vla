#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/eval3_runtime_common.sh"

usage() {
  cat <<'EOF'
Usage:
  ./eval_3/run_eval_3.sh setup
  ./eval_3/run_eval_3.sh server
  ./eval_3/run_eval_3.sh client
  ./eval_3/run_eval_3.sh paths

Eval 3 tasks 1/2 use robot-learning-group47/eval3_cotrain:
  - cycle_05_action_checkpoint as the SmolVLA action expert
  - cycle_05_vlm_lora with the first-frame preprompt/rewrite server patch

Common overrides:
  CHECKPOINT_ROOT=/home/shadeform/checkpoint
  HOST=0.0.0.0 PORT=8080
  SERVER_ADDRESS=216.81.248.32:8080
  SERVER_POLICY_PATH=/home/shadeform/checkpoint/eval3_cotrain/cycle_05_action_checkpoint
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
    eval3_prepare_cotrain_assets
    eval3_print_paths
    ;;
  server)
    eval3_prepare_cotrain_assets
    eval3_run_preprompt_server "$(eval3_cotrain_vlm_lora)" "$@"
    ;;
  client)
    eval3_run_client "${SERVER_POLICY_PATH:-$(eval3_cotrain_action_checkpoint)}" "$@"
    ;;
  paths)
    eval3_print_paths
    printf 'ACTION_CHECKPOINT=%s\n' "$(eval3_cotrain_action_checkpoint)"
    printf 'VLM_LORA=%s\n' "$(eval3_cotrain_vlm_lora)"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
