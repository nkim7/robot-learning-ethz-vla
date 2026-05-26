#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/eval3_runtime_common.sh"

usage() {
  cat <<'EOF'
Usage:
  ./eval_3/run_eval_3_celeb.sh setup
  ./eval_3/run_eval_3_celeb.sh server
  ./eval_3/run_eval_3_celeb.sh client
  ./eval_3/run_eval_3_celeb.sh paths

Eval 3 task 3 uses:
  - robot-learning-group47/eval3_phase2_celeb100 for celebrity/slot detection
  - robot-learning-group47/eval3_cotrain cycle_05_action_checkpoint as action expert
  - the same first-frame preprompt/rewrite server patch as tasks 1/2

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
    eval3_prepare_celeb_assets
    eval3_print_paths
    ;;
  server)
    eval3_prepare_celeb_assets
    eval3_run_preprompt_server "${EVAL3_CELEB_LORA_DIR}" "$@"
    ;;
  client)
    eval3_run_client "${SERVER_POLICY_PATH:-$(eval3_cotrain_action_checkpoint)}" "$@"
    ;;
  paths)
    eval3_print_paths
    printf 'ACTION_CHECKPOINT=%s\n' "$(eval3_cotrain_action_checkpoint)"
    printf 'CELEB_LORA=%s\n' "${EVAL3_CELEB_LORA_DIR}"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
