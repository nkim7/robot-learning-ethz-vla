#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/eval3_runtime_common.sh"

usage() {
  cat <<'EOF'
Usage:
  ./eval_3/setup_eval_3.sh all
  ./eval_3/setup_eval_3.sh cotrain
  ./eval_3/setup_eval_3.sh celeb
  ./eval_3/setup_eval_3.sh e2e
  ./eval_3/setup_eval_3.sh env
  ./eval_3/setup_eval_3.sh paths

Default:
  all

What it installs/downloads:
  env      Create .venv and install LeRobot runtime extras.
  cotrain  Download robot-learning-group47/eval3_cotrain.
  celeb    Download robot-learning-group47/eval3_phase2_celeb100.
  e2e      Download robot-learning-group47/eval3_bridge_cycle05_smolvla.
  all      Install env and download cotrain, celeb, e2e, and the SmolVLM base.

By default checkpoints go to the repository-local checkpoints/ folder:
  CHECKPOINT_ROOT=/path/to/repo/checkpoints

Use CHECKPOINT_ROOT=/home/shadeform/checkpoint to match the server paths used
during Eval 3.
EOF
}

command_name="${1:-all}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${command_name}" in
  all)
    eval3_setup_env
    eval3_snapshot_download "${EVAL3_COTRAIN_REPO_ID}" "${EVAL3_COTRAIN_DIR}"
    eval3_snapshot_download "${EVAL3_CELEB_REPO_ID}" "${EVAL3_CELEB_LORA_DIR}"
    eval3_snapshot_download "${EVAL3_E2E_REPO_ID}" "${EVAL3_E2E_POLICY_DIR}"
    eval3_download_base_model
    eval3_print_paths
    ;;
  cotrain)
    eval3_prepare_cotrain_assets
    eval3_print_paths
    ;;
  celeb)
    eval3_prepare_celeb_assets
    eval3_print_paths
    ;;
  e2e)
    eval3_prepare_e2e_assets
    eval3_print_paths
    ;;
  env)
    eval3_setup_env
    ;;
  paths)
    eval3_print_paths
    printf 'ACTION_CHECKPOINT=%s\n' "$(eval3_cotrain_action_checkpoint)"
    printf 'COTRAIN_VLM_LORA=%s\n' "$(eval3_cotrain_vlm_lora)"
    printf 'CELEB_LORA=%s\n' "${EVAL3_CELEB_LORA_DIR}"
    printf 'E2E_POLICY=%s\n' "${EVAL3_E2E_POLICY_DIR}"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
