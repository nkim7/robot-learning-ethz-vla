#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="outputs/train/eval3_run1_action_expert_frozen_vlm_lora/checkpoints"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
  echo "ERROR: checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

latest="$(
  find "${CHECKPOINT_DIR}" -type d -name "pretrained_model" -printf "%T@ %p\n" | sort -n | tail -1
)"

if [ -z "${latest}" ]; then
  echo "ERROR: no pretrained_model checkpoint found under ${CHECKPOINT_DIR}" >&2
  exit 1
fi

printf '%s\n' "${latest}" | awk '{print $2}'
