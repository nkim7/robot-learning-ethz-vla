#!/usr/bin/env bash
# Run Stage 1 checkpoint on robot for evaluation
# Usage: bash eval_stage1.sh [left|middle|right]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

[[ -f .env ]] && source .env

PROMPT_ARG="${1:-left}"
case "$PROMPT_ARG" in
  left)   TASK="Put the banana into the left bowl from the robot perspective." ;;
  middle) TASK="Put the banana into the middle bowl from the robot perspective." ;;
  right)  TASK="Put the banana into the right bowl from the robot perspective." ;;
  *)      echo "Usage: $0 [left|middle|right]"; exit 1 ;;
esac

[[ -n "${FOLLOWER_PORT:-}" ]] || { echo "ERROR: FOLLOWER_PORT not set."; exit 1; }
CKPT="${CKPT:-/tmp/stage1_10k_checkpoint}"
DATASET="robot-learning-group47/eval_stage1_10k_smoke"

echo "Prompt: $TASK  |  Port: $FOLLOWER_PORT  |  Ckpt: $CKPT"
rm -rf "$HOME/.cache/huggingface/lerobot/$DATASET"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=my_follower \
  --robot.calibration_dir="$REPO_ROOT/robot/calibration/so101_group47/follower" \
  --robot.cameras='{"camera1":{"type":"opencv_crop","index_or_path":0,"source_width":640,"source_height":480,"roi_x":215,"roi_y":0,"roi_size":425,"width":256,"height":256,"fps":15,"warmup_s":1}}' \
  --policy.path="$CKPT" \
  --dataset.rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --dataset.repo_id="$DATASET" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=3
