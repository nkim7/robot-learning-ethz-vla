#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

if [[ -n "${FOLLOWER_PORT:-}" ]]; then
  echo "Using FOLLOWER_PORT from env: $FOLLOWER_PORT"
else
  FOLLOWER_PORT=$(ls /dev/tty.usbmodem* 2>/dev/null | head -1)
  [[ -n "$FOLLOWER_PORT" ]] || { echo "ERROR: no /dev/tty.usbmodem* device found."; exit 1; }
  echo "Auto-detected port: $FOLLOWER_PORT"
fi

FOLLOWER_ID="${FOLLOWER_ID:-my_follower}"
CALIB_DIR="${CALIB_DIR:-$REPO_ROOT/robot/calibration/so101_group47/follower}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
POLICY="${POLICY:-$REPO_ROOT/models/eval2_2phase_resolvedcolor_016000/checkpoints/016000}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

reset_pose() {
  echo "  → resetting arm to initial pose..."
  python "$REPO_ROOT/robot/reset_from_eval1_pose.py" \
    --port "$FOLLOWER_PORT" --id "$FOLLOWER_ID" --calibration_dir "$CALIB_DIR"
}

run_trial() {
  local COLOR="$1" TASK="$2" LABEL="$3"
  local REPO="robot-learning-group47/eval_eval2_resolvedcolor_016000_${LABEL}"
  echo ""; echo "Color: $COLOR | Task: $TASK"; read -rp "Place banana and bowls, then press ENTER."
  reset_pose; sleep 1
  rm -rf "$HOME/.cache/huggingface/lerobot/${REPO}" 2>/dev/null || true
  lerobot-record \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
    --robot.calibration_dir="$CALIB_DIR" \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 15}}" \
    --dataset.repo_id="$REPO" --dataset.single_task="$TASK" \
    --dataset.num_episodes=1 --dataset.episode_time_s=20 --dataset.reset_time_s=10 \
    --dataset.fps=15 --dataset.push_to_hub="$PUSH_TO_HUB" \
    --policy.path="$POLICY" --policy.n_action_steps=50 --display_data=false
  echo "  ✓ done: $LABEL"
}

run_trial "blue"  "Put the banana in the blue bowl."  "blue"
run_trial "green" "Put the banana in the green bowl." "green"
run_trial "red"   "Put the banana in the red bowl."   "red"
