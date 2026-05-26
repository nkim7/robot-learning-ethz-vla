#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

command -v ffmpeg &>/dev/null || { [[ "$OSTYPE" == darwin* ]] && brew install ffmpeg || sudo apt-get install -y ffmpeg; }
command -v uv     &>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.cargo/bin:$PATH"; }

git submodule update --init --recursive
uv sync
source .venv/bin/activate

bash robot/apply_lerobot_camera_patch.sh

[[ -f .env ]] && source .env
[[ -n "${FOLLOWER_PORT:-}" ]] || { echo "ERROR: FOLLOWER_PORT not set. Run: lerobot-find-port"; exit 1; }

[[ -f "$REPO_ROOT/models/eval2/model.safetensors" ]] || {
  echo "Models not found. Downloading..."; bash scripts/download_models.sh
}

rm -rf ~/.cache/huggingface/lerobot/local/eval2_run

lerobot-record \
  --robot.type=so101_follower \
  --robot.id=my_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.calibration_dir="$REPO_ROOT/robot/calibration/so101_group47/follower" \
  --robot.cameras='{"camera1":{"type":"opencv_crop","index_or_path":0,"source_width":640,"source_height":480,"roi_x":215,"roi_y":0,"roi_size":425,"width":256,"height":256,"fps":15,"warmup_s":1}}' \
  --policy.path="$REPO_ROOT/models/eval2" \
  --dataset.rename_map='{"observation.images.camera1":"observation.images.front"}' \
  --dataset.repo_id=local/eval2_run \
  --dataset.push_to_hub=false \
  --dataset.fps=15 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Put the banana into the bowl to the right of the blue bowl." \
  --display_data=true
