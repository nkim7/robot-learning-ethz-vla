#!/usr/bin/env bash
# Task 3 — celebrity recognition (Rihanna, Elon Musk …)
# Usage: TASK="Place the coke on Rihanna." FOLLOWER_PORT=/dev/tty.usbmodemXXXX ./run_eval_3_celeb.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v ffmpeg &>/dev/null || { [[ "$OSTYPE" == darwin* ]] && brew install ffmpeg || sudo apt-get install -y ffmpeg; }
command -v uv     &>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.cargo/bin:$PATH"; }

git submodule update --init --recursive
uv sync
source .venv/bin/activate

bash robot/apply_lerobot_camera_patch.sh

[[ -f .env ]] && source .env

MODELS_ROOT="${MODELS_ROOT:-$REPO_ROOT/models}"
ACTION_CKPT="$MODELS_ROOT/eval3"
VLM_LORA="$MODELS_ROOT/eval3/celeb"
TASK="${TASK:-Place the coke on Rihanna.}"
FOLLOWER_PORT="${FOLLOWER_PORT:-}"
DEVICE="${DEVICE:-mps}"
NUM_EPISODES="${NUM_EPISODES:-1}"

[[ -n "$FOLLOWER_PORT" ]]                           || { echo "ERROR: FOLLOWER_PORT not set. Run: lerobot-find-port"; exit 1; }
[[ -f "$ACTION_CKPT/model.safetensors" ]]           || { echo "Models not found. Downloading..."; bash scripts/download_models.sh; }
[[ -f "$VLM_LORA/adapter_model.safetensors" ]]      || { echo "VLM LoRA not found. Downloading..."; bash scripts/download_models.sh; }

rm -rf ~/.cache/huggingface/lerobot/local/eval_3_celeb_run

echo "Resetting follower to initial pose..."
python robot/reset_from_eval1_pose.py

echo "Task: $TASK  |  Port: $FOLLOWER_PORT  |  Device: $DEVICE"

PYTHONPATH="$REPO_ROOT/lerobot/src:$REPO_ROOT/models/eval3/runtime:${PYTHONPATH:-}" \
EVAL3_VLM_LORA_PATH="$VLM_LORA" \
EVAL3_EXPECTED_VLM_BASE="HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
EVAL3_VLM_LORA_TRAINABLE=0 \
EVAL3_FIRST_FRAME_PROBE=1 \
EVAL3_INSTRUCTION_REWRITE=1 \
python "$REPO_ROOT/models/eval3/runtime/eval_3_runner.py" \
    --robot.type=so101_follower \
    --robot.id=my_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.calibration_dir="$REPO_ROOT/robot/calibration/so101_group47/follower" \
    --robot.cameras='{"front":{"type":"opencv_crop","index_or_path":0,"source_width":640,"source_height":480,"roi_x":215,"roi_y":0,"roi_size":425,"width":256,"height":256,"fps":15,"warmup_s":1}}' \
    --policy.path="$ACTION_CKPT" \
    --policy.device="$DEVICE" \
    --dataset.repo_id=local/eval_3_celeb_run \
    --dataset.push_to_hub=false \
    --dataset.fps=15 \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$TASK" \
    --display_data=true
