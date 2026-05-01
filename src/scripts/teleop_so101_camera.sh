#!/usr/bin/env zsh

# SO-101 leader-follower teleoperation with robot camera monitoring.
#
# Run from repository root:
#   ./src/scripts/teleop_so101_camera.sh
#
# Requirements:
#   1. .venv exists
#   2. .env contains LEADER_PORT and FOLLOWER_PORT
#   3. shared calibration files exist in calibration/so101_group47/
#   4. LeRobot camera stability patch is applied if using macOS/OpenCV camera

set -e

source .venv/bin/activate

if [ ! -f ".env" ]; then
  echo "Missing .env file."
  echo ""
  echo "Create one like this:"
  echo "export LEADER_PORT=/dev/tty.usbmodemXXXX"
  echo "export FOLLOWER_PORT=/dev/tty.usbmodemYYYY"
  echo ""
  echo "Use this to identify ports:"
  echo "cd lerobot && lerobot-find-port"
  exit 1
fi

source .env

echo "Leader port:   $LEADER_PORT"
echo "Follower port: $FOLLOWER_PORT"

if [ ! -e "$LEADER_PORT" ]; then
  echo "Leader port does not exist: $LEADER_PORT"
  echo "Run: cd lerobot && lerobot-find-port"
  exit 1
fi

if [ ! -e "$FOLLOWER_PORT" ]; then
  echo "Follower port does not exist: $FOLLOWER_PORT"
  echo "Run: cd lerobot && lerobot-find-port"
  exit 1
fi

if ! grep -q "Patched for robot-learning-ethz-vla" lerobot/src/lerobot/cameras/opencv/camera_opencv.py; then
  echo "WARNING: LeRobot camera stability patch may not be applied."
  echo "If teleop crashes on camera frame drops, apply the patch in patches/."
fi

cd lerobot

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=follower_arm \
  --robot.calibration_dir=../calibration/so101_group47/follower \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 3}}' \
  --teleop.type=so101_leader \
  --teleop.port=$LEADER_PORT \
  --teleop.id=leader_arm \
  --teleop.calibration_dir=../calibration/so101_group47/leader \
  --fps=15 \
  --display_data=true
