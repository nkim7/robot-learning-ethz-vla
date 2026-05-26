#!/usr/bin/env zsh

# Apply LeRobot OpenCV camera stability patch.
# Run from repository root:
#   ./src/scripts/apply_lerobot_camera_patch.sh

set -e

PATCH="robot/patches/lerobot_opencv_camera_no_crash.patch"
TARGET="lerobot/src/lerobot/cameras/opencv/camera_opencv.py"

if [ ! -f "$PATCH" ]; then
  echo "Missing patch file: $PATCH"
  exit 1
fi

if grep -q "Patched for robot-learning-ethz-vla" "$TARGET"; then
  echo "Patch already appears to be applied."
  exit 0
fi

git -C lerobot apply "../$PATCH"

echo "Applied LeRobot camera stability patch."
