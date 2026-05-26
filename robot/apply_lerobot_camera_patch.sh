#!/usr/bin/env zsh

# Apply LeRobot OpenCV camera stability patch.
# Run from repository root:
#   ./robot/apply_lerobot_camera_patch.sh

set -e

PATCH="robot/patches/lerobot_opencv_camera_no_crash.patch"
TARGET="lerobot/src/lerobot/cameras/opencv/camera_opencv.py"

if [ ! -f "$PATCH" ]; then
  echo "Missing patch file: $PATCH"
  exit 1
fi

# --- 1. Crash-prevention patch ---
if grep -q "Patched for robot-learning-ethz-vla" "$TARGET"; then
  echo "Crash-prevention patch already applied."
else
  git -C lerobot apply "../$PATCH"
  echo "Applied LeRobot camera stability patch."
fi

# --- 2. Copy opencv_crop package into the submodule if missing ---
# The opencv_crop camera type is not part of the upstream LeRobot release.
# We ship it in robot/patches/opencv_crop/ and copy it on first run.
OPENCV_CROP_SRC="robot/patches/opencv_crop"
OPENCV_CROP_DST="lerobot/src/lerobot/cameras/opencv_crop"
if [ -d "$OPENCV_CROP_DST" ]; then
  echo "opencv_crop package already present."
else
  cp -r "$OPENCV_CROP_SRC" "$OPENCV_CROP_DST"
  echo "Copied opencv_crop package into lerobot cameras."
fi

# --- 3. Register opencv_crop type so draccus can parse --robot.cameras='{"type":"opencv_crop",...}' ---
# The @CameraConfig.register_subclass("opencv_crop") decorator only runs on import, so we add the
# import to cameras/__init__.py without touching the submodule's git history.
CAMERAS_INIT="lerobot/src/lerobot/cameras/__init__.py"
MARKER="from .opencv_crop import OpenCVCropCameraConfig"
if ! grep -q "$MARKER" "$CAMERAS_INIT"; then
  echo "" >> "$CAMERAS_INIT"
  echo "# Register opencv_crop type for draccus CLI parsing (added by apply_lerobot_camera_patch.sh)" >> "$CAMERAS_INIT"
  echo "$MARKER  # noqa: F401" >> "$CAMERAS_INIT"
  echo "Registered OpenCVCropCameraConfig in cameras/__init__.py."
else
  echo "OpenCVCropCameraConfig already registered."
fi
