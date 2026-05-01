#TODO: Code for wrapping teleoperation function from lerobot#!/usr/bin/env python3

"""
SO-101 leader-follower teleoperation with robot camera monitoring.

Run from repository root:

    python src/scripts/teleop_camera.py

or, if a Makefile target exists:

    make teleop-camera

Required local .env:

    export LEADER_PORT=/dev/tty.usbmodemXXXX
    export FOLLOWER_PORT=/dev/tty.usbmodemYYYY
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find the repository root from this script location."""
    here = Path(__file__).resolve()

    for parent in [here.parent, *here.parents]:
        if (parent / "lerobot").exists() and (parent / "calibration").exists():
            return parent

    raise RuntimeError("Could not find repo root containing lerobot/ and calibration/.")


REPO_ROOT = find_repo_root()
LEROBOT_DIR = REPO_ROOT / "lerobot"

LEADER_ID = "leader_arm"
FOLLOWER_ID = "follower_arm"

LEADER_CALIBRATION_DIR = "../calibration/so101_group47/leader"
FOLLOWER_CALIBRATION_DIR = "../calibration/so101_group47/follower"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_S = 3

TELEOP_FPS = 15


def load_env_file(env_path: Path) -> None:
    """Load simple export KEY=VALUE lines from .env."""
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing {env_path}\n\n"
            "Create it with:\n"
            "  export LEADER_PORT=/dev/tty.usbmodemXXXX\n"
            "  export FOLLOWER_PORT=/dev/tty.usbmodemYYYY\n\n"
            "Find ports with:\n"
            "  cd lerobot && lerobot-find-port"
        )

    for line in env_path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :]

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def require_port(name: str) -> str:
    """Read and validate a serial port from the environment."""
    port = os.environ.get(name, "")

    if not port:
        raise RuntimeError(f"{name} is not set in .env")

    if not Path(port).exists():
        raise RuntimeError(
            f"{name} does not exist: {port}\n\n"
            "Robot USB port may have changed. Re-run:\n"
            "  cd lerobot && lerobot-find-port\n"
            "Then update .env."
        )

    return port


def find_lerobot_teleoperate() -> str:
    """Find lerobot-teleoperate from active env, repo .venv, or PATH."""
    candidates = []

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(Path(virtual_env) / "bin" / "lerobot-teleoperate")

    candidates.append(REPO_ROOT / ".venv" / "bin" / "lerobot-teleoperate")

    from_path = shutil.which("lerobot-teleoperate")
    if from_path:
        candidates.append(Path(from_path))

    # Legacy fallback, in case the old nested folder still exists.
    candidates.append(
        REPO_ROOT
        / "robot-learning-ethz-vla"
        / ".venv"
        / "bin"
        / "lerobot-teleoperate"
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "Could not find lerobot-teleoperate.\n\n"
        "Try one of these:\n"
        "  source .venv/bin/activate\n"
        "  uv sync --python 3.12\n"
        "  which lerobot-teleoperate\n"
    )


def main() -> int:
    load_env_file(REPO_ROOT / ".env")

    leader_port = require_port("LEADER_PORT")
    follower_port = require_port("FOLLOWER_PORT")
    teleoperate_bin = find_lerobot_teleoperate()

    print(f"Repo root:      {REPO_ROOT}")
    print(f"LeRobot dir:    {LEROBOT_DIR}")
    print(f"Teleop binary:  {teleoperate_bin}")
    print(f"Leader port:    {leader_port}")
    print(f"Follower port:  {follower_port}")
    print(f"Robot camera:   OpenCV index {CAMERA_INDEX}")
    print(f"Camera mode:    {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
    print(f"Teleop FPS:     {TELEOP_FPS}")

    camera_config = (
        "{ front: "
        f"{{type: opencv, index_or_path: {CAMERA_INDEX}, "
        f"width: {CAMERA_WIDTH}, height: {CAMERA_HEIGHT}, "
        f"fps: {CAMERA_FPS}, warmup_s: {CAMERA_WARMUP_S}}}"
        "}"
    )

    command = [
        teleoperate_bin,
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={FOLLOWER_ID}",
        f"--robot.calibration_dir={FOLLOWER_CALIBRATION_DIR}",
        f"--robot.cameras={camera_config}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={LEADER_ID}",
        f"--teleop.calibration_dir={LEADER_CALIBRATION_DIR}",
        f"--fps={TELEOP_FPS}",
        "--display_data=true",
    ]

    return subprocess.call(command, cwd=LEROBOT_DIR)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
