#!/usr/bin/env python3
"""Move both follower and leader to the fixed initial pose.

Ports are resolved in priority order:
  1. FOLLOWER_PORT / LEADER_PORT environment variables
  2. pose YAML 'ports' section (if present)
  3. Auto-detect: first two /dev/tty.usbmodem* devices found
"""
import glob
import os
from pathlib import Path
import time
import yaml

from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig

_ROBOT_DIR = Path(__file__).parent
POSE_PATH = _ROBOT_DIR / "config/poses/eval1_follower_initial_pose.yaml"

FOLLOWER_CALIBRATION_DIR = _ROBOT_DIR / "calibration/so101_group47/follower"
LEADER_CALIBRATION_DIR = _ROBOT_DIR / "calibration/so101_group47/leader"


def _resolve_ports(pose_cfg: dict):
    """Return (follower_port, leader_port) using env vars, yaml, or auto-detect."""
    follower = os.environ.get("FOLLOWER_PORT")
    leader = os.environ.get("LEADER_PORT")
    if follower and leader:
        return follower, leader

    # Try YAML ports section as fallback
    yaml_ports = pose_cfg.get("ports", {})
    if not follower:
        follower = yaml_ports.get("follower")
    if not leader:
        leader = yaml_ports.get("leader")
    if follower and leader:
        return follower, leader

    # Auto-detect: need at least 2 usbmodem ports
    detected = sorted(glob.glob("/dev/tty.usbmodem*"))
    if len(detected) < 2:
        raise RuntimeError(
            f"Found {len(detected)} /dev/tty.usbmodem* port(s); need 2.\n"
            "Set FOLLOWER_PORT and LEADER_PORT environment variables."
        )
    follower = follower or detected[0]
    leader = leader or detected[1]
    print(f"Auto-detected ports: FOLLOWER={follower}  LEADER={leader}")
    return follower, leader

CONTROL_HZ = 15
MOVE_STEPS = 15   # 15 = ~1 second. Use 20 if movement is too abrupt.


def make_follower(port: str):
    cfg = SOFollowerConfig(
        port=port,
        cameras={},
        use_degrees=True,
    )

    cfg.id = None
    cfg.calibration_dir = FOLLOWER_CALIBRATION_DIR
    cfg.disable_torque_on_disconnect = False
    cfg.max_relative_target = None

    return SOFollower(cfg)


def make_leader(port: str):
    cfg = SOLeaderConfig(
        port=port,
        use_degrees=True,
    )

    cfg.id = None
    cfg.calibration_dir = LEADER_CALIBRATION_DIR

    return SOLeader(cfg)


def send_repeated(device, target, steps=MOVE_STEPS):
    for _ in range(steps):
        device.send_action(target)
        time.sleep(1 / CONTROL_HZ)


def send_leader_repeated(leader, target, steps=MOVE_STEPS):
    if hasattr(leader, "send_action"):
        send_repeated(leader, target, steps)
        return

    for _ in range(steps):
        for name, value in target.items():
            motor = name.replace(".pos", "")
            leader.bus.write("Goal_Position", motor, value)
        time.sleep(1 / CONTROL_HZ)


def main():
    cfg_data = yaml.safe_load(POSE_PATH.read_text())
    pose = cfg_data["follower_pose"]
    follower_port, leader_port = _resolve_ports(cfg_data)

    target = {
        "shoulder_pan.pos": pose["shoulder_pan.pos"],
        "shoulder_lift.pos": pose["shoulder_lift.pos"],
        "elbow_flex.pos": pose["elbow_flex.pos"],
        "wrist_flex.pos": pose["wrist_flex.pos"],
        "wrist_roll.pos": pose["wrist_roll.pos"],
        "gripper.pos": pose["gripper.pos"],
    }

    follower = make_follower(follower_port)
    leader = make_leader(leader_port)

    try:
        print("Resetting follower and leader to initial pose...")
        follower.connect()
        leader.connect()

        send_repeated(follower, target)
        send_leader_repeated(leader, target)

        try:
            leader.bus.disable_torque()
        except Exception as e:
            print(f"Warning: could not disable leader torque cleanly: {e}")

        print("Done. Follower fixed; leader matched and torque-disabled.")

    finally:
        try:
            leader.disconnect()
        except Exception:
            pass

        try:
            follower.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()