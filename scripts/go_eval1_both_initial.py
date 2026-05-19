#!/usr/bin/env python3
from pathlib import Path
import time
import yaml

from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig

POSE_PATH = Path("config/initial/eval1_follower_initial_pose.yaml")

FOLLOWER_PORT = "/dev/tty.usbmodem5B141138561"
LEADER_PORT = "/dev/tty.usbmodem5B141126211"

FOLLOWER_CALIBRATION_DIR = Path("calibration/so101_group47/follower")
LEADER_CALIBRATION_DIR = Path("calibration/so101_group47/leader")

CONTROL_HZ = 15
MOVE_STEPS = 15   # 15 = ~1 second. Use 20 if movement is too abrupt.


def make_follower():
    cfg = SOFollowerConfig(
        port=FOLLOWER_PORT,
        cameras={},
        use_degrees=True,
    )

    cfg.id = None
    cfg.calibration_dir = FOLLOWER_CALIBRATION_DIR
    cfg.disable_torque_on_disconnect = False
    cfg.max_relative_target = None

    return SOFollower(cfg)


def make_leader():
    cfg = SOLeaderConfig(
        port=LEADER_PORT,
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

    target = {
        "shoulder_pan.pos": pose["shoulder_pan.pos"],
        "shoulder_lift.pos": pose["shoulder_lift.pos"],
        "elbow_flex.pos": pose["elbow_flex.pos"],
        "wrist_flex.pos": pose["wrist_flex.pos"],
        "wrist_roll.pos": pose["wrist_roll.pos"],
        "gripper.pos": pose["gripper.pos"],
    }

    follower = make_follower()
    leader = make_leader()

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