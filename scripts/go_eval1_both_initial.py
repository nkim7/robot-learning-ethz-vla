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


def make_follower():
    cfg = SOFollowerConfig(
        port=FOLLOWER_PORT,
        cameras={},
        use_degrees=True,
    )

    # Compatibility with your local LeRobot version
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

    # Compatibility with your local LeRobot version
    cfg.id = None
    cfg.calibration_dir = LEADER_CALIBRATION_DIR

    return SOLeader(cfg)


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
        print("Connecting follower and leader...")
        follower.connect()
        leader.connect()

        print("")
        print("Moving FOLLOWER to fixed Eval 1 initial pose:")
        print(target)

        for _ in range(45):
            follower.send_action(target)
            time.sleep(1 / 15)

        print("")
        print("Moving LEADER to matching teleop initial pose:")
        print(target)

        # Leader is also a motor bus, so command it through its bus.
        # First try LeRobot's send_action-style API if available.
        if hasattr(leader, "send_action"):
            for _ in range(45):
                leader.send_action(target)
                time.sleep(1 / 15)
        else:
            # Fallback: write calibrated goal positions directly through the bus.
            for _ in range(45):
                for name, value in target.items():
                    motor = name.replace(".pos", "")
                    leader.bus.write("Goal_Position", motor, value)
                time.sleep(1 / 15)

        print("")
        print("Disabling leader torque so it is free for teleoperation...")
        try:
            leader.bus.disable_torque()
        except Exception as e:
            print(f"Warning: could not disable leader torque cleanly: {e}")

        print("")
        print("Done.")
        print("Follower is fixed at initial pose.")
        print("Leader is matched to the same pose and torque-disabled.")
        print("Now start recording.")

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
