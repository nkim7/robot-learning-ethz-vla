import glob
import os
import time
import yaml
import argparse
from pathlib import Path

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

parser = argparse.ArgumentParser()
parser.add_argument("--pose_file", default=str(Path(__file__).parent / "config/poses/eval1_follower_initial_pose.yaml"))
parser.add_argument("--port", default=None,
                    help="Serial port (default: $FOLLOWER_PORT env var, then auto-detect /dev/tty.usbmodem*)")
parser.add_argument("--duration", type=float, default=3.0)
parser.add_argument("--hold", type=float, default=0.8)
parser.add_argument("--hz", type=float, default=30.0)
args = parser.parse_args()

with open(args.pose_file) as f:
    pose_cfg = yaml.safe_load(f)

target_pose = {k: float(v) for k, v in pose_cfg["follower_pose"].items()}

# Resolve port: CLI arg → FOLLOWER_PORT env var → auto-detect
port = args.port or os.environ.get("FOLLOWER_PORT")
if not port:
    _detected = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not _detected:
        raise RuntimeError("No /dev/tty.usbmodem* found. Set FOLLOWER_PORT env var.")
    port = _detected[0]
    print(f"Auto-detected port: {port}")

calibration_dir = Path(pose_cfg["calibration"]["follower"]).resolve()

cfg = SO101FollowerConfig(
    port=port,
    use_degrees=True,
    disable_torque_on_disconnect=False,
)
cfg.id = "chengming"
cfg.calibration_dir = calibration_dir

robot = SOFollower(cfg)
robot.connect()

try:
    obs = robot.get_observation()
    print("Current observation:")
    for k in target_pose:
        print(k, obs.get(k))

    start_pose = {
        k: float(obs[k]) if k in obs else float(v)
        for k, v in target_pose.items()
    }
except Exception as e:
    print("Could not read current pose. Using target as start.", e)
    start_pose = target_pose.copy()

steps = max(1, int(args.duration * args.hz))

print("Resetting robot smoothly...")

for i in range(1, steps + 1):
    alpha = i / steps
    action = {
        k: start_pose[k] * (1.0 - alpha) + target_pose[k] * alpha
        for k in target_pose
    }
    robot.send_action(action)
    time.sleep(1.0 / args.hz)

print("Holding target pose...")

for _ in range(max(1, int(args.hold * args.hz))):
    robot.send_action(target_pose)
    time.sleep(1.0 / args.hz)

robot.disconnect()
print("Reset complete.")
