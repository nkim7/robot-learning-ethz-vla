#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate

SKIP="12 19"
STEP=3  # send every Nth frame — increase to go faster

python3 - <<'PYEOF'
import sys, time, os, subprocess
import pyarrow.parquet as pq
import numpy as np

sys.path.insert(0, "lerobot/src")
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor import make_default_robot_action_processor

SKIP = set()
STEP = int(os.environ.get("STEP", 1))
PORT = "/dev/tty.usbmodem5AE60831701"
CALIB = "calibration/so101_group47/follower"
SNAP = (
    os.path.expanduser("~") +
    "/.cache/huggingface/hub/"
    "datasets--robot-learning-group47--eval1_60_fixed_banana_fixed_bowl/"
    "snapshots/a3faa7fc42fc73f47b6c8358a20a0e09e87be40d"
)
import glob
files = sorted(glob.glob(f"{SNAP}/data/chunk-000/file-*.parquet"))

from pathlib import Path
robot_cfg = SOFollowerRobotConfig(
    port=PORT,
    id="my_follower",
    calibration_dir=Path(CALIB),
    use_degrees=True,
)
robot = SOFollower(robot_cfg)
processor = make_default_robot_action_processor()
robot.connect()

NAMES = ["shoulder_pan.pos","shoulder_lift.pos","elbow_flex.pos",
         "wrist_flex.pos","wrist_roll.pos","gripper.pos"]

try:
    for f in files[0:]:
        import pandas as pd
        df = pq.read_table(f).to_pandas()
        ep = int(df["episode_index"].iloc[0])
        if ep in SKIP:
            continue

        subprocess.run(["say", f"Episode {ep}"])

        actions = np.vstack(df["action"].values)
        for i in range(0, len(actions)):
            if i % 3 == 2:  # skip every 3rd frame = 1.5x faster
                continue
            t0 = time.perf_counter()
            action = {name: float(actions[i, j]) for j, name in enumerate(NAMES)}
            obs = robot.get_observation()
            processed = processor((action, obs))
            robot.send_action(processed)
            precise_sleep(max(1/30 - (time.perf_counter() - t0), 0))
finally:
    robot.disconnect()
PYEOF
