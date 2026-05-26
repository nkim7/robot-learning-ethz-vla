#!/usr/bin/env python3
"""
Quick Stage 1 evaluator — loads model ONCE, runs left / middle / right.

Usage:
    python eval_stage1_quick.py --ckpt /tmp/stage1_10k_checkpoint
    python eval_stage1_quick.py --ckpt /tmp/stage1_12k_checkpoint
"""
import argparse
import logging
import os
import time
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSE_FILE = _REPO_ROOT / "robot/config/poses/eval1_follower_initial_pose.yaml"

from lerobot.cameras.opencv_crop import OpenCVCropCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.robot_utils import precise_sleep as busy_wait
from lerobot.utils.device_utils import get_safe_torch_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

PROMPTS = [
    ("left",   "Put the banana into the left bowl from the robot perspective."),
    ("middle", "Put the banana into the middle bowl from the robot perspective."),
    ("right",  "Put the banana into the right bowl from the robot perspective."),
]

RESET_POSE = yaml.safe_load(_POSE_FILE.read_text())["follower_pose"]

FPS             = 15
EPISODE_TIME_S  = 20
RESET_DURATION_S = 5.0
RENAME_MAP = {"observation.images.camera1": "observation.images.front"}


def go_to_reset_pose(robot, fps=FPS, duration_s=RESET_DURATION_S):
    print("  → resetting pose …")
    try:
        obs = robot.get_observation()
        start = {k: float(obs[k]) for k in RESET_POSE if k in obs}
    except Exception:
        start = dict(RESET_POSE)
    dt = 1.0 / fps
    steps = max(1, int(duration_s / dt))
    t0 = time.perf_counter()
    for i in range(steps):
        alpha = (i + 1) / steps
        action = {k: start.get(k, RESET_POSE[k]) + alpha * (RESET_POSE[k] - start.get(k, RESET_POSE[k]))
                  for k in RESET_POSE}
        robot.send_action(action)
        elapsed = time.perf_counter() - t0
        busy_wait((i + 1) * dt - elapsed)


def run_episode(robot, policy, preprocessor, postprocessor, task, fps=FPS, episode_time_s=EPISODE_TIME_S):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = get_safe_torch_device(policy.config.device)
    dt = 1.0 / fps
    steps = int(episode_time_s * fps)
    t0 = time.perf_counter()

    for step in range(steps):
        t_step = time.perf_counter()

        obs = robot.get_observation()
        # Rename camera1 → front to match policy
        for old, new in RENAME_MAP.items():
            if old in obs:
                obs[new] = obs.pop(old)
        obs["task"] = task

        action_values = predict_action(
            observation=preprocessor(obs),
            policy=policy,
            device=device,
            use_amp=False,
        )
        robot.send_action(postprocessor(action_values))

        busy_wait(dt - (time.perf_counter() - t_step))

    print("  ✓ episode finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", default=None,
                        help="Serial port (default: $FOLLOWER_PORT env var, then auto-detect /dev/tty.usbmodem*)")
    parser.add_argument("--calib", default=None,
                        help="Calibration dir (default: robot/calibration/so101_group47/follower)")
    args = parser.parse_args()

    # Resolve port
    port = args.port or os.environ.get("FOLLOWER_PORT")
    if not port:
        import glob
        ports = glob.glob("/dev/tty.usbmodem*")
        if not ports:
            raise RuntimeError("No /dev/tty.usbmodem* found. Set FOLLOWER_PORT env var.")
        port = ports[0]
        print(f"Auto-detected port: {port}")

    # Resolve calibration dir
    calib = args.calib or str(_REPO_ROOT / "robot/calibration/so101_group47/follower")

    ckpt = Path(args.ckpt).resolve()
    assert ckpt.exists(), f"Not found: {ckpt}"

    # ── Load policy (once) ────────────────────────────────────────
    print(f"\nLoading policy from {ckpt.name} …")
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_cfg.pretrained_path = str(ckpt)

    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    print("Policy loaded ✓")

    # ── Connect robot (once) ──────────────────────────────────────
    print("Connecting robot …")
    robot_cfg = SOFollowerRobotConfig(
        port=port,
        id="my_follower",
        calibration_dir=calib,
        cameras={
            "camera1": OpenCVCropCameraConfig(
                index_or_path=0,
                source_width=640, source_height=480,
                roi_x=215, roi_y=0, roi_size=425,
                width=256, height=256,
                fps=FPS, warmup_s=1,
            )
        }
    )
    robot = SOFollower(robot_cfg)
    robot.connect()
    print("Robot connected ✓\n")

    # ── Run left / middle / right ─────────────────────────────────
    results = {}
    for side, task in PROMPTS:
        print(f"{'='*55}")
        print(f"  PROMPT: {side.upper()}")
        print(f"  '{task}'")
        print(f"{'='*55}")
        input("  Place banana, then press Enter …")
        go_to_reset_pose(robot)
        run_episode(robot, policy, preprocessor, postprocessor, task)
        ans = input("  Success? (y/n): ").strip().lower()
        results[side] = ans == "y"

    robot.disconnect()

    print(f"\n{'='*55}")
    print(f"  RESULTS — {ckpt.name}")
    print(f"{'='*55}")
    for side, ok in results.items():
        print(f"  {side:6s}  {'✓ pass' if ok else '✗ fail'}")
    total = sum(results.values())
    print(f"  ──────────────────")
    print(f"  {total}/3 passed")
    print()


if __name__ == "__main__":
    main()
