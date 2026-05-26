#!/usr/bin/env python3
"""Eval2 prompt sweep that loads the model once.

This intentionally mirrors the simple one-off robot inference path: camera1 is
renamed to the policy's front camera, the policy/pre/post processors are reset
per rollout, and no layout/permutation logic is injected.

Usage:
    python run_eval2_pathC_20prompts.py --policy_path /tmp/eval2_stage2_20k_checkpoint
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from lerobot.cameras.opencv_crop import OpenCVCropCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep as busy_wait


FPS = 15
EPISODE_TIME_S = 20
RESET_DURATION_S = 5.0
RENAME_MAP = {"observation.images.camera1": "observation.images.front"}

RESET_POSE = {
    "shoulder_pan.pos": -1.6703296703296704,
    "shoulder_lift.pos": -102.02197802197803,
    "elbow_flex.pos": 92.96703296703296,
    "wrist_flex.pos": 70.1978021978022,
    "wrist_roll.pos": 7.34065934065934,
    "gripper.pos": 2.7184466019417477,
}

PROMPTS = [
    "Put the banana into the 1st bowl from the left from the robot perspective.",
    "Put the banana into the 2nd bowl from the left from the robot perspective.",
    "Put the banana into the 3rd bowl from the left from the robot perspective.",
    "Put the banana into the 1st bowl from the right from the robot perspective.",
    "Put the banana into the 2nd bowl from the right from the robot perspective.",
    "Put the banana into the 3rd bowl from the right from the robot perspective.",
    "Put the banana into the middle bowl from the robot perspective.",
    "Put the banana into the center bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the red bowl from the robot perspective.",
    "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the green bowl from the robot perspective.",
    "Put the banana into the bowl on the right of the green bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the blue bowl from the robot perspective.",
    "Put the banana into the bowl on the right of the blue bowl from the robot perspective.",
    "Put the banana into the bowl that is not red and not green.",
    "Put the banana into the bowl that is not red and not blue.",
    "Put the banana into the bowl that is not green and not blue.",
    "Put the banana into the bowl that is not green and not red.",
    "Put the banana into the bowl that is not blue and not red.",
    "Put the banana into the bowl that is not blue and not green.",
]


def go_to_reset_pose(robot) -> None:
    print("  resetting pose ...")
    try:
        obs = robot.get_observation()
        start = {k: float(obs[k]) for k in RESET_POSE if k in obs}
    except Exception:
        start = dict(RESET_POSE)

    dt = 1.0 / FPS
    steps = max(1, int(RESET_DURATION_S / dt))
    t0 = time.perf_counter()
    for i in range(steps):
        alpha = (i + 1) / steps
        action = {
            k: start.get(k, RESET_POSE[k]) + alpha * (RESET_POSE[k] - start.get(k, RESET_POSE[k]))
            for k in RESET_POSE
        }
        robot.send_action(action)
        busy_wait((i + 1) * dt - (time.perf_counter() - t0))


def run_episode(robot, policy, preprocessor, postprocessor, task: str) -> None:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = get_safe_torch_device(policy.config.device)
    dt = 1.0 / FPS
    steps = int(EPISODE_TIME_S * FPS)

    for _ in range(steps):
        t_step = time.perf_counter()
        obs = robot.get_observation()
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


def print_plan(policy_path: str) -> None:
    print("\nEval2 20-prompt sweep")
    print(f"Policy: {policy_path}")
    print("Model and robot load once. No layout/permutation logic is applied.\n")
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"{i:2d}. {prompt}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default="my_follower")
    parser.add_argument("--calibration_dir", default="calibration/so101_group47/follower")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--start_from", type=int, default=1)
    args = parser.parse_args()

    if args.port is None:
        ports = glob.glob("/dev/tty.usbmodem*")
        if not ports:
            print("ERROR: no /dev/tty.usbmodem* found.")
            sys.exit(1)
        args.port = ports[0]

    print_plan(args.policy_path)
    input("Press Enter to load model and connect robot, Ctrl+C to abort...")

    ckpt = Path(args.policy_path).resolve()
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_cfg.pretrained_path = str(ckpt)

    print(f"\nLoading policy from {ckpt} ...")
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
    print("Policy loaded.")

    robot_cfg = SOFollowerRobotConfig(
        port=args.port,
        id=args.id,
        calibration_dir=args.calibration_dir,
        cameras={
            "camera1": OpenCVCropCameraConfig(
                index_or_path=args.camera_index,
                source_width=640,
                source_height=480,
                roi_x=215,
                roi_y=0,
                roi_size=425,
                width=256,
                height=256,
                fps=FPS,
                warmup_s=1,
            )
        },
    )
    robot = SOFollower(robot_cfg)
    robot.connect()
    print("Robot connected.")

    results = []
    try:
        for i, prompt in enumerate(PROMPTS, start=1):
            if i < args.start_from:
                continue
            print("\n" + "=" * 72)
            print(f"PROMPT {i}/{len(PROMPTS)}")
            print(prompt)
            print("=" * 72)
            input("Place banana/bowls, then press Enter to run...")
            go_to_reset_pose(robot)
            run_episode(robot, policy, preprocessor, postprocessor, prompt)
            ans = input("Success? [y/n]: ").strip().lower()
            results.append((prompt, ans == "y"))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")

    if results:
        ok = sum(success for _, success in results)
        print(f"\nScore: {ok}/{len(results)}")
        for prompt, success in results:
            print(f"{'PASS' if success else 'FAIL'}  {prompt}")


if __name__ == "__main__":
    main()
