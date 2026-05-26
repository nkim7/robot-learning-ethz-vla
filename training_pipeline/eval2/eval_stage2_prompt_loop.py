#!/usr/bin/env python3
"""Interactive Eval2 prompt loop that loads one checkpoint once.

Example:
    python eval_stage2_prompt_loop.py --ckpt /tmp/eval2_stage2_10k_checkpoint
"""

from __future__ import annotations

import argparse
import logging
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

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

DEFAULT_PROMPTS = [
    "Put the banana into the 1st bowl from the left from the robot perspective.",
    "Put the banana into the 2nd bowl from the left from the robot perspective.",
    "Put the banana into the 3rd bowl from the left from the robot perspective.",
    "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the green bowl from the robot perspective.",
    "Put the banana into the bowl that is not green and not blue.",
]


def go_to_reset_pose(robot, fps: int = FPS, duration_s: float = RESET_DURATION_S) -> None:
    print("  resetting pose ...")
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


def choose_prompt() -> str | None:
    print("\nChoose prompt:")
    for i, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        print(f"  {i}. {prompt}")
    print("  c. custom prompt")
    print("  q. quit")
    choice = input("> ").strip()
    if choice.lower() == "q":
        return None
    if choice.lower() == "c":
        return input("Custom prompt: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(DEFAULT_PROMPTS):
        return DEFAULT_PROMPTS[int(choice) - 1]
    print("Bad choice.")
    return choose_prompt()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B140319121")
    parser.add_argument("--calib", default="calibration/so101_group47/follower")
    args = parser.parse_args()

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    print(f"Loading policy once from {ckpt} ...")
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
    print("Policy loaded.")

    robot_cfg = SOFollowerRobotConfig(
        port=args.port,
        id="my_follower",
        calibration_dir=args.calib,
        cameras={
            "camera1": OpenCVCropCameraConfig(
                index_or_path=0,
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

    try:
        while True:
            task = choose_prompt()
            if task is None:
                break
            print(f"\nPROMPT: {task}")
            input("Place banana/layout, then press Enter to run ...")
            go_to_reset_pose(robot)
            run_episode(robot, policy, preprocessor, postprocessor, task)
            print("Episode finished.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
