#!/usr/bin/env python3
"""
Run Eval 2 with separate pickup/lift and placement policies.

Phase 1: pickup policy runs for --lift_seconds, then arm servos to LIFT_POSE.
Phase 2: place policy runs for --place_seconds.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower


MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

INITIAL_POSE = {
    "shoulder_pan.pos": -2.24,
    "shoulder_lift.pos": -0.48,
    "elbow_flex.pos": -2.86,
    "wrist_flex.pos": 75.34,
    "wrist_roll.pos": -122.59,
    "gripper.pos": 0.38,
}

LIFT_POSE = np.array([7.43, 56.57, -82.95, 64.18, -121.27, 0.0], dtype=np.float32)
LIFT_POSE_ACTION = {key: float(value) for key, value in zip(MOTOR_KEYS, LIFT_POSE)}

FPS = 15
EMPTY_CAMS = {
    "observation.images.empty_camera_0": (480, 640, 3),
    "observation.images.empty_camera_1": (480, 640, 3),
}

PICKUP_PROMPT = "Pick up the banana and lift it until all three bowls are visible."


def reset_to_initial_pose(robot, duration: float = 2.5, hold: float = 0.5, hz: float = 30.0) -> None:
    print("Resetting to initial pose ...")
    try:
        obs = robot.get_observation()
        start = {k: float(obs[k]) if k in obs else v for k, v in INITIAL_POSE.items()}
    except Exception:
        start = INITIAL_POSE.copy()
    steps = max(1, int(duration * hz))
    for i in range(1, steps + 1):
        alpha = i / steps
        action = {k: start[k] * (1.0 - alpha) + INITIAL_POSE[k] * alpha for k in INITIAL_POSE}
        robot.send_action(action)
        time.sleep(1.0 / hz)
    for _ in range(max(1, int(hold * hz))):
        robot.send_action(INITIAL_POSE)
        time.sleep(1.0 / hz)


def servo_to_lift_pose(robot, duration: float = 1.5, hold: float = 0.5, hz: float = 30.0) -> None:
    print("Servoing to lift pose ...")
    try:
        obs = robot.get_observation()
        start = {k: float(obs[k]) if k in obs else v for k, v in LIFT_POSE_ACTION.items()}
    except Exception:
        start = LIFT_POSE_ACTION.copy()
    steps = max(1, int(duration * hz))
    for i in range(1, steps + 1):
        alpha = i / steps
        action = {k: start[k] * (1.0 - alpha) + LIFT_POSE_ACTION[k] * alpha for k in LIFT_POSE_ACTION}
        robot.send_action(action)
        time.sleep(1.0 / hz)
    for _ in range(max(1, int(hold * hz))):
        robot.send_action(LIFT_POSE_ACTION)
        time.sleep(1.0 / hz)


def action_to_dict(action_tensor: torch.Tensor) -> dict[str, float]:
    action = action_tensor.reshape(-1).cpu()
    return {k: float(action[i]) for i, k in enumerate(MOTOR_KEYS)}


def build_obs(obs: dict, camera_obs_key: str, policy_image_key: str) -> dict[str, np.ndarray]:
    frame = {
        "observation.state": np.array([obs[k] for k in MOTOR_KEYS], dtype=np.float32),
        policy_image_key: obs[camera_obs_key],
    }
    for key, shape in EMPTY_CAMS.items():
        frame[key] = np.zeros(shape, dtype=np.uint8)
    return frame


def load_policy(path: str, device: torch.device, n_action_steps: int | None = None):
    policy = SmolVLAPolicy.from_pretrained(path)
    if n_action_steps is not None:
        if n_action_steps < 1:
            raise ValueError("--*_n_action_steps must be >= 1")
        if n_action_steps > policy.config.chunk_size:
            raise ValueError(
                f"n_action_steps={n_action_steps} exceeds chunk_size={policy.config.chunk_size} for {path}"
            )
        policy.config.n_action_steps = n_action_steps
        policy.reset()
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def predict_policy_action(
    *,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    obs: dict,
    camera_obs_key: str,
    policy_image_key: str,
    task: str,
) -> torch.Tensor:
    return predict_action(
        observation=build_obs(obs, camera_obs_key, policy_image_key),
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=(device.type == "cuda"),
        task=task,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default="chengming")
    parser.add_argument("--calibration_dir", default="calibration/so101_group47/follower")
    parser.add_argument("--pickup_policy", default="./models/eval2_pickuplift_020000")
    parser.add_argument("--place_policy", default="./models/eval2_place_020000")
    parser.add_argument("--pickup_n_action_steps", type=int, default=50)
    parser.add_argument("--place_n_action_steps", type=int, default=40)
    parser.add_argument("--lift_seconds", type=float, default=8.0)  # phase 1 duration
    parser.add_argument("--place_seconds", type=float, default=10.0)
    parser.add_argument("--place_prompt", required=True)
    args = parser.parse_args()

    if args.port is None:
        ports = glob.glob("/dev/tty.usbmodem*")
        if not ports:
            print("ERROR: no /dev/tty.usbmodem* device found. Plug in the arm.")
            sys.exit(1)
        args.port = ports[0]
        print(f"Auto-detected port: {args.port}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading pickup policy: {args.pickup_policy}")
    pickup_policy, pickup_pre, pickup_post = load_policy(
        args.pickup_policy, device, n_action_steps=args.pickup_n_action_steps
    )
    print(f"Pickup n_action_steps: {pickup_policy.config.n_action_steps}")

    print(f"Loading place policy: {args.place_policy}")
    place_policy, place_pre, place_post = load_policy(
        args.place_policy, device, n_action_steps=args.place_n_action_steps
    )
    print(f"Place n_action_steps: {place_policy.config.n_action_steps}")

    cfg = SO101FollowerConfig(
        port=args.port,
        use_degrees=True,
        disable_torque_on_disconnect=False,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=0,
                width=640,
                height=480,
                fps=FPS,
                warmup_s=5.0,
            )
        },
    )
    cfg.id = args.id
    cfg.calibration_dir = Path(args.calibration_dir).resolve()

    robot = SOFollower(cfg)
    robot.connect()
    try:
        reset_to_initial_pose(robot)
        input("Place banana and bowls, then press ENTER to start rollout...")

        pickup_policy.reset()
        pickup_pre.reset()
        pickup_post.reset()

        # --- Phase 1: pickup for lift_seconds ---
        print(f"Phase 1: pickup for {args.lift_seconds:.1f}s  |  {PICKUP_PROMPT}")
        lift_steps = int(args.lift_seconds * FPS)
        for step in range(lift_steps):
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            action = predict_policy_action(
                policy=pickup_policy, preprocessor=pickup_pre, postprocessor=pickup_post,
                device=device, obs=obs,
                camera_obs_key="front", policy_image_key="observation.images.camera1",
                task=PICKUP_PROMPT,
            )
            robot.send_action(action_to_dict(action))
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - loop_start)))

        place_policy.reset()
        place_pre.reset()
        place_post.reset()

        # --- Phase 2: placement for place_seconds ---
        print(f"Phase 2: placement for {args.place_seconds:.1f}s  |  {args.place_prompt}")
        place_steps = int(args.place_seconds * FPS)
        for step in range(place_steps):
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            action = predict_policy_action(
                policy=place_policy, preprocessor=place_pre, postprocessor=place_post,
                device=device, obs=obs,
                camera_obs_key="front", policy_image_key="observation.images.front",
                task=args.place_prompt,
            )
            robot.send_action(action_to_dict(action))
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - loop_start)))

        reset_to_initial_pose(robot)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
