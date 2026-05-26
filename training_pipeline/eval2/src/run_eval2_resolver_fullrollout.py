#!/usr/bin/env python3
"""
Delayed resolver-assisted Eval 2 backup runner.

This first runs a generic grasp/lift behavior so the wrist camera can see the
bowls. It then resolves the original TA prompt on visible frames and continues
with a simple color prompt:
    "Put the banana in the red/green/blue bowl."
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.device_utils import get_safe_torch_device

from run_eval2_resolver_twophase import (
    EMPTY_CAMS,
    FPS,
    INITIAL_POSE,
    LIFT_DIST_THRESH,
    LIFT_POSE,
    MOTOR_KEYS,
    POSE_WEIGHTS,
    action_to_dict,
    load_resolver,
    reset_to_initial_pose,
    resolve_color,
)


MAX_STEPS = FPS * 20
MAX_LIFT_STEPS = FPS * 10
MAX_PLACE_STEPS = FPS * 10


BOOTSTRAP_TASK = "Put the banana in the red bowl."


def build_obs_frame(obs: dict) -> dict[str, np.ndarray]:
    frame = {
        "observation.state": np.array([obs[k] for k in MOTOR_KEYS], dtype=np.float32),
        "observation.images.camera1": obs["camera1"],
    }
    for key, shape in EMPTY_CAMS.items():
        frame[key] = np.zeros(shape, dtype=np.uint8)
    return frame


def pose_dist(obs: dict) -> float:
    state = np.array([obs[k] for k in MOTOR_KEYS], dtype=np.float32)
    return float(np.sum(POSE_WEIGHTS * np.abs(state - LIFT_POSE)))


def resolve_visible_frame_with_vote(
    robot,
    resolver_model,
    resolver_processor,
    prompt: str,
    device: torch.device,
    votes_needed: int = 2,
    max_rounds: int = 3,
) -> str:
    votes: list[str] = []
    raws: list[str] = []

    for round_idx in range(max_rounds):
        votes.clear()
        raws.clear()
        print(f"  Resolver vote round {round_idx + 1}/{max_rounds} on visible frames ...")
        for _ in range(3):
            obs = robot.get_observation()
            image = Image.fromarray(obs["camera1"])
            color, raw = resolve_color(resolver_model, resolver_processor, image, prompt, device)
            raws.append(raw)
            if color is not None:
                votes.append(color)
            time.sleep(0.15)

        counts = Counter(votes)
        print(f"  Resolver votes: {dict(counts)}")
        if counts:
            color, count = counts.most_common(1)[0]
            if count >= votes_needed:
                return color

        print("  No resolver majority yet. Raw outputs:")
        for raw in raws:
            print(f"    {raw}")
        if round_idx < max_rounds - 1:
            print("  Waiting for a clearer visible frame before retrying ...")
            time.sleep(0.4)

    raise RuntimeError("Resolver failed to reach a majority on visible frames; refusing to execute an arbitrary target.")


def run_trial(robot, policy, preprocessor, postprocessor, resolver_model, resolver_processor, prompt, device, trial_num):
    print(f"\n{'=' * 54}")
    print(f"Trial {trial_num}")
    print(f"Original prompt: {prompt}")
    print(f"{'=' * 54}")

    reset_to_initial_pose(robot)
    input("Place banana and bowls, then press ENTER to start delayed resolver rollout...")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    use_amp = device.type == "cuda"

    print(f"Bootstrap task: {BOOTSTRAP_TASK}")
    phase2_task: str | None = None

    for step in range(MAX_STEPS):
        t0 = time.perf_counter()
        obs = robot.get_observation()

        if phase2_task is None:
            dist = pose_dist(obs)
            if dist < LIFT_DIST_THRESH or step >= MAX_LIFT_STEPS:
                print(f"\n  Lift/bowl-visible gate reached at step {step} (pose_dist={dist:.1f}).")
                target_color = resolve_visible_frame_with_vote(
                    robot,
                    resolver_model,
                    resolver_processor,
                    prompt,
                    device,
                )
                phase2_task = f"Put the banana in the {target_color} bowl."
                print(f"  Resolved task: {phase2_task}\n")
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

        task = phase2_task or BOOTSTRAP_TASK
        action_tensor = predict_action(
            observation=build_obs_frame(obs),
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
        )
        robot.send_action(action_to_dict(action_tensor))
        if phase2_task is not None and step >= MAX_LIFT_STEPS + MAX_PLACE_STEPS:
            break
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / FPS - elapsed))

    reset_to_initial_pose(robot)
    print(f"Trial {trial_num} done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default="chengming")
    parser.add_argument("--calibration_dir", default="calibration/so101_group47/follower")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--policy_path", default="models/eval2_2phase_resolvedcolor_016000/checkpoints/016000")
    parser.add_argument("--resolver_path", default="models/eval2_resolver_v2_best/best")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompts", nargs="+", default=["Put the banana into the 2nd bowl from the left."])
    args = parser.parse_args()

    if args.port is None:
        ports = glob.glob("/dev/tty.usbmodem*")
        if not ports:
            print("ERROR: no /dev/tty.usbmodem* device found. Plug in the arm.")
            sys.exit(1)
        args.port = ports[0]
        print(f"Auto-detected port: {args.port}")

    if args.device:
        device = get_safe_torch_device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading action policy ...")
    policy_path = str(args.policy_path)
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print("Loading resolver ...")
    resolver_model, resolver_processor = load_resolver(args.resolver_path, device)

    cfg = SO101FollowerConfig(
        port=args.port,
        use_degrees=True,
        disable_torque_on_disconnect=False,
        cameras={
            "camera1": OpenCVCameraConfig(
                index_or_path=args.camera_index,
                width=640,
                height=480,
                fps=FPS,
            )
        },
    )
    cfg.id = args.id
    cfg.calibration_dir = Path(args.calibration_dir).resolve()

    robot = SOFollower(cfg)
    robot.connect()
    try:
        for i, prompt in enumerate(args.prompts, start=1):
            run_trial(robot, policy, preprocessor, postprocessor, resolver_model, resolver_processor, prompt, device, i)
            if i < len(args.prompts):
                input("Press ENTER when ready for next trial...")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
