#!/usr/bin/env python3
"""
Run a specific eval1 checkpoint on the SO-101 robot.

Usage:
    python run_eval1_checkpoint.py --step 10000
    python run_eval1_checkpoint.py --step 8000
    python run_eval1_checkpoint.py --repo robot-learning-group47/eval1-fixed-bowl-step006000

Available checkpoints:
    robot-learning-group47/eval1-fixed-bowl-step002000
    robot-learning-group47/eval1-fixed-bowl-step004000
    robot-learning-group47/eval1-fixed-bowl-step006000
    robot-learning-group47/eval1-fixed-bowl-step008000
    robot-learning-group47/eval1-fixed-bowl-step010000  (final)
"""
import argparse, glob, os, sys, time
from pathlib import Path

sys.path.insert(0, "lerobot/src")

_REPO_ROOT = Path(__file__).resolve().parents[3]

STEPS = [2000, 4000, 6000, 8000, 10000]


def _detect_follower_port() -> str:
    port = os.environ.get("FOLLOWER_PORT")
    if port:
        return port
    ports = glob.glob("/dev/tty.usbmodem*")
    if not ports:
        raise RuntimeError("No /dev/tty.usbmodem* found. Set FOLLOWER_PORT env var.")
    print(f"Auto-detected port: {ports[0]}")
    return ports[0]

def get_repo(step):
    s = f"{step:06d}"
    return f"robot-learning-group47/eval1-fixed-bowl-step{s}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=STEPS, default=10000)
    parser.add_argument("--repo", type=str, default=None, help="Override repo directly")
    parser.add_argument("--task", type=str, default="Pick up the banana and place it in the red bowl",
                        help="Task instruction")
    parser.add_argument("--n_episodes", type=int, default=5)
    args = parser.parse_args()

    repo = args.repo or get_repo(args.step)
    print(f"Loading checkpoint: {repo}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.n_episodes}")
    print()

    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.processor import make_default_robot_action_processor

    from huggingface_hub import snapshot_download
    print(f"Downloading {repo}...")
    ckpt_path = snapshot_download(repo)
    print(f"Cached at: {ckpt_path}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    policy.eval()
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy = policy.to(device)
    print(f"Policy loaded on {device}")

    robot_cfg = SOFollowerRobotConfig(
        port=_detect_follower_port(),
        id="my_follower",
        calibration_dir=_REPO_ROOT / "robot/calibration/so101_group47/follower",
        use_degrees=True,
    )
    robot = SOFollower(robot_cfg)
    processor = make_default_robot_action_processor()
    robot.connect()
    print("Robot connected.")

    try:
        for ep in range(args.n_episodes):
            input(f"\nEpisode {ep+1}/{args.n_episodes} — press Enter to start...")
            policy.reset()

            for step in range(300):  # max 20s at 15Hz
                t0 = time.perf_counter()
                obs = robot.get_observation()
                action, _ = policy.select_action({"observation": obs, "task": args.task})
                processed = processor((action, obs))
                robot.send_action(processed)
                precise_sleep(max(1/15 - (time.perf_counter() - t0), 0))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")

if __name__ == "__main__":
    main()
