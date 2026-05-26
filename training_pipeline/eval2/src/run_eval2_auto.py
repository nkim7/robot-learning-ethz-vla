#!/usr/bin/env python3
"""
Eval2 runner with automatic reset between episodes.
Usage:
    python run_eval2_auto.py \
        --task "Put the banana into the bowl to the right of the blue bowl." \
        --num-episodes 5
"""
import argparse
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = str(REPO_ROOT / ".venv/bin/python")
LEROBOT_RECORD = str(REPO_ROOT / ".venv/bin/lerobot-record")
RESET_SCRIPT = str(REPO_ROOT / "robot/reset_from_eval1_pose.py")
POSE_FILE = str(REPO_ROOT / "robot/config/poses/eval1_follower_initial_pose.yaml")
CALIBRATION_DIR = str(REPO_ROOT / "robot/calibration/so101_group47/follower")
CAMERAS = (
    '{"camera1":{"type":"opencv_crop","index_or_path":0,'
    '"source_width":640,"source_height":480,'
    '"roi_x":215,"roi_y":0,"roi_size":425,'
    '"width":256,"height":256,"fps":15,"warmup_s":1}}'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Put the banana into the bowl to the right of the blue bowl.")
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--episode-time-s", type=float, default=20.0)
    p.add_argument("--policy", default=str(REPO_ROOT / "models/eval2"))
    p.add_argument("--repo-id", default="local/eval2_auto_run")
    p.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", ""))
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def reset_pose(python: str, reset_script: str, pose_file: str):
    print("\n" + "="*50 + "\n  AUTO RESET → moving to start pose...\n" + "="*50)
    subprocess.run([python, reset_script, "--pose_file", pose_file, "--duration", "3.0"], check=True)
    time.sleep(1.0)


def run_episode(ep_idx: int, args, total: int, port: str):
    print(f"\n{'='*50}\n  EPISODE {ep_idx + 1} / {total}\n  Task: {args.task}\n{'='*50}\n")
    if ep_idx == 0:
        cache_path = Path.home() / ".cache/huggingface/lerobot" / args.repo_id
        if cache_path.exists():
            shutil.rmtree(cache_path)

    cmd = [
        LEROBOT_RECORD,
        "--robot.type=so101_follower",
        f"--robot.port={port}",
        "--robot.id=my_follower",
        f"--robot.calibration_dir={CALIBRATION_DIR}",
        f"--robot.cameras={CAMERAS}",
        f"--policy.path={args.policy}",
        '--dataset.rename_map={"observation.images.camera1":"observation.images.front"}',
        f"--dataset.repo_id={args.repo_id}",
        "--dataset.push_to_hub=false",
        "--dataset.fps=15",
        "--dataset.num_episodes=1",
        f"--dataset.episode_time_s={args.episode_time_s}",
        "--dataset.reset_time_s=0",
        f"--dataset.single_task={args.task}",
        f"--display_data={'false' if args.no_display else 'true'}",
    ]
    if ep_idx > 0:
        cmd.append("--resume")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Episode {ep_idx + 1} failed (returncode={result.returncode})")
        sys.exit(result.returncode)


def main():
    args = parse_args()
    port = args.port
    if not port:
        print("ERROR: FOLLOWER_PORT not set. Pass --port or export FOLLOWER_PORT.")
        sys.exit(1)

    print(f"\nEval2 Auto-Reset Runner\n  Policy: {args.policy}\n  Task: {args.task}\n  Episodes: {args.num_episodes}\n")
    reset_pose(PYTHON, RESET_SCRIPT, POSE_FILE)

    for ep in range(args.num_episodes):
        run_episode(ep, args, args.num_episodes, port)
        if ep < args.num_episodes - 1:
            reset_pose(PYTHON, RESET_SCRIPT, POSE_FILE)

    print(f"\n{'='*50}\n  All {args.num_episodes} episodes done!\n{'='*50}\n")


if __name__ == "__main__":
    main()
