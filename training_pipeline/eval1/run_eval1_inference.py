#!/usr/bin/env python3
"""
Run eval1 fixed-bowl checkpoint on the SO-101 robot.

Usage:
    python run_eval1_inference.py --step 10000 --task "Put the banana in the red colored bowl."
    python run_eval1_inference.py --step 8000  --task "Put the banana in the blue colored bowl."
    python run_eval1_inference.py --step 6000  --task "Put the banana in the green colored bowl."
"""
import argparse, sys, time, torch
from pathlib import Path

sys.path.insert(0, "lerobot/src")

PORT  = "/dev/tty.usbmodem5AE60831701"
CALIB = "calibration/so101_group47/follower"
STEPS = {2000, 4000, 6000, 8000, 10000}

TASKS = [
    "Put the banana in the red colored bowl.",
    "Put the banana in the blue colored bowl.",
    "Put the banana in the green colored bowl.",
]

def repo_for_step(step):
    return f"robot-learning-group47/eval1-fixed-bowl-step{step:06d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=10000, choices=sorted(STEPS))
    parser.add_argument("--task", type=str, default=TASKS[0])
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode (300 = 20s at 15Hz)")
    args = parser.parse_args()

    print(f"Checkpoint : {repo_for_step(args.step)}")
    print(f"Task       : {args.task}")
    print(f"Episodes   : {args.n_episodes}")
    print()

    # ── Load policy ──────────────────────────────────────────────────────────
    from huggingface_hub import snapshot_download
    ckpt_path = snapshot_download(repo_for_step(args.step))

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    policy.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy = policy.to(device)
    print(f"Policy loaded on {device}")

    # Build zero tensors for the extra cameras the model expects but robot doesn't have
    # Model expects: camera1 (real), camera2, camera3, empty_camera_0, empty_camera_1 (all zeros)
    H, W = 256, 256
    dummy_img = torch.zeros(1, 3, H, W, device=device)

    # ── Load robot ───────────────────────────────────────────────────────────
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.utils.robot_utils import precise_sleep

    robot_cfg = SOFollowerRobotConfig(
        port=PORT,
        id="my_follower",
        calibration_dir=Path(CALIB),
        use_degrees=True,
    )
    robot = SOFollower(robot_cfg)
    robot.connect()
    print("Robot connected.\n")

    try:
        for ep in range(args.n_episodes):
            input(f"Episode {ep+1}/{args.n_episodes} — press Enter to start (Ctrl+C to quit)...")
            policy.reset()

            for _ in range(args.max_steps):
                t0 = time.perf_counter()
                raw_obs = robot.get_observation()

                # Build observation dict matching the model's expected keys
                obs = {}
                for k, v in raw_obs.items():
                    if k == "observation.images.front":
                        # rename front -> camera1, fill rest with zeros
                        img = torch.as_tensor(v, dtype=torch.float32, device=device)
                        if img.ndim == 3:  # H,W,C -> 1,C,H,W
                            img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
                        obs["observation.images.camera1"]     = img
                        obs["observation.images.camera2"]     = dummy_img
                        obs["observation.images.camera3"]     = dummy_img
                        obs["observation.images.empty_camera_0"] = dummy_img
                        obs["observation.images.empty_camera_1"] = dummy_img
                    else:
                        v_t = torch.as_tensor(v, dtype=torch.float32, device=device)
                        obs[k] = v_t.unsqueeze(0) if v_t.ndim == 1 else v_t

                obs["task"] = [args.task]

                with torch.no_grad():
                    action = policy.select_action(obs)

                # Convert action tensor back to robot dict
                action_np = action.squeeze(0).cpu().numpy()
                names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
                action_dict = {n: float(action_np[i]) for i, n in enumerate(names)}

                robot.send_action(action_dict)
                precise_sleep(max(1/15 - (time.perf_counter() - t0), 0))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
