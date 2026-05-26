#!/usr/bin/env python3
"""
Action-conditioning probe.

Loads a SmolVLA checkpoint, feeds the SAME synthetic scene image through the policy
with DIFFERENT prompts, and measures whether the predicted action changes.

If the action does not change across prompts → model is ignoring language (positional fallback).
If the action changes meaningfully → model is prompt-conditioned.

Usage:
    python probe_action_conditioning.py --model models/frozen-vqa-expert-014000/pretrained_model
    python probe_action_conditioning.py --model models/lora-vqa-expert-014000/pretrained_model
    python probe_action_conditioning.py --model models/frozen-vqa-expert-014000/pretrained_model \
                                        --model models/lora-vqa-expert-014000/pretrained_model

Reports:
    - Per-prompt predicted gripper target (shoulder_pan, most discriminative for left/centre/right)
    - L2 distance between action pairs
    - Prompt-conditioned variance vs single-prompt variance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "lerobot/src")

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.control_utils import predict_action

MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

PROMPTS = {
    # COLOR
    "red":    "Put the banana in the red colored bowl.",
    "green":  "Put the banana in the green colored bowl.",
    "blue":   "Put the banana in the blue colored bowl.",
    # ORDINAL
    "1st_L":  "Put the banana into the 1st bowl from the left from the robot perspective.",
    "2nd_L":  "Put the banana into the 2nd bowl from the left from the robot perspective.",
    "3rd_L":  "Put the banana into the 3rd bowl from the left from the robot perspective.",
    # RELATIVE
    "2nd_R":  "Put the banana into the 2nd bowl from the right from the robot perspective.",
    # NEGATION
    "not_rg": "Put the banana into the bowl that is not red and not green.",
    "not_rb": "Put the banana into the bowl that is not red and not blue.",
}


def make_dummy_obs(device):
    """Synthetic observation: white image, zero state, zero empty cameras."""
    obs = {
        "observation.images.camera1": np.ones((256, 256, 3), dtype=np.uint8) * 128,
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.empty_camera_0": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.empty_camera_1": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    return obs


def load_policy(path: str, device):
    print(f"\nLoading {path} ...")
    adapter_config_path = Path(path) / "adapter_config.json"

    if adapter_config_path.exists():
        # PEFT model: load base then apply adapter
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(path)
        base_path = peft_config.base_model_name_or_path
        print(f"  PEFT model detected. Base: {base_path}")
        policy = SmolVLAPolicy.from_pretrained(base_path)
        policy = PeftModel.from_pretrained(policy, path, config=peft_config)
    else:
        policy = SmolVLAPolicy.from_pretrained(path)

    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"  Loaded. Config: chunk_size={policy.config.chunk_size}")
    return policy, preprocessor, postprocessor


def probe_model(name: str, policy, preprocessor, postprocessor, device):
    print(f"\n{'═'*70}")
    print(f"  MODEL: {name}")
    print(f"{'═'*70}")

    obs = make_dummy_obs(device)
    actions = {}

    for prompt_key, prompt_text in PROMPTS.items():
        policy.reset()
        with torch.no_grad():
            action = predict_action(
                observation=obs,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=prompt_text,
            )
        # action shape: (chunk_size, 6) — take first step
        a = action[0].cpu().numpy()
        actions[prompt_key] = a
        pan = a[0]  # shoulder_pan — most discriminative for left/centre/right
        print(f"  {prompt_key:8s}  pan={pan:+7.2f}  lift={a[1]:+7.2f}  elbow={a[2]:+7.2f}  gripper={a[5]:+6.2f}")

    # ── Divergence analysis ───────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Pairwise L2 distances (first action step, all 6 joints)")
    keys = list(actions.keys())
    dists = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            d = np.linalg.norm(actions[k1] - actions[k2])
            dists[(k1, k2)] = d

    # Sort by distance
    for (k1, k2), d in sorted(dists.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k1:8s} ↔ {k2:8s}  L2={d:.4f}")

    # ── Variance summary ──────────────────────────────────────────────────────
    all_actions = np.stack(list(actions.values()))  # (N, 6)
    per_joint_std = all_actions.std(axis=0)
    total_var = all_actions.var(axis=0).sum()
    print(f"\n  Per-joint std across prompts:")
    for i, (k, s) in enumerate(zip(MOTOR_KEYS, per_joint_std)):
        bar = "█" * int(s * 20)
        print(f"    {k:25s}  std={s:.4f}  {bar}")
    print(f"\n  Total action variance across prompts: {total_var:.6f}")
    print(f"  Mean max pairwise L2: {max(dists.values()):.4f}")

    # Color-specific check
    color_actions = np.stack([actions["red"], actions["green"], actions["blue"]])
    color_var = color_actions.var(axis=0).sum()
    color_pan_range = color_actions[:, 0].max() - color_actions[:, 0].min()
    print(f"\n  Color-only (red/green/blue):")
    print(f"    pan range:   {color_pan_range:.4f}  {'← GROUNDING' if color_pan_range > 5.0 else '← COLLAPSED (< 5°)'}")
    print(f"    total var:   {color_var:.6f}")

    # Ordinal check
    ord_actions = np.stack([actions["1st_L"], actions["2nd_L"], actions["3rd_L"]])
    ord_pan_range = ord_actions[:, 0].max() - ord_actions[:, 0].min()
    print(f"\n  Ordinal-only (1st/2nd/3rd from left):")
    print(f"    pan range:   {ord_pan_range:.4f}  {'← GROUNDING' if ord_pan_range > 5.0 else '← COLLAPSED (< 5°)'}")

    return actions, total_var, color_pan_range, ord_pan_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", required=True,
                        help="One or more checkpoint paths to probe")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    results = {}
    for model_path in args.model:
        name = Path(model_path).parent.name  # e.g. frozen-vqa-expert-014000
        policy, pre, post = load_policy(model_path, device)
        actions, total_var, color_pan_range, ord_pan_range = probe_model(
            name, policy, pre, post, device
        )
        results[name] = {
            "total_var": total_var,
            "color_pan_range": color_pan_range,
            "ord_pan_range": ord_pan_range,
        }
        del policy  # free memory

    if len(results) > 1:
        print(f"\n{'═'*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'═'*70}")
        print(f"  {'Model':40s}  {'ColorPan':>10}  {'OrdPan':>10}  {'TotalVar':>12}")
        for name, r in results.items():
            print(f"  {name:40s}  {r['color_pan_range']:>10.3f}  {r['ord_pan_range']:>10.3f}  {r['total_var']:>12.6f}")


if __name__ == "__main__":
    main()
