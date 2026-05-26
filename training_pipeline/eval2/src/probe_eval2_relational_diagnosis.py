#!/usr/bin/env python3
"""Diagnose Eval2 relational failures: VLM/slot head vs action routing.

For each fixed image/state frame, this probes:
  1. Expected target slot from known bowl layout.
  2. Auxiliary slot-head prediction for paired relative prompts.
  3. Whether the action chunk is closest to the expected ordinal motor anchor.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval2_layout_utils import SLOTS, layout_id, normalize_layout, target_slot
from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


COLORS = ("red", "green", "blue")
SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}
ID_TO_SLOT = {i: slot for slot, i in SLOT_TO_ID.items()}
CHUNK = 50
DEFAULT_CKPT_BASE = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints"
DATASET_REPO = "robot-learning-group47/eval2_compositional_augmented"
DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented"

COMBOS = [
    ("bgr", "blue", 0),
    ("bgr", "green", 6),
    ("bgr", "red", 12),
    ("brg", "blue", 18),
    ("brg", "green", 24),
    ("brg", "red", 30),
    ("gbr", "blue", 36),
    ("gbr", "green", 42),
    ("gbr", "red", 48),
    ("grb", "blue", 54),
    ("grb", "green", 60),
    ("grb", "red", 66),
    ("rbg", "blue", 72),
    ("rbg", "green", 78),
    ("rbg", "red", 84),
    ("rgb", "blue", 90),
    ("rgb", "green", 96),
    ("rgb", "red", 102),
]


@dataclass(frozen=True)
class FrameInfo:
    global_index: int
    episode_index: int
    orig_episode_index: int
    layout: list[str]
    layout_id: str
    demo_target_color: str
    demo_target_slot: str


@dataclass(frozen=True)
class PromptCase:
    key: str
    prompt: str
    family: str
    expected_slot: str
    expected_color: str


def augmented_episode_info(episode_index: int) -> tuple[list[str], str, str, str, int]:
    cursor = 0
    for layout_code, color, orig_start in COMBOS:
        layout = normalize_layout(layout_code)
        prompts_per_demo = 8 if color == layout[1] else 6
        next_cursor = cursor + 6 * prompts_per_demo
        if cursor <= episode_index < next_cursor:
            orig_episode = orig_start + ((episode_index - cursor) // prompts_per_demo)
            return layout, layout_id(layout), color, target_slot(layout, color), orig_episode
        cursor = next_cursor
    raise ValueError(f"Episode {episode_index} outside augmented mapping")


def pick_frames(ds: LeRobotDataset) -> list[FrameInfo]:
    desired_layouts = {"rgb", "gbr", "brg"}
    found: dict[str, FrameInfo] = {}
    ep_values = list(ds.hf_dataset["episode_index"])
    for global_idx, ep_value in enumerate(ep_values):
        ep = int(ep_value)
        layout, lid, color, slot, orig = augmented_episode_info(ep)
        code = "".join(c[0] for c in layout)
        if code in desired_layouts and code not in found:
            found[code] = FrameInfo(global_idx, ep, orig, layout, lid, color, slot)
        if len(found) == len(desired_layouts):
            break
    return [found[k] for k in sorted(found)]


def make_cases(layout: list[str]) -> list[PromptCase]:
    cases = []
    for idx, ord_text in enumerate(("1st", "2nd", "3rd")):
        cases.append(PromptCase(
            f"ord_{SLOTS[idx]}",
            f"Put the banana into the {ord_text} bowl from the left from the robot perspective.",
            "ordinal_anchor",
            SLOTS[idx],
            layout[idx],
        ))

    for ref_idx, ref_color in enumerate(layout):
        if ref_idx > 0:
            cases.append(PromptCase(
                f"left_of_{ref_color}",
                f"Put the banana into the bowl on the left of the {ref_color} bowl from the robot perspective.",
                "relative",
                SLOTS[ref_idx - 1],
                layout[ref_idx - 1],
            ))
        if ref_idx < 2:
            cases.append(PromptCase(
                f"right_of_{ref_color}",
                f"Put the banana into the bowl on the right of the {ref_color} bowl from the robot perspective.",
                "relative",
                SLOTS[ref_idx + 1],
                layout[ref_idx + 1],
            ))
    return cases


def frame_to_obs(ds: LeRobotDataset, global_idx: int) -> dict:
    sample = ds[global_idx]
    image = sample["observation.images.front"]
    state = sample["observation.state"]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    obs = {
        "observation.images.front": image,
        "observation.state": state.astype(np.float32),
        "observation.images.empty_camera_0": np.zeros((480, 640, 3), np.uint8),
        "observation.images.empty_camera_1": np.zeros((480, 640, 3), np.uint8),
    }
    return obs


def add_slot_hook(policy, ckpt_dir: Path, storage: dict):
    slot_head = nn.Linear(policy.model.vlm_with_expert.expert_hidden_size, 3).to(policy.config.device)
    slot_head.load_state_dict(torch.load(ckpt_dir / "target_slot_head.pt", map_location=policy.config.device))
    slot_head.eval()
    action_out = policy.model.action_out_proj

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        with torch.no_grad():
            storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

    return action_out.register_forward_hook(hook), slot_head


def run_prompt(policy, preprocessor, postprocessor, obs, prompt: str, storage: dict, device) -> dict:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    storage.clear()
    actions = []
    slot_pred = None
    logits_np = None
    for i in range(CHUNK):
        action = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=prompt,
        )
        arr = action.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.squeeze(0)
        actions.append(arr)
        if i == 0 and "slot_logits" in storage:
            logits_np = storage["slot_logits"].detach().cpu().numpy()
            if logits_np.ndim > 1:
                logits_np = logits_np[0]
            slot_pred = int(np.argmax(logits_np))
    return {
        "actions": np.stack(actions),
        "slot_pred": slot_pred,
        "slot_pred_str": ID_TO_SLOT.get(slot_pred, "?"),
        "slot_logits": logits_np,
    }


def nearest_anchor_slot(action_pan: float, anchor_pans: dict[str, float]) -> str:
    return min(anchor_pans, key=lambda slot: abs(anchor_pans[slot] - action_pan))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["010000", "015000", "020000", "030000"])
    ap.add_argument("--ckpt-base", default=DEFAULT_CKPT_BASE)
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--n-frames", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", "/home/shadeform/lerobot_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading dataset")
    ds = LeRobotDataset(DATASET_REPO, root=args.dataset_root, video_backend="pyav")
    frames = pick_frames(ds)[: args.n_frames]
    for fi in frames:
        print(
            f"Frame global={fi.global_index} ep={fi.episode_index} orig={fi.orig_episode_index} "
            f"layout={fi.layout_id} demo={fi.demo_target_slot}/{fi.demo_target_color}"
        )

    for step in args.ckpts:
        ckpt_dir = Path(args.ckpt_base) / step / "pretrained_model"
        if not (ckpt_dir / "model.safetensors").exists():
            print(f"\n=== {step}: missing {ckpt_dir}, skipping ===")
            continue

        print("\n" + "=" * 100)
        print(f"CHECKPOINT {step}: {ckpt_dir}")
        print("=" * 100)
        policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir))
        policy.config.device = str(device)
        policy = policy.to(device).eval()
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        storage: dict = {}
        hook, _slot_head = add_slot_hook(policy, ckpt_dir, storage)

        slot_hits = []
        action_hits = []
        for fi in frames:
            print(f"\nLayout {fi.layout_id}: {fi.layout}")
            obs = frame_to_obs(ds, fi.global_index)
            cases = make_cases(fi.layout)
            results = {case.key: run_prompt(policy, pre, post, obs, case.prompt, storage, device) for case in cases}

            anchor_pans = {}
            for slot in SLOTS:
                r = results[f"ord_{slot}"]
                anchor_pans[slot] = float(r["actions"][:10, 0].mean())
            print(f"  Ordinal action pan anchors mean[0:10]: {anchor_pans}")

            print(
                f"  {'prompt':34s} {'expected':15s} {'slot_pred':10s} "
                f"{'slot_ok':7s} {'action_near':11s} {'action_ok':9s} {'pan_m10':>9s}"
            )
            print("  " + "-" * 110)
            for case in cases:
                if case.family != "relative":
                    continue
                r = results[case.key]
                pan_m10 = float(r["actions"][:10, 0].mean())
                action_near = nearest_anchor_slot(pan_m10, anchor_pans)
                slot_ok = r["slot_pred_str"] == case.expected_slot
                action_ok = action_near == case.expected_slot
                slot_hits.append(slot_ok)
                action_hits.append(action_ok)
                print(
                    f"  {case.key:34s} {case.expected_slot+'/'+case.expected_color:15s} "
                    f"{r['slot_pred_str']:10s} {str(slot_ok):7s} {action_near:11s} "
                    f"{str(action_ok):9s} {pan_m10:+9.3f}"
                )

        slot_acc = sum(slot_hits) / max(1, len(slot_hits))
        action_acc = sum(action_hits) / max(1, len(action_hits))
        print("\nSUMMARY")
        print(f"  relative slot-head accuracy: {slot_acc*100:.1f}% ({sum(slot_hits)}/{len(slot_hits)})")
        print(f"  relative action-nearest accuracy: {action_acc*100:.1f}% ({sum(action_hits)}/{len(action_hits)})")
        if slot_acc < 0.8:
            print("  diagnosis: VLM/slot-head relational reasoning failure")
        elif action_acc < 0.8:
            print("  diagnosis: action-side routing failure")
        else:
            print("  diagnosis: offline probe looks good; investigate robot/camera/reset mismatch")

        hook.remove()
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
