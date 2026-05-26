#!/usr/bin/env python3
"""Counterfactual routing probe.

For a given checkpoint, takes the SAME frame per layout and runs all 4 relative
prompts through the model.  The key success signal is NOT global accuracy — it is
whether the model's predictions DIFFER across the 4 prompts for the same frame.

  Layout shortcut (broken):   bgr frame → [left, left, left, left]
  Prompt routing (working):   bgr frame → [left, right, middle, middle]

Usage:
  python probe_counterfactual_routing.py --ckpt /path/to/checkpoint
  python probe_counterfactual_routing.py --ckpt /path/to/checkpoint --layouts bgr gbr rgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch import nn

sys.path.insert(0, "/home/shadeform")
sys.path.insert(0, "/home/shadeform/lerobot/src")

from probe_eval2_relational_diagnosis import (
    DATASET_REPO, CHUNK, ID_TO_SLOT, SLOTS, frame_to_obs, make_cases,
    augmented_episode_info,
)
from eval2_layout_utils import normalize_layout
from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

BASE_POLICY = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
    "/checkpoints/020000/pretrained_model"
)
DATASET_ROOT = (
    "/home/shadeform/lerobot_cache"
    "/robot-learning-group47/eval2_compositional_augmented"
)
ALL_LAYOUTS = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")


# ── model loading ───────────────────────────────────────────────────────────────
def load_policy(ckpt: Path, device: torch.device) -> SmolVLAPolicy:
    if (ckpt / "adapter_config.json").exists():
        base = SmolVLAPolicy.from_pretrained(BASE_POLICY)
        base.config.device = str(device)
        peft_model = PeftModel.from_pretrained(base, str(ckpt))
        policy = peft_model.merge_and_unload()
        print(f"LoRA adapter merged: {ckpt.name}")
    else:
        policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        print(f"Full checkpoint loaded: {ckpt.name}")
    policy.config.device = str(device)
    return policy.to(device).eval()


def attach_slot_hook(policy: SmolVLAPolicy, slot_head: nn.Module, storage: dict):
    def hook(_module, inputs, _output):
        h = inputs[0].float()
        with torch.no_grad():
            storage["slot_logits"] = slot_head(h.mean(dim=1)).detach().cpu().numpy()
    return policy.model.action_out_proj.register_forward_hook(hook)


def run_prompt(policy, pre, post, obs, prompt, storage, device):
    policy.reset(); pre.reset(); post.reset(); storage.clear()
    for i in range(CHUNK):
        predict_action(
            observation=obs, policy=policy, device=device,
            preprocessor=pre, postprocessor=post, use_amp=False, task=prompt,
        )
        if i == 0 and "slot_logits" in storage:
            logits = storage["slot_logits"]
            if logits.ndim > 1:
                logits = logits[0]
            return int(np.argmax(logits)), logits
    return -1, np.zeros(3)


# ── frame picking ───────────────────────────────────────────────────────────────
def pick_layout_frame(ds: LeRobotDataset, code: str) -> int:
    """Return the global index of the first frame for this layout in ds."""
    for global_idx, ep_val in enumerate(ds.hf_dataset["episode_index"]):
        layout, _lid, _color, _slot, _orig = augmented_episode_info(int(ep_val))
        if "".join(c[0] for c in layout) == code:
            return global_idx
    raise RuntimeError(f"Layout {code} not found in dataset")


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint dir")
    parser.add_argument(
        "--layouts", nargs="+", default=["bgr", "gbr", "rgb"],
        help="Layouts to probe (default: bgr gbr rgb)",
    )
    parser.add_argument("--all-layouts", action="store_true",
                        help="Probe all 6 layouts")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    args = parser.parse_args()

    layouts = ALL_LAYOUTS if args.all_layouts else args.layouts
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt    = Path(args.ckpt).resolve()

    print(f"\nCheckpoint : {ckpt}")
    print(f"Layouts    : {layouts}")
    print(f"Device     : {device}\n")

    ds = LeRobotDataset(DATASET_REPO, root=args.dataset_root, video_backend="pyav")

    policy = load_policy(ckpt, device)

    storage: dict = {}
    slot_head = nn.Linear(policy.model.vlm_with_expert.expert_hidden_size, len(SLOTS)).to(device)
    slot_head.load_state_dict(
        torch.load(ckpt / "target_slot_head.pt", map_location=device)
    )
    slot_head.eval()
    hook = attach_slot_hook(policy, slot_head, storage)

    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    total_ok = 0
    total_n  = 0
    routing_ok = 0    # layouts where ≥2 distinct slot predictions across 4 prompts

    for code in layouts:
        layout     = normalize_layout(code)
        global_idx = pick_layout_frame(ds, code)
        obs        = frame_to_obs(ds, global_idx)
        cases      = [c for c in make_cases(layout) if c.family == "relative"]

        predictions = []
        all_logits  = []

        print(f"{'='*65}")
        print(f"  Layout: {code}  ({' | '.join(layout)})")
        print(f"  Frame:  global_idx={global_idx}")
        print(f"{'='*65}")
        print(f"  {'prompt key':22s}  {'exp':7s}  {'pred':7s}  {'ok':3s}  "
              f"logits[L  M  R]")
        print(f"  {'-'*62}")

        for case in cases:
            pred_id, logits = run_prompt(policy, pre, post, obs, case.prompt, storage, device)
            pred_str = ID_TO_SLOT.get(pred_id, "?")
            ok = pred_str == case.expected_slot
            total_ok += int(ok)
            total_n  += 1
            predictions.append(pred_str)
            all_logits.append(logits)

            ok_sym = "✓" if ok else "✗"
            lstr = f"[{logits[0]:+.2f} {logits[1]:+.2f} {logits[2]:+.2f}]"
            print(f"  {case.key:22s}  {case.expected_slot:7s}  {pred_str:7s}  "
                  f"{ok_sym:3s}  {lstr}")

        # Key check: are predictions uniform (stuck) or varied (routing)?
        unique_preds = set(predictions)
        is_routing   = len(unique_preds) > 1
        routing_ok  += int(is_routing)

        print()
        if is_routing:
            print(f"  ✓ ROUTING: {predictions} — {len(unique_preds)} distinct "
                  f"predictions across 4 prompts")
        else:
            print(f"  ✗ STUCK:   {predictions} — all prompts map to '{predictions[0]}'")

        # Show logit spread: how much do logits vary by prompt?
        logit_matrix = np.stack(all_logits)   # (4, 3)
        spread = logit_matrix.max(axis=0) - logit_matrix.min(axis=0)
        print(f"  Logit spread across prompts: "
              f"left={spread[0]:.2f}  mid={spread[1]:.2f}  right={spread[2]:.2f}")
        print()

    print(f"{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Accuracy:        {total_ok}/{total_n}  "
          f"({total_ok/max(1,total_n)*100:.1f}%)")
    print(f"  Routing layouts: {routing_ok}/{len(layouts)}  "
          f"(layouts with ≥2 distinct predictions)")
    print()
    if routing_ok == len(layouts):
        print("  ✓ All probed layouts are ROUTING by prompt.")
        print("    Continue training — check global gate at 500/1000/2000.")
    elif routing_ok > 0:
        print(f"  ~ {routing_ok}/{len(layouts)} layouts routing; "
              f"others still stuck.")
        print("    Continue training — routing is breaking through.")
    else:
        print("  ✗ All layouts STUCK (same prediction regardless of prompt).")
        print("    Check training loss — if not decreasing, LoRA may not be updating.")

    hook.remove()


if __name__ == "__main__":
    main()
