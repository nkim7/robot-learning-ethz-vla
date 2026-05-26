#!/usr/bin/env python3
"""Action-direction probe for D2_v5 (and any SmolVLA checkpoint).

Tests whether the policy differentiates between relational prompts that map
to different target slots in the same layout.

Usage:
    python3 probe_eval2_stage2d_v5_action_direction.py \
        --ckpt /home/shadeform/outputs/eval2_stage2_20k_relation_cotrain_d2_v5_lora_connector_safe/checkpoints/000200/pretrained_model \
        --dataset-root /home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only

The probe uses real dataset frames (layout = blue|red|green, middle demo frame)
and applies all 4 priority prompts, comparing the predicted action chunks.

Output:
  - Mean action target position estimate per prompt
  - Pairwise cosine similarity between action chunks
  - Pass/fail whether each prompt maps to a different region of action space
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/shadeform")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ── Priority prompts for probe layout: blue | red | green ──────────────────

PROBE_LAYOUT   = "blue | red | green"   # left=blue, middle=red, right=green

PROBE_PROMPTS = [
    {
        "label":    "right-of-blue → middle/red",
        "text":     "Put the banana into the bowl on the right of the blue bowl from the robot perspective.",
        "expected": "middle",
    },
    {
        "label":    "left-of-green → middle/red",
        "text":     "Put the banana into the bowl on the left of the green bowl from the robot perspective.",
        "expected": "middle",
    },
    {
        "label":    "left-of-red → left/blue",
        "text":     "Put the banana into the bowl on the left of the red bowl from the robot perspective.",
        "expected": "left",
    },
    {
        "label":    "right-of-red → right/green",
        "text":     "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
        "expected": "right",
    },
]

# Layout code for blue|red|green = "brg"
TARGET_LAYOUT_CODE = "brg"

# Arm joint 0 (shoulder pan) typically discriminates left/middle/right
# Final action chunk position [joint 0] used as a proxy for target direction
PROBE_JOINT_IDX = 0


def find_probe_frames(dataset_root: Path, layout_code: str = TARGET_LAYOUT_CODE,
                      frame_idx: int = 30) -> list[int]:
    """Find global indices of frames from target layout, near mid-demo."""
    import csv
    combo_csv = dataset_root / "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
    episode_ranges = []
    with combo_csv.open() as f:
        for row in csv.DictReader(f):
            code = row["bowl_order_left_to_right"].strip().lower()
            if code == layout_code:
                ep_start = int(row["episode_start"])
                ep_end   = int(row["episode_end"])
                episode_ranges.append((ep_start, ep_end))
    return episode_ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         required=True,  help="Path to pretrained_model dir")
    ap.add_argument("--dataset",      default="robot-learning-group47/eval2_108_position_only")
    ap.add_argument("--dataset-root", default="/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only")
    ap.add_argument("--frame-idx",    type=int, default=30, help="Episode frame to use as observation")
    ap.add_argument("--n-episodes",   type=int, default=3,  help="Number of episodes per layout to probe")
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)

    print(f"Probe checkpoint: {ckpt_path}")
    print(f"Layout under test: {PROBE_LAYOUT}  (code={TARGET_LAYOUT_CODE})")
    print()

    # ── Load policy ──────────────────────────────────────────────────────────
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    policy = policy.to(device)
    policy.eval()

    # ── Load dataset ─────────────────────────────────────────────────────────
    ds_root = Path(args.dataset_root)
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=str(ds_root))
    delta_ts = resolve_delta_timestamps(policy.config, ds_meta)
    ds = LeRobotDataset(args.dataset, root=str(ds_root),
                        delta_timestamps=delta_ts, video_backend="pyav")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # ── Find probe frames ────────────────────────────────────────────────────
    episode_ranges = find_probe_frames(ds_root, TARGET_LAYOUT_CODE, args.frame_idx)
    if not episode_ranges:
        print(f"ERROR: No episodes found for layout code '{TARGET_LAYOUT_CODE}'")
        sys.exit(1)

    # Collect global frame indices close to frame_idx in target episodes
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    probe_global_indices = []
    for ep_start, ep_end in episode_ranges[:args.n_episodes]:
        best_gi   = None
        best_diff = 99999
        for gi, (ep, fr) in enumerate(zip(ep_vals, fr_vals)):
            if ep_start <= ep <= ep_end and abs(fr - args.frame_idx) < best_diff:
                best_diff = abs(fr - args.frame_idx)
                best_gi   = gi
        if best_gi is not None:
            probe_global_indices.append(best_gi)

    print(f"Found {len(probe_global_indices)} probe frame(s) (episode range: {episode_ranges[:args.n_episodes]})")
    print()

    # ── Run probe ────────────────────────────────────────────────────────────
    results = []  # (prompt_label, expected, chunk_mean_per_joint)

    for gi in probe_global_indices:
        ep = int(ep_vals[gi])
        fr = int(fr_vals[gi])
        print(f"  Episode {ep}, frame {fr}:")

        chunks_by_prompt = []
        for p in PROBE_PROMPTS:
            item = dict(ds[gi])
            item["task"] = p["text"]
            item.pop("task_index", None)

            # Scale images
            for key in ds.meta.camera_keys:
                if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
                    item[key] = item[key].float() / 255.0

            from torch.utils.data import default_collate
            tasks = [item.pop("task")]
            batch = default_collate([item])
            batch["task"] = tasks
            batch = preprocessor(batch)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                output = policy.select_action(batch)

            # output shape: [1, chunk_size, action_dim] or similar
            action = output.squeeze(0)  # [chunk_size, action_dim] or [action_dim]
            if action.dim() == 1:
                action = action.unsqueeze(0)  # treat as [1, action_dim]

            # Use last 10 frames of chunk as target direction proxy
            chunk_tail = action[-10:, :6].float().cpu()  # first 6 joints
            mean_pos   = chunk_tail.mean(dim=0)  # [6]

            chunks_by_prompt.append(mean_pos)
            print(f"    [{p['expected']:8s}] {p['label'][:55]:55s}  joint0={mean_pos[0]:.3f}")

        # Pairwise similarity between chunks for same vs different expected targets
        print()
        print("    Pairwise cosine similarity (should differ between different targets):")
        for i, pi in enumerate(PROBE_PROMPTS):
            for j, pj in enumerate(PROBE_PROMPTS):
                if j <= i:
                    continue
                sim = F.cosine_similarity(
                    chunks_by_prompt[i].unsqueeze(0),
                    chunks_by_prompt[j].unsqueeze(0),
                ).item()
                same = pi["expected"] == pj["expected"]
                print(f"      {pi['expected']:6s} vs {pj['expected']:6s}: sim={sim:.3f} {'✓same' if same else '✗diff'}")
        print()

        results.append(chunks_by_prompt)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — Joint-0 mean over all probe frames:")
    all_joint0 = [[chunks[i][0].item() for chunks in results] for i in range(len(PROBE_PROMPTS))]
    for i, p in enumerate(PROBE_PROMPTS):
        vals  = all_joint0[i]
        mean  = sum(vals) / len(vals) if vals else float("nan")
        print(f"  [{p['expected']:8s}] {p['label'][:55]:55s}  joint0={mean:.3f}")

    print()
    print("Pass criteria:")
    # middle (right-of-blue) and middle (left-of-green) should be similar
    # left (left-of-red) and right (right-of-red) should differ from each other and from middle
    j0_middle_rblue  = sum(all_joint0[0]) / len(all_joint0[0])
    j0_middle_lgreen = sum(all_joint0[1]) / len(all_joint0[1])
    j0_left_lred     = sum(all_joint0[2]) / len(all_joint0[2])
    j0_right_rred    = sum(all_joint0[3]) / len(all_joint0[3])

    same_middle_ok = abs(j0_middle_rblue - j0_middle_lgreen) < 0.05
    left_vs_right  = abs(j0_left_lred - j0_right_rred) > 0.05
    middle_vs_left = abs(j0_middle_rblue - j0_left_lred) > 0.05

    print(f"  Two 'middle' prompts agree:  {'PASS ✓' if same_middle_ok else 'FAIL ✗'}  (delta={abs(j0_middle_rblue-j0_middle_lgreen):.3f})")
    print(f"  left vs right differ:        {'PASS ✓' if left_vs_right  else 'FAIL ✗'}  (delta={abs(j0_left_lred-j0_right_rred):.3f})")
    print(f"  middle vs left differ:       {'PASS ✓' if middle_vs_left else 'FAIL ✗'}  (delta={abs(j0_middle_rblue-j0_left_lred):.3f})")


if __name__ == "__main__":
    main()
