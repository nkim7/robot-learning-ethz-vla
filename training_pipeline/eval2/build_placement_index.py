#!/usr/bin/env python3
"""
build_placement_index.py — eval2_pathC_lora_placement_grouped_v1

Builds two files needed by the grouped-placement sampler:

  1. placement_index.csv
     Columns: global_idx, episode_index, local_frame, task, target_slot, layout_id

  2. layout_groups.csv
     Columns: layout_id, left_pool, middle_pool, right_pool
     Pools are semicolon-separated lists of augmented episode indices.

Usage (run on the training server after downloading the dataset):

    python build_placement_index.py \
        --repo_id robot-learning-group47/eval2_compositional_augmented \
        --dataset_root /data/lerobot_cache/robot-learning-group47/eval2_compositional_augmented \
        --out_dir /data/lerobot_outputs/sampler_indices \
        [--chunk_size 50] \
        [--audit_frames 20]

After running, visually audit:
    placement_index_audit.png  (20 randomly sampled placement frames saved side-by-side)
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Placement window heuristic ────────────────────────────────────────────────
# For a full 225-frame episode: frames 90–170 are placement-heavy.
# For shorter episodes: 45%–80% of the episode length.
PLACEMENT_START_FRAC = 0.40   # start at 40% of episode  (conservative: ≥90 for 225f)
PLACEMENT_END_FRAC   = 0.80   # end   at 80% of episode  (≤180 for 225f)
PLACEMENT_START_ABS  = 90     # absolute minimum local frame
PLACEMENT_END_ABS    = 170    # absolute maximum local frame

# Target-slot keyword mapping
# We derive target_slot (left / middle / right) from the task string.
# Color → slot depends on the layout, so we handle BOTH:
#   (a) explicit position prompts ("left bowl", "1st from left", etc.)
#   (b) color prompts (red/green/blue) — these need layout metadata.
# For color-only prompts WITHOUT layout metadata, we keep target_slot="unknown".
SLOT_KEYWORDS: dict[str, list[str]] = {
    "left": [
        "left bowl", "1st bowl from the left", "first bowl from the left",
        "3rd bowl from the right", "third bowl from the right",
        "bowl on the left of", "leftmost",
    ],
    "middle": [
        "middle bowl", "2nd bowl from the left", "second bowl from the left",
        "2nd bowl from the right", "second bowl from the right",
        "center bowl",
    ],
    "right": [
        "right bowl", "3rd bowl from the left", "third bowl from the left",
        "1st bowl from the right", "first bowl from the right",
        "bowl on the right of", "rightmost",
    ],
}

# Color → canonical target slot for each known layout.
# Key = frozenset of bowl colors in the layout (e.g. {"red", "green", "blue"}).
# Value = dict[color, slot].
# These come from the real Eval2 demo metadata.
# If the dataset has layout metadata in parquet, we use that instead.
COLOR_TO_SLOT_BY_LAYOUT: dict[frozenset, dict[str, str]] = {
    # Example layouts — update from actual dataset metadata.
    # frozenset({"red", "green", "blue"}): {"red": "left", "green": "middle", "blue": "right"},
}


def infer_slot_from_task(task: str) -> str:
    """Infer target slot from task string using keyword matching."""
    t = task.lower()
    for slot, keywords in SLOT_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return slot
    # Negation patterns: "not X and not Y" → the remaining one
    if "not" in t:
        colors_not = []
        for color in ("red", "green", "blue"):
            if f"not {color}" in t:
                colors_not.append(color)
        colors_all = {"red", "green", "blue"}
        remaining = colors_all - set(colors_not)
        if len(remaining) == 1:
            color = next(iter(remaining))
            # Slot from color depends on layout — mark as color-derived
            return f"color:{color}"
    # Color-only prompts
    for color in ("red", "green", "blue"):
        if color in t and "not" not in t:
            return f"color:{color}"
    return "unknown"


def placement_window(ep_length: int, chunk_size: int) -> tuple[int, int]:
    """Return (start_local, end_local) for placement-phase sampling in an episode."""
    start = max(
        int(ep_length * PLACEMENT_START_FRAC),
        PLACEMENT_START_ABS,
    )
    end = min(
        int(ep_length * PLACEMENT_END_FRAC),
        PLACEMENT_END_ABS,
        ep_length - chunk_size,   # must have a full action chunk available
    )
    return start, end


def build_layout_id(task_base: str, ep_idx: int, n_real_demos: int = 108) -> str:
    """
    Heuristic layout_id: group every 3 consecutive real demos (left/middle/right)
    into one layout. Augmented episodes inherit from their source real demo.

    If the dataset has explicit layout metadata, replace this with that.
    For eval2_compositional_augmented with 108 real demos → 36 layouts.
    Each real demo has ~6.7 augmented variants → episodes 0-719 (approx).

    Mapping:
        real_demo_idx = ep_idx % n_real_demos   (for augmented dataset)
        group = real_demo_idx // 3
        slot_within = real_demo_idx % 3  → 0=left, 1=middle, 2=right
    """
    real_demo_idx = ep_idx % n_real_demos
    layout_id = f"layout_{real_demo_idx // 3:03d}"
    return layout_id


def slot_within_layout(ep_idx: int, n_real_demos: int = 108) -> str:
    real_demo_idx = ep_idx % n_real_demos
    return ["left", "middle", "right"][real_demo_idx % 3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build placement index and layout groups for grouped sampler.")
    ap.add_argument("--repo_id", default="robot-learning-group47/eval2_compositional_augmented")
    ap.add_argument("--dataset_root", required=True, help="Local cache path for the dataset")
    ap.add_argument("--out_dir", required=True, help="Where to write placement_index.csv and layout_groups.csv")
    ap.add_argument("--chunk_size", type=int, default=50, help="Action chunk size (for valid-frame check)")
    ap.add_argument("--audit_frames", type=int, default=20, help="How many frames to audit")
    ap.add_argument("--n_real_demos", type=int, default=108, help="Number of real Eval2 demos (for layout ID mapping)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.repo_id}")
    print(f"  root: {args.dataset_root}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    # ── Episode boundaries ────────────────────────────────────────────────────
    # Reconstruct from_index / to_index from dataset metadata
    ep_meta = ds.meta.episodes  # DataFrame with dataset_from_index, dataset_to_index
    from_indices = ep_meta["dataset_from_index"].tolist()
    to_indices   = ep_meta["dataset_to_index"].tolist()

    print(f"\nBuilding placement index ...")
    rows = []
    for ep_idx in range(ds.num_episodes):
        ep_start = from_indices[ep_idx]
        ep_end   = to_indices[ep_idx]
        ep_length = ep_end - ep_start

        # Get task string from first frame of episode
        first_frame = ds[ep_start]
        task = first_frame.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0]
        task = str(task).strip()

        # Determine target slot
        # First: use layout-based structural mapping
        slot = slot_within_layout(ep_idx, n_real_demos=args.n_real_demos)
        layout_id = build_layout_id(task, ep_idx, n_real_demos=args.n_real_demos)

        # Determine placement window
        pstart, pend = placement_window(ep_length, args.chunk_size)
        if pstart >= pend:
            # Episode too short for placement window — skip
            continue

        # Emit one row per valid placement frame
        for local_frame in range(pstart, pend + 1):
            global_idx = ep_start + local_frame
            if global_idx + args.chunk_size > ep_end:
                # Not enough frames for a full action chunk
                break
            rows.append({
                "global_idx":    global_idx,
                "episode_index": ep_idx,
                "local_frame":   local_frame,
                "task":          task,
                "target_slot":   slot,
                "layout_id":     layout_id,
            })

    placement_df = pd.DataFrame(rows)
    placement_path = out_dir / "placement_index.csv"
    placement_df.to_csv(placement_path, index=False)
    print(f"  Placement index: {len(placement_df)} valid frames across {placement_df['episode_index'].nunique()} episodes")
    print(f"  Written to: {placement_path}")

    # ── Layout group mapping ──────────────────────────────────────────────────
    print(f"\nBuilding layout group mapping ...")
    # Group by layout_id + target_slot → collect episode_index sets
    layout_records = []
    for layout_id, grp in placement_df.groupby("layout_id"):
        pools: dict[str, list[int]] = {"left": [], "middle": [], "right": []}
        for slot in ("left", "middle", "right"):
            eps = grp[grp["target_slot"] == slot]["episode_index"].unique().tolist()
            pools[slot] = sorted(eps)
        # Only keep layouts that have ALL three slots non-empty
        if all(len(pools[s]) > 0 for s in ("left", "middle", "right")):
            layout_records.append({
                "layout_id":   layout_id,
                "left_pool":   ";".join(map(str, pools["left"])),
                "middle_pool": ";".join(map(str, pools["middle"])),
                "right_pool":  ";".join(map(str, pools["right"])),
            })

    layout_df = pd.DataFrame(layout_records)
    layout_path = out_dir / "layout_groups.csv"
    layout_df.to_csv(layout_path, index=False)
    print(f"  Layout groups: {len(layout_df)} complete layouts (left+middle+right)")
    print(f"  Written to: {layout_path}")

    # ── Slot distribution ─────────────────────────────────────────────────────
    print(f"\nTarget slot distribution:")
    for slot, cnt in placement_df.groupby("target_slot").size().items():
        print(f"  {slot:10s}: {cnt:6d} frames")

    # ── Placement audit ───────────────────────────────────────────────────────
    print(f"\nSaving {args.audit_frames} audit frames ...")
    audit_dir = out_dir / "placement_audit"
    audit_dir.mkdir(exist_ok=True)

    audit_rows = placement_df.sample(min(args.audit_frames, len(placement_df)), random_state=42)
    audit_log = []
    for i, (_, row) in enumerate(audit_rows.iterrows()):
        fidx = int(row["global_idx"])
        sample = ds[fidx]

        # Get image
        img_key = "observation.images.front"
        if img_key not in sample:
            # Try camera1
            img_key = next((k for k in sample if "image" in k and "empty" not in k), None)

        audit_entry = {
            "audit_idx":    i,
            "global_idx":   fidx,
            "episode_index": int(row["episode_index"]),
            "local_frame":  int(row["local_frame"]),
            "task":         row["task"][:80],
            "target_slot":  row["target_slot"],
            "layout_id":    row["layout_id"],
        }
        if "observation.state" in sample:
            state = sample["observation.state"].numpy()
            audit_entry["pan"]    = float(state[0])
            audit_entry["lift"]   = float(state[1])
            audit_entry["gripper"] = float(state[5])

        audit_log.append(audit_entry)

        # Save image if available
        if img_key and img_key in sample:
            try:
                import cv2
                img = sample[img_key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
                label = f"ep{int(row['episode_index'])} f{int(row['local_frame'])} {row['target_slot']}"
                cv2.putText(bgr, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imwrite(str(audit_dir / f"audit_{i:02d}_ep{int(row['episode_index'])}_f{int(row['local_frame'])}_{row['target_slot']}.jpg"), bgr)
            except Exception as e:
                print(f"    [warn] Could not save image for audit frame {i}: {e}")

    audit_df = pd.DataFrame(audit_log)
    audit_df.to_csv(out_dir / "placement_audit_log.csv", index=False)
    print(f"  Audit frames saved to: {audit_dir}/")
    print(f"  Audit log: {out_dir}/placement_audit_log.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PLACEMENT INDEX SUMMARY")
    print(f"{'='*60}")
    print(f"  Total placement frames : {len(placement_df)}")
    print(f"  Episodes with placement: {placement_df['episode_index'].nunique()}")
    print(f"  Complete layout groups : {len(layout_df)}")
    print(f"  Expected batch groups  : {len(layout_df)} (use ≥ 10 per batch for bs=30)")
    print(f"\n  Files written:")
    print(f"    {placement_path}")
    print(f"    {layout_path}")
    print(f"\n  Next step: run sampler audit with:")
    print(f"    python train_eval2_pathC_lora_placement_grouped_v1.py --audit_only \\")
    print(f"      --placement_index {placement_path} \\")
    print(f"      --layout_groups {layout_path}")


if __name__ == "__main__":
    main()
