#!/usr/bin/env python3
"""Build frame-level target-branching indices for Eval 2 VLA routing training."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from eval2_layout_utils import COLOR_NAMES, layout_id, normalize_layout, prompt_type, resolve_prompt_target_color, target_slot

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PHASE_WINDOWS = {
    "post_grasp": (0.38, 0.50),
    "bowl_visible": (0.55, 0.66),
    "branch_precommit": (0.68, 0.76),
}


def load_fixed_combo_map(repo_id: str) -> dict[int, dict]:
    path = hf_hub_download(
        repo_id,
        "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path)
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        layout = normalize_layout(str(row["bowl_order_left_to_right"]))
        color = str(row["target_bowl_color"]).lower()
        start = int(row["episode_start"])
        for ep in range(start, int(row["episode_end"]) + 1):
            out[ep] = {
                "orig_episode_index": ep,
                "layout_left_to_right": layout,
                "layout_id": layout_id(layout),
                "target_color": color,
                "target_slot": target_slot(layout, color),
            }
    return out


def load_augmented_combo_map(root: Path, total_episodes: int):
    from eval2_layout_utils import Eval2ComboMap

    combo = Eval2ComboMap(root, total_episodes)
    out = {}
    for ep in range(total_episodes):
        meta = combo.episode(ep)
        out[ep] = {
            "orig_episode_index": meta.orig_episode_index,
            "layout_left_to_right": meta.layout_left_to_right,
            "layout_id": meta.layout_id,
            "target_color": meta.target_color,
            "target_slot": meta.target_slot,
        }
    return out


def image_from_sample(sample: dict) -> Image.Image:
    key = "observation.images.front"
    if key not in sample:
        key = next((k for k in sample if "image" in k and "empty" not in k), None)
    if key is None:
        raise KeyError("No image key found")
    image = sample[key]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if hasattr(image, "numpy"):
        image = image.numpy()
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="robot-learning-group47/eval2_fixed_bowl_fixed_banana_108")
    parser.add_argument("--dataset-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase", choices=list(PHASE_WINDOWS), default="branch_precommit")
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--audit-frames", type=int, default=48)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    random.seed(args.seed)
    root = Path(args.dataset_root) if args.dataset_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_kwargs = {"video_backend": "pyav"}
    if root is not None:
        ds_kwargs["root"] = root
    ds = LeRobotDataset(args.repo_id, **ds_kwargs)
    if "eval2_fixed_bowl_fixed_banana_108" in args.repo_id:
        combo_by_episode = load_fixed_combo_map(args.repo_id)
    else:
        if root is None:
            raise ValueError("--dataset-root is required for augmented datasets with combo_augmented_map.csv")
        combo_by_episode = load_augmented_combo_map(root, ds.num_episodes)
    ep_meta = ds.meta.episodes
    starts = list(ep_meta["dataset_from_index"])
    ends = list(ep_meta["dataset_to_index"])

    rows = []
    n_eps = ds.num_episodes if args.max_episodes <= 0 else min(args.max_episodes, ds.num_episodes)
    start_frac, end_frac = PHASE_WINDOWS[args.phase]
    for ep_idx in range(n_eps):
        ep_start = int(starts[ep_idx])
        ep_end = int(ends[ep_idx])
        ep_len = ep_end - ep_start
        max_local = ep_len - args.chunk_size
        if max_local <= 0:
            continue
        lo = min(max(int(round(ep_len * start_frac)), 0), max_local)
        hi = min(max(int(round(ep_len * end_frac)), lo), max_local)
        if hi < lo:
            continue
        if args.frames_per_episode <= 1:
            local_frames = [(lo + hi) // 2]
        else:
            local_frames = sorted(set(np.linspace(lo, hi, args.frames_per_episode).round().astype(int).tolist()))

        meta = combo_by_episode[ep_idx]
        task = str(ds[ep_start].get("task", "")).strip()
        resolved = resolve_prompt_target_color(task, meta["layout_left_to_right"])
        if resolved != meta["target_color"]:
            raise ValueError(f"Prompt/combo mismatch ep={ep_idx} prompt={task!r} resolved={resolved} meta={meta}")
        for local_frame in local_frames:
            rows.append(
                {
                    "global_index": ep_start + int(local_frame),
                    "episode_index": ep_idx,
                    "orig_episode_index": meta["orig_episode_index"],
                    "layout_id": meta["layout_id"],
                    "layout_left_to_right": json.dumps(meta["layout_left_to_right"]),
                    "target_slot": meta["target_slot"],
                    "target_color": meta["target_color"],
                    "local_frame": int(local_frame),
                    "phase": args.phase,
                    "prompt_type": prompt_type(task),
                    "prompt": task,
                    "valid_action_chunk": True,
                }
            )

    index = pd.DataFrame(rows)
    index_path = out_dir / "target_branching_index.csv"
    index.to_csv(index_path, index=False)

    group_rows = []
    for layout_id, grp in index.groupby("layout_id"):
        pools = {}
        for slot in ("left", "middle", "right"):
            eps = sorted(grp[grp["target_slot"] == slot]["episode_index"].unique().tolist())
            pools[slot] = eps
        if all(pools[s] for s in ("left", "middle", "right")):
            group_rows.append(
                {
                    "layout_id": layout_id,
                    "left_pool": ";".join(map(str, pools["left"])),
                    "middle_pool": ";".join(map(str, pools["middle"])),
                    "right_pool": ";".join(map(str, pools["right"])),
                }
            )
    groups = pd.DataFrame(group_rows)
    group_path = out_dir / "target_layout_groups.csv"
    groups.to_csv(group_path, index=False)

    audit_dir = out_dir / "target_branching_audit"
    audit_dir.mkdir(exist_ok=True)
    for i, row in index.sample(min(args.audit_frames, len(index)), random_state=args.seed).iterrows():
        img = image_from_sample(ds[int(row["global_index"])])
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 32), fill=(255, 255, 255))
        draw.text((4, 4), f"{row['phase']} {row['target_slot']} {row['prompt'][:58]}", fill=(0, 0, 0))
        img.save(audit_dir / f"{int(row['global_index']):08d}_{row['target_slot']}.jpg", quality=92)

    summary = {
        "rows": len(index),
        "episodes": int(index["episode_index"].nunique()) if len(index) else 0,
        "complete_layout_groups": len(groups),
        "slot_counts": index["target_slot"].value_counts().to_dict() if len(index) else {},
        "prompt_type_counts": index["prompt_type"].value_counts().to_dict() if len(index) else {},
        "index": str(index_path),
        "groups": str(group_path),
        "audit_dir": str(audit_dir),
    }
    (out_dir / "target_branching_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
