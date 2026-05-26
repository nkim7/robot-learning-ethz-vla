#!/usr/bin/env python3
"""Build visible Eval 2 target-grounding VQA JSONL data.

Data source: robot-learning-group47/eval2_compositional_augmented.

The dataset is treated as 108 underlying real trajectories plus augmented
prompt/layout episodes. Splits are by physical layout or original trajectory,
never by random augmented episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eval2_layout_utils import Eval2ComboMap, prompt_type, resolve_prompt_target_color, target_slot

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


INITIAL_FRAC = 0.05
ROUTING_WINDOW = (0.58, 0.78)
ROUTING_CANDIDATES = 9

HSV_THRESHOLDS = {
    "red": ((0, 55), (0.22, 1.0), (0.18, 1.0)),
    "green": ((80, 155), (0.18, 1.0), (0.12, 1.0)),
    "blue": ((145, 235), (0.18, 1.0), (0.12, 1.0)),
}


def image_from_sample(sample: dict) -> Image.Image:
    key = "observation.images.front"
    if key not in sample:
        key = next((k for k in sample.keys() if "image" in k and "empty" not in k), None)
    if key is None:
        raise KeyError("No image key found in sample")
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


def visible_bowls(image: Image.Image, min_pixels: int) -> list[str]:
    hsv = np.asarray(image.convert("HSV"))
    hue = hsv[:, :, 0].astype(np.float32) * 360.0 / 255.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    visible = []
    for color, ((h0, h1), (s0, s1), (v0, v1)) in HSV_THRESHOLDS.items():
        if color == "red":
            hmask = (hue >= 340) | (hue <= h1)
        else:
            hmask = (hue >= h0) & (hue <= h1)
        mask = hmask & (sat >= s0) & (sat <= s1) & (val >= v0) & (val <= v1)
        if int(mask.sum()) >= min_pixels:
            visible.append(color)
    return sorted(visible)


def answer_text(target_color_value: str, target_slot_value: str) -> str:
    return f"target color: {target_color_value}\ntarget slot: {target_slot_value}"


def split_for_rank(rank: int, train_cut: int, val_cut: int) -> str:
    if rank < train_cut:
        return "train"
    if rank < val_cut:
        return "val"
    return "test"


def split_cuts(num_groups: int, train_frac: float, val_frac: float) -> tuple[int, int]:
    if num_groups < 3:
        raise ValueError("Need at least 3 split groups to create train/val/test splits")
    n_val = max(1, int(round(num_groups * val_frac)))
    n_test = max(1, num_groups - int(round(num_groups * (train_frac + val_frac))))
    if n_val + n_test >= num_groups:
        n_train = max(1, num_groups - 2)
        n_val = 1
        n_test = num_groups - n_train - n_val
    else:
        n_train = num_groups - n_val - n_test
    return n_train, n_train + n_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--root", default="/data/lerobot_cache/robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--out-dir", default="/data/branchphase_vqa")
    parser.add_argument("--n-real-demos", type=int, default=108)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--split-by", choices=["layout", "orig_demo"], default="layout")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--min-color-pixels", type=int, default=120)
    parser.add_argument("--allow-partial-routing-visibility", action="store_true")
    parser.add_argument("--counterfactual-initial", action="store_true")
    parser.add_argument("--dedupe-initial-images", action="store_true")
    parser.add_argument("--disable-routing", action="store_true")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    random.seed(args.seed)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.repo_id}")
    ds = LeRobotDataset(repo_id=args.repo_id, root=root, video_backend="pyav")
    print(f"  frames={len(ds)} episodes={ds.num_episodes}")
    task_prompts = list(ds.meta.tasks.reset_index()["task"])

    combo = Eval2ComboMap(root, ds.num_episodes)
    n_eps = ds.num_episodes if args.max_episodes <= 0 else min(ds.num_episodes, args.max_episodes)
    all_meta = [combo.episode(ep_idx) for ep_idx in range(n_eps)]
    if args.split_by == "layout":
        split_keys = sorted({m.layout_id for m in all_meta})
    else:
        split_keys = list(range(args.n_real_demos))
    random.shuffle(split_keys)
    train_cut, val_cut = split_cuts(len(split_keys), args.train_frac, args.val_frac)
    split_rank = {key: rank for rank, key in enumerate(split_keys)}

    ep_meta = ds.meta.episodes
    from_indices = list(ep_meta["dataset_from_index"])
    to_indices = list(ep_meta["dataset_to_index"])

    rows = []
    seen_initial_hashes: set[str] = set()
    for ep_idx in range(n_eps):
        ep_start = int(from_indices[ep_idx])
        ep_end = int(to_indices[ep_idx])
        ep_len = ep_end - ep_start
        meta = all_meta[ep_idx]
        split_key = meta.layout_id if args.split_by == "layout" else meta.orig_episode_index
        split = split_for_rank(split_rank[split_key], train_cut, val_cut)

        first = ds[ep_start]
        task = str(first.get("task", "")).strip()
        layout = meta.layout_left_to_right
        # Validate the original prompt resolves to the same target as combo metadata.
        prompt_target_color = resolve_prompt_target_color(task, layout)
        if prompt_target_color != meta.target_color:
            raise ValueError(
                f"Prompt/combo target mismatch for episode {ep_idx}: "
                f"prompt={task!r} layout={layout} prompt_target={prompt_target_color} combo_target={meta.target_color}"
            )
        tgt_slot = target_slot(layout, meta.target_color)
        answer = answer_text(meta.target_color, tgt_slot)

        candidates = []
        initial_frame = min(max(int(round(ep_len * INITIAL_FRAC)), 0), ep_len - 1)
        candidates.append(("initial_visible", initial_frame, initial_frame))

        if not args.disable_routing:
            lo = min(max(int(round(ep_len * ROUTING_WINDOW[0])), 0), ep_len - 1)
            hi = min(max(int(round(ep_len * ROUTING_WINDOW[1])), lo), ep_len - 1)
            best_routing = None
            for local_frame in np.linspace(lo, hi, ROUTING_CANDIDATES).round().astype(int).tolist():
                global_idx = ep_start + int(local_frame)
                sample = ds[global_idx]
                image = image_from_sample(sample)
                visible = visible_bowls(image, args.min_color_pixels)
                visibility_ok = set(visible) == {"blue", "green", "red"}
                if visibility_ok or (args.allow_partial_routing_visibility and len(visible) >= 2):
                    best_routing = (int(local_frame), visible, visibility_ok)
                    break
            if best_routing is not None:
                candidates.append(("routing_visible", best_routing[0], int(round(ep_len * 0.68))))

        for phase, local_frame, candidate_local_frame in candidates:
            global_idx = ep_start + local_frame
            sample = ds[global_idx]
            image = image_from_sample(sample)
            visible = visible_bowls(image, args.min_color_pixels)
            visibility_ok = set(visible) == {"blue", "green", "red"}
            if phase == "initial_visible" and not visibility_ok:
                # Initial frames should be clean. Skip if they do not show the whole layout.
                continue
            image_path = image_dir / f"ep{ep_idx:06d}_f{local_frame:04d}_{phase}.jpg"
            image.save(image_path, quality=95)
            image_md5 = hashlib.md5(image_path.read_bytes()).hexdigest()
            if phase == "initial_visible" and args.dedupe_initial_images:
                if image_md5 in seen_initial_hashes:
                    continue
                seen_initial_hashes.add(image_md5)

            if phase == "initial_visible" and args.counterfactual_initial:
                prompt_rows = []
                for prompt in task_prompts:
                    try:
                        prompt_target_color = resolve_prompt_target_color(prompt, layout)
                    except ValueError:
                        continue
                    prompt_rows.append((prompt, prompt_target_color, target_slot(layout, prompt_target_color)))
            else:
                prompt_rows = [(task, meta.target_color, tgt_slot)]

            for prompt, prompt_target_color, prompt_target_slot in prompt_rows:
                rows.append(
                    {
                        "image": str(image_path),
                        "image_or_clip": str(image_path),
                        "image_md5": image_md5,
                        "episode_index": ep_idx,
                        "orig_episode_index": meta.orig_episode_index,
                        "layout_id": meta.layout_id,
                        "local_frame": local_frame,
                        "candidate_local_frame": candidate_local_frame,
                        "phase": phase,
                        "visible_bowls": visible,
                        "visibility_ok": visibility_ok,
                        "prompt_type": prompt_type(prompt),
                        "split": split,
                        "prompt": prompt,
                        "layout_left_to_right": layout,
                        "target_color": prompt_target_color,
                        "target_slot": prompt_target_slot,
                        "answer": answer_text(prompt_target_color, prompt_target_slot),
                        "prompt_source": "counterfactual_original_eval2_prompt"
                        if phase == "initial_visible" and args.counterfactual_initial
                        else "episode_task",
                    }
                )

        if (ep_idx + 1) % 50 == 0:
            print(f"  processed {ep_idx + 1}/{n_eps}")

    out_jsonl = out_dir / "branchphase_vqa.jsonl"
    with out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "repo_id": args.repo_id,
        "root": str(root),
        "num_rows": len(rows),
        "num_episodes": n_eps,
        "split_by": args.split_by,
        "splits": pd.Series([r["split"] for r in rows]).value_counts().to_dict(),
        "data_source": args.repo_id,
        "frame_policy": "single frames: initial_visible plus visibility-filtered routing_visible",
        "counterfactual_initial": args.counterfactual_initial,
        "dedupe_initial_images": args.dedupe_initial_images,
        "disable_routing": args.disable_routing,
        "phases": pd.Series([r["phase"] for r in rows]).value_counts().to_dict(),
        "visibility_ok": pd.Series([r["visibility_ok"] for r in rows]).value_counts().to_dict(),
        "prompt_types": pd.Series([r["prompt_type"] for r in rows]).value_counts().to_dict(),
        "target_slots": pd.Series([r["target_slot"] for r in rows]).value_counts().to_dict(),
        "unique_images": len({r.get("image_md5", r["image"]) for r in rows}),
        "output": str(out_jsonl),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
