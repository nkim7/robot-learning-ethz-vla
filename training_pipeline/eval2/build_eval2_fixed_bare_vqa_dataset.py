#!/usr/bin/env python3
"""Build bare-prompt VQA data from eval2_fixed_bowl_fixed_banana_108.

Visual source:
  robot-learning-group47/eval2_fixed_bowl_fixed_banana_108

Frame source:
  first frame of each real episode. This is the initial-visible decision frame.

Text input:
  bare Eval2-style instruction only, with no system prompt and no output-format
  instruction. Rows include counterfactual prompts for the same image/layout.

Text target:
  target color: <blue|red|green>
  target slot: <left|middle|right>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


DATASET_ID = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
COLOR_ABBR = {"b": "blue", "g": "green", "r": "red"}
SLOTS = ("left", "middle", "right")

PROMPT_VARIANTS = [
    ("ordinal", "Put the banana into the 2nd bowl from the left from the robot perspective"),
    ("ordinal", "Put the banana into the second bowl from the left from the robot perspective"),
    ("ordinal", "Put the banana into the middle bowl from the robot perspective"),
    ("ordinal", "Put the banana into the center bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl on the right of the red bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl to the right of the red bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl immediately right of the red bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl just to the right of the red bowl from the robot perspective"),
    ("negation", "Put the banana into the bowl that is not green and not blue."),
    ("negation", "Put the banana into the bowl that is not blue and not green."),
    ("negation", "Put the banana into the bowl that is neither green nor blue."),
    ("negation", "Put the banana into the non-green and non-blue bowl."),
]


def to_pil_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.numpy()
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return Image.fromarray(arr.astype("uint8")).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(value)}")


def answer_text(color: str, slot: str, key: str) -> str:
    """Return one of several semantically equivalent answer styles.

    The goal is target understanding, not a rigid output template.
    """
    styles = [
        f"{color}, {slot}",
        f"{slot}",
        f"The target color is {color} and the target slot is {slot}.",
        f"{{\"target_color\": \"{color}\", \"target_slot\": \"{slot}\"}}",
        f"{slot} slot, {color} bowl",
        f"Go to the {slot} bowl. It is {color}.",
    ]
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return styles[int(digest[:8], 16) % len(styles)]


def resolve(prompt: str, layout: list[str]) -> tuple[str, str] | None:
    text = prompt.lower()
    if (
        "2nd bowl from the left" in text
        or "second bowl from the left" in text
        or "middle bowl" in text
        or "center bowl" in text
    ):
        return layout[1], "middle"
    if "right of the red bowl" in text:
        red_idx = layout.index("red")
        target_idx = red_idx + 1
        if target_idx >= len(layout):
            return None
        return layout[target_idx], SLOTS[target_idx]
    if (
        "not green and not blue" in text
        or "not blue and not green" in text
        or "neither green nor blue" in text
        or "non-green and non-blue" in text
    ):
        return "red", SLOTS[layout.index("red")]
    raise ValueError(prompt)


def load_episode_truth(dataset_id: str) -> dict[int, dict]:
    map_path = hf_hub_download(
        dataset_id,
        "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv",
        repo_type="dataset",
    )
    combo_df = pd.read_csv(map_path)
    truth = {}
    for _, row in combo_df.iterrows():
        layout = [COLOR_ABBR[ch] for ch in str(row["bowl_order_left_to_right"])]
        combo = str(row["combo"])
        for episode_index in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
            truth[episode_index] = {
                "combo": combo,
                "layout_code": str(row["bowl_order_left_to_right"]),
                "layout_left_to_right": layout,
                "target_bowl_color_demo": str(row["target_bowl_color"]),
                "replicate_index": episode_index - int(row["episode_start"]),
            }
    return truth


def split_for_replicate(rep: int) -> str:
    if rep <= 3:
        return "train"
    if rep == 4:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval2_fixed_bare_vqa"))
    parser.add_argument("--image-quality", type=int, default=95)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.dataset)
    truth = load_episode_truth(args.dataset)
    rows: list[dict] = []

    for ep in dataset.meta.episodes:
        episode_index = int(ep["episode_index"])
        first_index = int(ep["dataset_from_index"])
        item = dataset[first_index]
        info = truth[episode_index]
        split = split_for_replicate(int(info["replicate_index"]))
        image_path = image_dir / f"ep{episode_index:03d}_{info['combo']}_frame000.jpg"
        to_pil_image(item["observation.images.front"]).save(image_path, quality=args.image_quality)

        for prompt_type, prompt in PROMPT_VARIANTS:
            resolved = resolve(prompt, info["layout_left_to_right"])
            if resolved is None:
                continue
            target_color, target_slot = resolved
            rows.append(
                {
                    "image": str(image_path),
                    "image_or_clip": str(image_path),
                    "source_dataset": args.dataset,
                    "episode_index": episode_index,
                    "orig_episode_index": episode_index,
                    "global_index": first_index,
                    "local_frame": 0,
                    "phase": "initial_visible",
                    "combo": info["combo"],
                    "layout_code": info["layout_code"],
                    "layout_id": "-".join(info["layout_left_to_right"]),
                    "layout_left_to_right": info["layout_left_to_right"],
                    "demo_target_color": info["target_bowl_color_demo"],
                    "replicate_index": info["replicate_index"],
                    "split": split,
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "target_color": target_color,
                    "target_slot": target_slot,
                    "answer": answer_text(target_color, target_slot, f"{episode_index}:{prompt}"),
                }
            )

    jsonl = args.out_dir / "vqa.jsonl"
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    metadata = {
        "dataset": args.dataset,
        "frame_policy": "first frame per episode",
        "text_input_policy": "bare Eval2-style prompt only; no system prompt and no answer-format instruction",
        "split_policy": "within each combo, replicate_index 0-3 train, 4 val, 5 test",
        "num_rows": len(rows),
        "num_images": len(list(image_dir.glob("*.jpg"))),
        "counts": {
            "split": dict(Counter(row["split"] for row in rows)),
            "prompt_type": dict(Counter(row["prompt_type"] for row in rows)),
            "target_slot": dict(Counter(row["target_slot"] for row in rows)),
            "layout_id": dict(Counter(row["layout_id"] for row in rows)),
        },
        "prompt_variants": [prompt for _, prompt in PROMPT_VARIANTS],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {len(rows)} rows to {jsonl}")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
