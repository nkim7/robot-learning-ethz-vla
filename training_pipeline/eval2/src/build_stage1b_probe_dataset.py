#!/usr/bin/env python3
"""Build Stage 1B hidden-state probe samples for Eval2 target-slot probing."""

from __future__ import annotations

import argparse
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

SCAFFOLD_SYSTEM = (
    "You are looking at a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Given the image and the original robot instruction, identify the target bowl. "
    "Reply with exactly two short lines, not JSON and not prose. "
    "The exact format is: "
    "target color: green\n"
    "target slot: right"
)

PROMPTS = [
    ("color", "Put the banana in the blue colored bowl."),
    ("color", "Put the banana in the green colored bowl."),
    ("color", "Put the banana in the red colored bowl."),
    ("ordinal", "Put the banana into the 1st bowl from the left from the robot perspective."),
    ("ordinal", "Put the banana into the 2nd bowl from the left from the robot perspective."),
    ("ordinal", "Put the banana into the 3rd bowl from the left from the robot perspective."),
    ("ordinal", "Put the banana into the 1st bowl from the right from the robot perspective."),
    ("ordinal", "Put the banana into the 2nd bowl from the right from the robot perspective."),
    ("ordinal", "Put the banana into the 3rd bowl from the right from the robot perspective."),
    ("ordinal", "Put the banana into the middle bowl from the robot perspective."),
    ("ordinal", "Put the banana into the center bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl on the left of the blue bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl on the right of the blue bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl on the left of the green bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl on the right of the green bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl on the left of the red bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl on the right of the red bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl to the right of the red bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl immediately right of the red bowl from the robot perspective."),
    ("relative", "Put the banana into the bowl just to the right of the red bowl from the robot perspective."),
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


def split_for_replicate(rep: int) -> str:
    if rep <= 3:
        return "train"
    if rep == 4:
        return "val"
    return "test"


def scaffold_prompt(bare_prompt: str) -> str:
    return f"{SCAFFOLD_SYSTEM}\n\nInstruction: {bare_prompt}"


def mentioned_color(text: str) -> list[str]:
    return [c for c in ("blue", "green", "red") if c in text.lower().split()]


def resolve(prompt: str, layout: list[str]) -> tuple[str, str] | None:
    text = prompt.lower()
    colors = ("blue", "green", "red")
    if "not " in text or "neither" in text or "non-green" in text:
        if "not green and not blue" in text or "not blue and not green" in text:
            color = "red"
        elif "neither green nor blue" in text or "non-green and non-blue" in text:
            color = "red"
        else:
            return None
        return color, SLOTS[layout.index(color)]
    if "left of" in text or "right of" in text or "right of the red" in text:
        ref = None
        for color in colors:
            if color in text:
                ref = color
        if ref is None:
            return None
        ref_idx = layout.index(ref)
        target_idx = ref_idx - 1 if "left of" in text else ref_idx + 1
        if target_idx < 0 or target_idx >= len(layout):
            return None
        return layout[target_idx], SLOTS[target_idx]
    if "1st bowl from the left" in text or "first bowl from the left" in text or "3rd bowl from the right" in text:
        return layout[0], "left"
    if (
        "2nd bowl from the left" in text
        or "second bowl from the left" in text
        or "2nd bowl from the right" in text
        or "second bowl from the right" in text
        or "middle bowl" in text
        or "center bowl" in text
    ):
        return layout[1], "middle"
    if "3rd bowl from the left" in text or "third bowl from the left" in text or "1st bowl from the right" in text:
        return layout[2], "right"
    for color in colors:
        if color in text:
            return color, SLOTS[layout.index(color)]
    return None


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
        start = int(row["episode_start"])
        for episode_index in range(start, int(row["episode_end"]) + 1):
            truth[episode_index] = {
                "combo": str(row["combo"]),
                "layout_code": str(row["bowl_order_left_to_right"]),
                "layout_left_to_right": layout,
                "replicate_index": episode_index - start,
            }
    return truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--out-dir", type=Path, default=Path("data/stage1b_probe"))
    parser.add_argument("--image-quality", type=int, default=95)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.dataset)
    truth = load_episode_truth(args.dataset)
    rows = []
    for ep in dataset.meta.episodes:
        episode_index = int(ep["episode_index"])
        first_index = int(ep["dataset_from_index"])
        item = dataset[first_index]
        info = truth[episode_index]
        image_path = image_dir / f"ep{episode_index:03d}_{info['combo']}_frame000.jpg"
        to_pil_image(item["observation.images.front"]).save(image_path, quality=args.image_quality)
        for prompt_type, bare_prompt in PROMPTS:
            resolved = resolve(bare_prompt, info["layout_left_to_right"])
            if resolved is None:
                continue
            target_color, target_slot = resolved
            rows.append(
                {
                    "image": str(image_path),
                    "global_index": first_index,
                    "orig_episode_index": episode_index,
                    "episode_index": episode_index,
                    "layout_id": "-".join(info["layout_left_to_right"]),
                    "layout_code": info["layout_code"],
                    "phase": "initial",
                    "bare_prompt": bare_prompt,
                    "scaffold_prompt": scaffold_prompt(bare_prompt),
                    "prompt_type": prompt_type,
                    "layout_left_to_right": info["layout_left_to_right"],
                    "target_color": target_color,
                    "target_slot": target_slot,
                    "split": split_for_replicate(int(info["replicate_index"])),
                    "replicate_index": info["replicate_index"],
                    "combo": info["combo"],
                }
            )

    jsonl = args.out_dir / "stage1b_probe.jsonl"
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    metadata = {
        "dataset": args.dataset,
        "model_target": "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged",
        "frame_policy": "first frame per real episode",
        "split_policy": "replicate_index 0-3 train, 4 val, 5 test inside each physical layout/target combo",
        "num_rows": len(rows),
        "num_images": len(list(image_dir.glob("*.jpg"))),
        "counts": {
            "split": dict(Counter(row["split"] for row in rows)),
            "prompt_type": dict(Counter(row["prompt_type"] for row in rows)),
            "target_slot": dict(Counter(row["target_slot"] for row in rows)),
            "layout_id": dict(Counter(row["layout_id"] for row in rows)),
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(rows)} rows to {jsonl}")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
