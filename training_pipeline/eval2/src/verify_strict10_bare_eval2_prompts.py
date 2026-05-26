#!/usr/bin/env python3
"""Evaluate strict10 SmolVLM2 with bare Eval2 prompts only.

Text input is exactly one of the requested Eval2 instructions, with no system
prompt and no output-format instruction.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


MODEL_ID = "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
DATASET_ID = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
COLOR_ABBR = {"b": "blue", "g": "green", "r": "red"}
SLOTS = ("left", "middle", "right")

PROMPTS = [
    # 2nd from left / middle-slot variants.
    "Put the banana into the 2nd bowl from the left from the robot perspective",
    "Put the banana into the second bowl from the left from the robot perspective",
    "Put the banana into the middle bowl from the robot perspective",
    "Put the banana into the center bowl from the robot perspective",
    # Relative-to-red variants.
    "Put the banana into the bowl on the right of the red bowl from the robot perspective",
    "Put the banana into the bowl to the right of the red bowl from the robot perspective",
    "Put the banana into the bowl immediately right of the red bowl from the robot perspective",
    "Put the banana into the bowl just to the right of the red bowl from the robot perspective",
    # Negation variants whose target is red.
    "Put the banana into the bowl that is not green and not blue.",
    "Put the banana into the bowl that is not blue and not green.",
    "Put the banana into the bowl that is neither green nor blue.",
    "Put the banana into the non-green and non-blue bowl.",
]


def parse_output(raw: str) -> dict[str, str]:
    text = raw.lower()
    result: dict[str, str] = {}
    color_match = re.search(r"target\s+color\s*[:\-]\s*(blue|red|green)", text)
    slot_match = re.search(r"target\s+slot\s*[:\-]\s*(left|middle|right)", text)
    if color_match:
        result["target_color"] = color_match.group(1)
    if slot_match:
        result["target_slot"] = slot_match.group(1)
    if "target_color" not in result:
        colors = re.findall(r"\b(blue|red|green)\b", text)
        if len(set(colors)) == 1:
            result["target_color"] = colors[0]
    if "target_slot" not in result:
        slots = re.findall(r"\b(left|middle|right)\b", text)
        if len(set(slots)) == 1:
            result["target_slot"] = slots[0]
    return result


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


def build_prompt(processor, instruction: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    rows = pd.read_csv(map_path)
    truth = {}
    for _, row in rows.iterrows():
        layout = [COLOR_ABBR[ch] for ch in str(row["bowl_order_left_to_right"])]
        for episode_index in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
            truth[episode_index] = {
                "combo": str(row["combo"]),
                "layout_left_to_right": layout,
            }
    return truth


def summarize(groups: dict[str, list[bool]]) -> dict[str, dict]:
    return {
        key: {"correct": int(sum(vals)), "total": len(vals), "accuracy": float(sum(vals) / len(vals))}
        for key, vals in sorted(groups.items())
        if vals
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default="logs/strict10_bare_eval2_prompts.json")
    parser.add_argument("--examples", type=int, default=36)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading dataset {args.dataset} ...")
    dataset = LeRobotDataset(args.dataset)
    truth = load_episode_truth(args.dataset)

    rows = []
    for ep in dataset.meta.episodes:
        episode_index = int(ep["episode_index"])
        first_index = int(ep["dataset_from_index"])
        item = dataset[first_index]
        image = to_pil_image(item["observation.images.front"])
        info = truth[episode_index]
        for prompt in PROMPTS:
            resolved = resolve(prompt, info["layout_left_to_right"])
            if resolved is None:
                continue
            target_color, target_slot = resolved
            rows.append(
                {
                    "episode_index": episode_index,
                    "dataset_index": first_index,
                    "prompt": prompt,
                    "combo": info["combo"],
                    "layout_left_to_right": info["layout_left_to_right"],
                    "target_color": target_color,
                    "target_slot": target_slot,
                    "image": image,
                }
            )

    print(f"Evaluating {len(rows)} bare-prompt first-frame examples on {device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    model.eval()

    records = []
    color_correct = slot_correct = both_correct = 0
    by_prompt_slot: dict[str, list[bool]] = defaultdict(list)
    by_prompt_color: dict[str, list[bool]] = defaultdict(list)

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [build_prompt(processor, row["prompt"]) for row in batch]
        images = [[row["image"]] for row in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        for row, raw in zip(batch, decoded, strict=True):
            parsed = parse_output(raw)
            pred_color = parsed.get("target_color")
            pred_slot = parsed.get("target_slot")
            ok_color = pred_color == row["target_color"]
            ok_slot = pred_slot == row["target_slot"]
            ok_both = ok_color and ok_slot
            color_correct += int(ok_color)
            slot_correct += int(ok_slot)
            both_correct += int(ok_both)
            by_prompt_color[row["prompt"]].append(ok_color)
            by_prompt_slot[row["prompt"]].append(ok_slot)
            records.append(
                {
                    "episode_index": row["episode_index"],
                    "combo": row["combo"],
                    "layout_left_to_right": row["layout_left_to_right"],
                    "prompt": row["prompt"],
                    "target_color": row["target_color"],
                    "target_slot": row["target_slot"],
                    "pred_color": pred_color,
                    "pred_slot": pred_slot,
                    "raw": raw.strip(),
                    "color_correct": ok_color,
                    "slot_correct": ok_slot,
                    "both_correct": ok_both,
                }
            )
        print(f"  {min(start + args.batch_size, len(rows)):4d}/{len(rows)}", end="\r", flush=True)

    total = len(rows)
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "prompts": PROMPTS,
        "total": total,
        "color_accuracy": color_correct / total,
        "slot_accuracy": slot_correct / total,
        "both_accuracy": both_correct / total,
        "counts": {
            "prompt": dict(Counter(row["prompt"] for row in rows)),
            "target_slot": dict(Counter(row["target_slot"] for row in rows)),
        },
        "by_prompt_color": summarize(by_prompt_color),
        "by_prompt_slot": summarize(by_prompt_slot),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print()
    print(f"Color accuracy: {color_correct}/{total} = {100 * color_correct / total:.1f}%")
    print(f"Slot accuracy : {slot_correct}/{total} = {100 * slot_correct / total:.1f}%")
    print(f"Both accuracy : {both_correct}/{total} = {100 * both_correct / total:.1f}%")
    print("By prompt slot accuracy:")
    for prompt, stats in report["by_prompt_slot"].items():
        print(f"  {stats['correct']:3d}/{stats['total']:3d} = {100 * stats['accuracy']:5.1f}%  {prompt}")
    print("Examples:")
    for record in records[: args.examples]:
        print(
            f"  ep {record['episode_index']:03d} {record['combo']:5s} "
            f"gt={record['target_color']}/{record['target_slot']} "
            f"pred={record['pred_color']}/{record['pred_slot']} raw={record['raw']!r}"
        )
    print(f"Saved full report to {out}")


if __name__ == "__main__":
    main()
