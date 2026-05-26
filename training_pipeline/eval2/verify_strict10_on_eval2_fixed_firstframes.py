#!/usr/bin/env python3
"""Evaluate strict10 SmolVLM2 on first frames from eval2_fixed_bowl_fixed_banana_108.

This is intentionally a narrow diagnostic:
one real robot initial frame per episode + that episode's original prompt.
Ground truth comes from the dataset's combo episode map.
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

SYSTEM = (
    "You are looking at a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Given the image and the original robot instruction, identify the target bowl. "
    "Reply with exactly two short lines, not JSON and not prose. "
    "The exact format is: "
    "target color: green\n"
    "target slot: right"
)


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
        if len(colors) == 1:
            result["target_color"] = colors[0]
    if "target_slot" not in result:
        slots = re.findall(r"\b(left|middle|right)\b", text)
        if len(slots) == 1:
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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{SYSTEM}\n\nInstruction: {instruction}"},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_type(prompt: str) -> str:
    text = prompt.lower()
    if "not " in text:
        return "negation"
    if "left of" in text or "right of" in text:
        return "relative"
    if any(word in text for word in ["1st", "2nd", "3rd", "first", "second", "third", "middle"]):
        return "ordinal"
    return "color"


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
        target_color = str(row["target_bowl_color"]).lower()
        target_slot = SLOTS[layout.index(target_color)]
        for episode_index in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
            truth[episode_index] = {
                "combo": str(row["combo"]),
                "layout_left_to_right": layout,
                "target_color": target_color,
                "target_slot": target_slot,
            }
    return truth


def summarize(groups: dict[str, list[bool]]) -> dict[str, dict]:
    return {
        key: {
            "correct": int(sum(values)),
            "total": len(values),
            "accuracy": float(sum(values) / len(values)) if values else 0.0,
        }
        for key, values in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default="logs/strict10_eval2_fixed_firstframes.json")
    parser.add_argument("--examples", type=int, default=24)
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
        prompt = str(item["task"])
        rows.append(
            {
                "episode_index": episode_index,
                "dataset_index": first_index,
                "prompt": prompt,
                "prompt_type": prompt_type(prompt),
                **truth[episode_index],
                "image": to_pil_image(item["observation.images.front"]),
            }
        )

    print(f"Evaluating {len(rows)} first frames on {device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    model.eval()

    records = []
    by_combo_color: dict[str, list[bool]] = defaultdict(list)
    by_combo_slot: dict[str, list[bool]] = defaultdict(list)
    by_layout_color: dict[str, list[bool]] = defaultdict(list)
    by_layout_slot: dict[str, list[bool]] = defaultdict(list)
    by_target_slot: dict[str, list[bool]] = defaultdict(list)
    by_prompt_type_slot: dict[str, list[bool]] = defaultdict(list)
    color_correct = 0
    slot_correct = 0
    both_correct = 0

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
            layout_key = "-".join(row["layout_left_to_right"])
            by_combo_color[row["combo"]].append(ok_color)
            by_combo_slot[row["combo"]].append(ok_slot)
            by_layout_color[layout_key].append(ok_color)
            by_layout_slot[layout_key].append(ok_slot)
            by_target_slot[row["target_slot"]].append(ok_slot)
            by_prompt_type_slot[row["prompt_type"]].append(ok_slot)
            records.append(
                {
                    "episode_index": row["episode_index"],
                    "combo": row["combo"],
                    "layout_left_to_right": row["layout_left_to_right"],
                    "prompt": row["prompt"],
                    "prompt_type": row["prompt_type"],
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
        "total": total,
        "color_accuracy": color_correct / total,
        "slot_accuracy": slot_correct / total,
        "both_accuracy": both_correct / total,
        "color_correct": color_correct,
        "slot_correct": slot_correct,
        "both_correct": both_correct,
        "counts": {
            "combo": dict(Counter(row["combo"] for row in rows)),
            "target_slot": dict(Counter(row["target_slot"] for row in rows)),
            "prompt_type": dict(Counter(row["prompt_type"] for row in rows)),
        },
        "by_combo_color": summarize(by_combo_color),
        "by_combo_slot": summarize(by_combo_slot),
        "by_layout_color": summarize(by_layout_color),
        "by_layout_slot": summarize(by_layout_slot),
        "by_target_slot": summarize(by_target_slot),
        "by_prompt_type_slot": summarize(by_prompt_type_slot),
        "records": records,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print()
    print(f"Color accuracy: {color_correct}/{total} = {100 * color_correct / total:.1f}%")
    print(f"Slot accuracy : {slot_correct}/{total} = {100 * slot_correct / total:.1f}%")
    print(f"Both accuracy : {both_correct}/{total} = {100 * both_correct / total:.1f}%")
    print("By target slot:")
    for key, stats in report["by_target_slot"].items():
        print(f"  {key:6s}: {stats['correct']}/{stats['total']} = {100 * stats['accuracy']:.1f}%")
    print("Examples:")
    for record in records[: args.examples]:
        print(
            f"  ep {record['episode_index']:03d} {record['combo']:5s} "
            f"gt={record['target_color']}/{record['target_slot']} "
            f"pred={record['pred_color']}/{record['pred_slot']} "
            f"raw={record['raw']!r}"
        )
    print(f"Saved full report to {out}")


if __name__ == "__main__":
    main()
