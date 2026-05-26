#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVisionTextModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVisionTextModel


def load_video_frames(video_path: Path, seconds: float | None) -> list[tuple[int, float, Image.Image]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = 1 if seconds is None else None
    if seconds is not None and fps and fps > 0:
        max_frames = int(seconds * fps + 1e-6)

    frames = []
    frame_index = 0
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = frame_index / fps if fps and fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if seconds is not None and max_frames is None and timestamp_s >= seconds:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((frame_index, timestamp_s, Image.fromarray(rgb)))
        frame_index += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    return frames


def build_prompt(person: str) -> str:
    return f"Is {person} on the left, middle, or right? Answer only one word: left, middle, or right."


def build_text(processor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def build_inputs(processor, images: list[Image.Image], prompt: str):
    text = build_text(processor, prompt)
    return processor(text=[text] * len(images), images=[[image] for image in images], return_tensors="pt", padding=True)


def move_inputs(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def load_model(checkpoint: Path, dtype: torch.dtype):
    peft_config = PeftConfig.from_pretrained(checkpoint)
    base_model = peft_config.base_model_name_or_path
    if not base_model:
        raise RuntimeError(f"{checkpoint}/adapter_config.json has no base_model_name_or_path")

    processor_source = checkpoint if (checkpoint / "preprocessor_config.json").exists() else base_model
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True)
    try:
        base = AutoVisionTextModel.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )
    except TypeError:
        base = AutoVisionTextModel.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )
    model = PeftModel.from_pretrained(base, checkpoint, local_files_only=True)
    model.eval()
    return processor, model


def model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def normalize_answer(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    for word in words:
        if word in {"left", "middle", "center", "centre", "right"}:
            return "middle" if word in {"center", "centre"} else word
    return ""


def target_person_for_episode(episode_index: int) -> str:
    if 0 <= episode_index <= 29:
        return "Taylor Swift"
    if 30 <= episode_index <= 59:
        return "Barack Obama"
    if 60 <= episode_index <= 89:
        return "Yann LeCun"
    raise ValueError(f"Unsupported episode index: {episode_index}")


def ground_truth_for_episode(episode_index: int) -> str:
    offset = episode_index % 30
    if offset < 10:
        return "left"
    if offset < 20:
        return "middle"
    return "right"


def sample_start_middle_end(frames: list[tuple[int, float, Image.Image]]) -> list[tuple[int, float, Image.Image]]:
    if len(frames) <= 3:
        return frames
    indices = [0, len(frames) // 2, len(frames) - 1]
    return [frames[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/shadeform/data/eval3_90_permutation"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/shadeform/checkpoint/eval3_phase2_TOY"))
    parser.add_argument("--output", type=Path, default=Path("/home/shadeform/outputs/eval3_phase2_toy_position_accuracy.csv"))
    parser.add_argument("--seconds", type=float, default=None, help="Evaluate every frame before this time. Default: first frame only.")
    parser.add_argument("--sample-start-middle-end", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor, model = load_model(args.checkpoint, dtype)
    device = model_device(model)

    rows = []

    for episode in range(90):
        person = target_person_for_episode(episode)
        video_path = args.dataset / "videos" / "observation.images.front" / "chunk-000" / f"file-{episode:03d}.mp4"
        frames = load_video_frames(video_path, args.seconds)
        if args.sample_start_middle_end:
            frames = sample_start_middle_end(frames)
        prompt = build_prompt(person)
        gt = ground_truth_for_episode(episode)
        episode_rows = []

        for start in range(0, len(frames), args.batch_size):
            batch = frames[start : start + args.batch_size]
            frame_indices = [frame_index for frame_index, _, _ in batch]
            timestamps = [timestamp_s for _, timestamp_s, _ in batch]
            images = [image for _, _, image in batch]
            inputs = move_inputs(build_inputs(processor, images, prompt), device, dtype)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            outputs = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            for frame_index, timestamp_s, output in zip(frame_indices, timestamps, outputs, strict=True):
                output = output.strip()
                pred = normalize_answer(output)
                row = {
                    "episode_index": episode,
                    "frame_index": frame_index,
                    "timestamp_s": f"{timestamp_s:.6f}",
                    "target_person": person,
                    "prompt": prompt,
                    "prediction_raw": output,
                    "prediction": pred,
                    "ground_truth": gt,
                    "correct": pred == gt,
                }
                rows.append(row)
                episode_rows.append(row)

        episode_correct = sum(row["correct"] for row in episode_rows)
        preds = Counter(row["prediction"] or "?" for row in episode_rows)
        print(
            f"{episode:02d} {person:13s} gt={gt:6s} "
            f"{episode_correct}/{len(episode_rows)} acc={episode_correct / len(episode_rows):.3f} "
            f"preds={dict(preds)}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    grouped = {}
    for key in ("target_person", "ground_truth"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        grouped[key] = {
            name: {
                "correct": sum(row["correct"] for row in group_rows),
                "total": len(group_rows),
                "accuracy": sum(row["correct"] for row in group_rows) / len(group_rows),
                "prediction_counts": dict(Counter(row["prediction"] or "?" for row in group_rows)),
            }
            for name, group_rows in sorted(groups.items())
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "ground_truth_source": "user-provided episode blocks: 0-9/30-39/60-69 left, 10-19/40-49/70-79 middle, 20-29/50-59/80-89 right",
        "seconds": args.seconds,
        "sample_start_middle_end": args.sample_start_middle_end,
        "batch_size": args.batch_size,
        "examples": total,
        "correct": correct,
        "accuracy": correct / total,
        "grouped": grouped,
        "output_csv": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
