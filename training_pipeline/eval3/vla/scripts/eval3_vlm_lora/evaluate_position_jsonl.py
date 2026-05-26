#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from evaluate_position_accuracy import (
    build_inputs,
    load_model,
    model_device,
    move_inputs,
    normalize_answer,
)


def group_stats(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        name: {
            "correct": sum(item["correct"] for item in values),
            "total": len(values),
            "accuracy": sum(item["correct"] for item in values) / len(values),
            "prediction_counts": dict(Counter(item["prediction"] or "?" for item in values)),
        }
        for name, values in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    print(f"Loading {args.checkpoint} dtype={dtype}", flush=True)
    processor, model = load_model(args.checkpoint, dtype)
    device = model_device(model)
    print(f"Model device: {device}; examples: {len(examples)}", flush=True)

    rows = []
    for index, example in enumerate(examples, start=1):
        image = Image.open(example["image"]).convert("RGB")
        inputs = move_inputs(build_inputs(processor, [image], example["prompt"]), device, dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        output = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0].strip()
        pred = normalize_answer(output)
        row = dict(example)
        row.update(
            {
                "prediction_raw": output,
                "prediction": pred,
                "correct": pred == example["ground_truth"],
            }
        )
        rows.append(row)
        if index % 10 == 0 or index == len(examples):
            acc = sum(item["correct"] for item in rows) / len(rows)
            print(f"{index}/{len(examples)} acc={acc:.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "jsonl": str(args.jsonl),
        "examples": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "prediction_counts": dict(Counter(row["prediction"] or "?" for row in rows)),
        "by_target_person": group_stats(rows, "target_person"),
        "by_ground_truth": group_stats(rows, "ground_truth"),
        "by_sample": group_stats(rows, "sample"),
        "output_csv": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
