#!/usr/bin/env python3
"""Evaluate a SmolVLM2 VQA checkpoint on Eval-2 bowl prompt variations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


SYSTEM = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls, then answer with only the target bowl color."
)


def parse_color(text: str) -> str | None:
    text = text.strip().lower()
    for color in ("blue", "red", "green"):
        if re.search(rf"\b{color}\b", text):
            return color
    return None


def load_rows(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    return rows


def make_prompt(processor, question: str) -> str:
    prompt_text = f"{SYSTEM}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="robot-learning-group47/smolvlm2-500m-vqa-lora-merged")
    parser.add_argument("--jsonl", default="data/eval2_vqa_fresh/vqa.jsonl")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out", default="logs/eval_hf_vqa_variations.json")
    parser.add_argument("--failures", type=int, default=40)
    args = parser.parse_args()

    rows = load_rows(Path(args.jsonl), args.split)
    if not rows:
        raise SystemExit(f"No rows found for split={args.split}")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading {args.model} on {device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    model.eval()

    by_type: dict[str, list[bool]] = defaultdict(list)
    by_layout: dict[str, list[bool]] = defaultdict(list)
    by_split: dict[str, list[bool]] = defaultdict(list)
    failures: list[dict] = []
    correct = 0

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [make_prompt(processor, row["question"]) for row in batch]
        images = [[Image.open(row["image"]).convert("RGB")] for row in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(output[:, input_len:], skip_special_tokens=True)

        for row, raw in zip(batch, decoded, strict=True):
            pred = parse_color(raw)
            ok = pred == row["target"]
            correct += int(ok)
            by_type[row["type"]].append(ok)
            by_layout[row["layout"]].append(ok)
            by_split[row["split"]].append(ok)
            if not ok:
                failures.append(
                    {
                        "image": Path(row["image"]).name,
                        "layout": row["layout"],
                        "type": row["type"],
                        "question": row["question"],
                        "target": row["target"],
                        "pred": pred,
                        "raw": raw.strip(),
                    }
                )

        done = min(start + args.batch_size, len(rows))
        print(f"  {done:4d}/{len(rows)}", end="\r", flush=True)

    def summarize(groups: dict[str, list[bool]]) -> dict[str, dict]:
        return {
            key: {"correct": sum(vals), "total": len(vals), "accuracy": sum(vals) / len(vals)}
            for key, vals in sorted(groups.items())
        }

    report = {
        "model": args.model,
        "jsonl": str(args.jsonl),
        "split": args.split,
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "counts": {
            "type": dict(Counter(row["type"] for row in rows)),
            "layout": dict(Counter(row["layout"] for row in rows)),
            "split": dict(Counter(row["split"] for row in rows)),
        },
        "by_type": summarize(by_type),
        "by_layout": summarize(by_layout),
        "by_split": summarize(by_split),
        "failures": failures[: args.failures],
        "num_failures": len(failures),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print()
    print(f"Overall: {correct}/{len(rows)} = {correct / len(rows) * 100:.1f}%")
    print("By type:")
    for key, stats in report["by_type"].items():
        print(f"  {key:8s}: {stats['correct']}/{stats['total']} = {stats['accuracy'] * 100:.1f}%")
    print("By layout:")
    for key, stats in report["by_layout"].items():
        print(f"  {key:3s}: {stats['correct']}/{stats['total']} = {stats['accuracy'] * 100:.1f}%")
    print(f"Failures saved: {len(failures)} total, first {len(report['failures'])} in {out}")


if __name__ == "__main__":
    main()
