#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVisionTextModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVisionTextModel


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_answer(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    for word in words:
        if word in {"left", "middle", "center", "centre", "right"}:
            return "middle" if word in {"center", "centre"} else word
    return "unknown"


def build_prompt(processor, prompt: str) -> str:
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


def format_prompt(template: str, row: dict) -> str:
    name = row.get("target_person") or row.get("name") or row.get("person")
    if not name:
        raise KeyError(f"Row is missing target_person/name/person: {row}")
    return template.format(name=name)


def input_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


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


def load_adapter(checkpoint: Path, base_model: str | None, dtype: torch.dtype):
    peft_config = PeftConfig.from_pretrained(checkpoint)
    base = base_model or peft_config.base_model_name_or_path
    if not base:
        raise RuntimeError("Provide --base-model or use a LoRA with base_model_name_or_path.")

    processor_source = checkpoint if (checkpoint / "preprocessor_config.json").exists() else base
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    try:
        base_loaded = AutoVisionTextModel.from_pretrained(
            base,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )
    except TypeError:
        base_loaded = AutoVisionTextModel.from_pretrained(
            base,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )
    model = PeftModel.from_pretrained(base_loaded, checkpoint, local_files_only=True)
    model.eval()
    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="LoRA adapter path.")
    parser.add_argument("--lora-path", type=Path, default=None, help="Alias for --checkpoint.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data", type=Path, default=Path("/home/shadeform/data/eval3_position_qa_3frames/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/home/shadeform/data/eval3_position_qa_3frames/cotrain_predictions.jsonl"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional prompt template. May reference {name}; otherwise uses each JSONL row's prompt.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.lora_path
    if checkpoint is None:
        raise SystemExit("Pass --checkpoint or --lora-path.")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor, model = load_adapter(checkpoint, args.base_model, dtype)
    device = input_device(model)
    rows = load_jsonl(args.data)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    predictions = []
    correct_examples = []
    incorrect_examples = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            images = [Image.open(row["image"]).convert("RGB") for row in batch]
            prompts = [
                format_prompt(args.prompt_template, row) if args.prompt_template else row["prompt"]
                for row in batch
            ]
            texts = [build_prompt(processor, prompt) for prompt in prompts]
            inputs = processor(text=texts, images=[[image] for image in images], return_tensors="pt", padding=True)
            inputs = move_inputs(inputs, device, dtype)
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            prompt_len = inputs["input_ids"].shape[-1]
            outputs = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            for row, prompt, raw in zip(batch, prompts, outputs, strict=True):
                prediction = normalize_answer(raw)
                expected = row.get("answer") or row.get("ground_truth")
                record = {
                    **row,
                    "eval_prompt": prompt,
                    "prediction_raw": raw.strip(),
                    "prediction": prediction,
                    "expected": expected,
                    "correct": prediction == expected,
                }
                predictions.append(record)
                if record["correct"] and len(correct_examples) < 5:
                    correct_examples.append(record)
                if not record["correct"] and len(incorrect_examples) < 10:
                    incorrect_examples.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in predictions:
            f.write(json.dumps(record) + "\n")

    correct = sum(record["correct"] for record in predictions)
    print(json.dumps({"examples": len(predictions), "correct": correct, "accuracy": correct / len(predictions), "output": str(args.output)}, indent=2))
    print("Correct examples:")
    for record in correct_examples:
        print(json.dumps({k: record[k] for k in ("image", "expected", "prediction", "prediction_raw")}, indent=2))
    print("Incorrect examples:")
    for record in incorrect_examples:
        print(json.dumps({k: record[k] for k in ("image", "expected", "prediction", "prediction_raw")}, indent=2))


if __name__ == "__main__":
    main()
