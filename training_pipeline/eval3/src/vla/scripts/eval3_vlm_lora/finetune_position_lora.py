#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

try:
    from transformers import AutoModelForImageTextToText as AutoVisionTextModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVisionTextModel


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(prompt: str, answer: str | None = None) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def chat_text(processor, prompt: str, answer: str | None) -> str:
    messages = build_messages(prompt, answer)
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=answer is None,
        )
    return prompt if answer is None else f"{prompt}\n{answer}"


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


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


def collate_examples(processor, examples: list[dict], device: torch.device, dtype: torch.dtype):
    images = [load_image(row["image"]) for row in examples]
    full_texts = [chat_text(processor, row["prompt"], row["answer"]) for row in examples]
    prompt_texts = [chat_text(processor, row["prompt"], None) for row in examples]

    full = processor(
        text=full_texts,
        images=[[image] for image in images],
        return_tensors="pt",
        padding=True,
    )
    prompt = processor(
        text=prompt_texts,
        images=[[image] for image in images],
        return_tensors="pt",
        padding=True,
    )

    labels = full["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    for i in range(labels.shape[0]):
        prompt_len = int(prompt["attention_mask"][i].sum().item())
        labels[i, :prompt_len] = -100
    if pad_id is not None:
        labels[labels == pad_id] = -100
    labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels

    return move_inputs(full, device, dtype)


def normalize_answer(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    for word in words:
        if word in {"left", "middle", "center", "centre", "right"}:
            return "middle" if word in {"center", "centre"} else word
    return ""


def model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def load_trainable_lora(checkpoint: Path, dtype: torch.dtype):
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

    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True, local_files_only=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return processor, model


def batched(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def evaluate(model, processor, rows: list[dict], batch_size: int, dtype: torch.dtype, output_csv: Path) -> dict:
    model.eval()
    device = model_device(model)
    results = []

    with torch.inference_mode():
        for batch in tqdm(list(batched(rows, batch_size)), desc="eval"):
            images = [load_image(row["image"]) for row in batch]
            texts = [chat_text(processor, row["prompt"], None) for row in batch]
            inputs = processor(text=texts, images=[[image] for image in images], return_tensors="pt", padding=True)
            inputs = move_inputs(inputs, device, dtype)
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            prompt_len = inputs["input_ids"].shape[-1]
            outputs = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            for row, output in zip(batch, outputs, strict=True):
                prediction = normalize_answer(output)
                results.append(
                    {
                        **row,
                        "prediction_raw": output.strip(),
                        "prediction": prediction,
                        "correct": prediction == row["answer"],
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    correct = sum(row["correct"] for row in results)
    grouped = {}
    for key in ("target_person", "ground_truth"):
        groups = defaultdict(list)
        for row in results:
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

    return {
        "examples": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "grouped": grouped,
        "output_csv": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/shadeform/data/eval3_position_qa_3frames"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/shadeform/checkpoint/eval3_phase2_TOY"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/shadeform/outputs/vlm_lora/eval3_phase2_TOY_position_qa_3frames"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-steps", type=int, default=0, help="Debug limit. 0 means full training.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    eval_rows = load_jsonl(args.data_dir / "eval.jsonl")
    processor, model = load_trainable_lora(args.checkpoint, dtype)
    device = model_device(model)

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    total_update_steps = (steps_per_epoch * args.epochs + args.grad_accum - 1) // args.grad_accum
    if args.max_steps:
        total_update_steps = min(total_update_steps, args.max_steps)
    warmup_steps = max(1, int(total_update_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "train_examples": len(train_rows),
                "eval_examples": len(eval_rows),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "total_update_steps": total_update_steps,
                "device": str(device),
                "dtype": str(dtype),
            },
            indent=2,
        )
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    global_step = 0
    accum_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        progress = tqdm(list(batched(train_rows, args.batch_size)), desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            inputs = collate_examples(processor, batch, device, dtype)
            outputs = model(**inputs)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_step += 1

            if accum_step % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}", step=global_step)
                if args.max_steps and global_step >= args.max_steps:
                    break

        model.save_pretrained(args.output_dir / f"epoch_{epoch + 1:02d}")
        processor.save_pretrained(args.output_dir / f"epoch_{epoch + 1:02d}")
        if args.max_steps and global_step >= args.max_steps:
            break

    if accum_step % args.grad_accum:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    eval_summary = evaluate(
        model,
        processor,
        eval_rows,
        batch_size=max(1, min(8, args.batch_size * 4)),
        dtype=dtype,
        output_csv=args.output_dir / "eval_predictions.csv",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "global_step": global_step,
        "eval": eval_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
