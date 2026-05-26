#!/usr/bin/env python3
"""Evaluate eval3_yann_left_repair_adapter on eval3_position_qa_3frames with a new prompt."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

CHECKPOINT = Path("/home/shadeform/outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_vlm_lora")
DATASET = Path("/home/shadeform/data/eval3_position_qa_3frames/all.jsonl")
BATCH_SIZE = 16


def build_prompt(name: str) -> str:
    return (
        f"Place the Coke on {name}, there are three celebrity cards in this robot scene, "
        f"arranged in the left, middle, and right slots. "
        f"Which slot is {name} in? "
        f"Answer with exactly one word: left, center, or right."
    )


def build_text(processor, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def normalize(text: str) -> str:
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in {"left", "middle", "center", "centre", "right"}:
            return "middle" if word in {"center", "centre"} else word
    return "unknown"


def main():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    peft_config = PeftConfig.from_pretrained(CHECKPOINT)
    processor_src = CHECKPOINT if (CHECKPOINT / "preprocessor_config.json").exists() else peft_config.base_model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_src, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        peft_config.base_model_name_or_path, torch_dtype=dtype, device_map="auto", local_files_only=True
    )
    model = PeftModel.from_pretrained(base, CHECKPOINT, local_files_only=True)
    model.eval()
    device = next(model.parameters()).device

    records = [json.loads(l) for l in DATASET.read_text().splitlines()]
    results = []

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        images = [Image.open(r["image"]).convert("RGB") for r in batch]
        texts = [build_text(processor, build_prompt(r["target_person"])) for r in batch]
        inputs = processor(text=texts, images=[[img] for img in images], return_tensors="pt", padding=True)
        inputs = {k: (v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device)) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)

        prompt_len = inputs["input_ids"].shape[-1]
        raw_outputs = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

        for rec, raw in zip(batch, raw_outputs):
            pred = normalize(raw.strip())
            results.append({**rec, "pred_raw": raw.strip(), "pred": pred, "correct": pred == rec["ground_truth"]})

        print(f"  {min(start + BATCH_SIZE, len(records))}/{len(records)}", end="\r", flush=True)

    print()

    # Overall
    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\nOverall: {correct}/{total} = {correct/total:.1%}\n")

    # By person
    print("--- By person ---")
    by_person = defaultdict(list)
    for r in results:
        by_person[r["target_person"]].append(r)
    for person, rows in sorted(by_person.items()):
        c = sum(r["correct"] for r in rows)
        preds = Counter(r["pred"] for r in rows)
        print(f"  {person}: {c}/{len(rows)} = {c/len(rows):.1%}  preds={dict(preds)}")

    # By person x ground_truth
    print("\n--- By person x ground truth ---")
    by_person_gt = defaultdict(list)
    for r in results:
        by_person_gt[(r["target_person"], r["ground_truth"])].append(r)
    for (person, gt), rows in sorted(by_person_gt.items()):
        c = sum(r["correct"] for r in rows)
        preds = Counter(r["pred"] for r in rows)
        print(f"  {person:13s} gt={gt:6s}: {c}/{len(rows)} = {c/len(rows):.1%}  preds={dict(preds)}")

    # Save
    out = Path("/home/shadeform/outputs/eval_new_prompt_cycle05_vlm_lora.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
